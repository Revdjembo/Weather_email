#!/usr/bin/env python3
"""
Automated Ecowitt Weather Station Data Processor - Matplotlib Windrose Edition

- Downloads previous day's Ecowitt data (with optional catch-up)
- Stores daily + hourly data in SQLite
- Creates:
    * Minimal temperature strip PNG
    * Matplotlib windrose PNG (uses `windrose` if installed; fallback otherwise)
- Sends an HTML email:
    * Metrics grid
    * Rainfall "beaker" widget (HTML/CSS)
    * Inline windrose image (cid:windrose)
    * Temperature strip attached

Environment variables (GitHub Actions friendly):
- ECOWITT_API_KEY, ECOWITT_APP_KEY, ECOWITT_MAC
- GMAIL_EMAIL, GMAIL_PASSWORD (or GMAIL_APP_PASSWORD)
- RECEIVER_EMAILS  (comma-separated)
"""

import os
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np

import matplotlib
# Non-interactive backend in CI/headless
if os.getenv('GITHUB_ACTIONS'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib

from scipy.interpolate import interp1d


# --------------------------- Processor ---------------------------

class EcowittWeatherProcessor:
    def __init__(self, config_file: str = 'weather_config.json'):
        """Initialize the processor."""
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.setup_database()
        self.__init_temperature_chart__()

    # --------------------------- Config & Setup ---------------------------

    def load_config(self, config_file: str = 'weather_config.json'):
        """Load configuration from JSON file or environment variables."""
        if os.getenv('GITHUB_ACTIONS') or os.getenv('ECOWITT_API_KEY'):
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger = logging.getLogger(__name__)
            logger.info("Running in cloud environment - using environment variables")

            # Accept either GMAIL_PASSWORD or GMAIL_APP_PASSWORD
            gmail_password = os.getenv('GMAIL_PASSWORD') or os.getenv('GMAIL_APP_PASSWORD') or ''

            cloud_config = {
                "ecowitt": {
                    "api_key": os.getenv('ECOWITT_API_KEY', ''),
                    "application_key": os.getenv('ECOWITT_APP_KEY', ''),
                    "mac": os.getenv('ECOWITT_MAC', 'F0:F5:BD:8A:FA:9C'),
                    "base_url": "https://api.ecowitt.net/api/v3"
                },
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": os.getenv('GMAIL_EMAIL', 'revdjem@gmail.com'),
                    "sender_password": gmail_password,
                    "receiver_email": os.getenv('RECEIVER_EMAILS', 'revdjem@gmail.com').split(',')
                },
                "data": {
                    "database_file": "weather_data.db",
                    "charts_directory": "weather_charts"
                }
            }

            required = {
                'ECOWITT_API_KEY': cloud_config['ecowitt']['api_key'],
                'ECOWITT_APP_KEY': cloud_config['ecowitt']['application_key'],
                'GMAIL_EMAIL': cloud_config['email']['sender_email'],
                'GMAIL_PASSWORD/GMAIL_APP_PASSWORD': gmail_password,
            }
            missing = [k for k, v in required.items() if not v]
            if missing:
                logger.error(f"Missing required environment variables: {missing}")
                raise ValueError(f"Missing required environment variables: {missing}")

            return cloud_config

        # Local file config if not running in cloud / env mode
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Local sensible defaults (edit as needed)
        return {
            "ecowitt": {
                "api_key": "",
                "application_key": "",
                "mac": "F0:F5:BD:8A:FA:9C",
                "base_url": "https://api.ecowitt.net/api/v3"
            },
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "you@example.com",
                "sender_password": "",
                "receiver_email": ["you@example.com"]
            },
            "data": {
                "database_file": "weather_data.db",
                "charts_directory": "weather_charts"
            }
        }

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('weather_processor.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Create/open SQLite DB and tables."""
        db_file = self.config['data']['database_file']
        self.conn = sqlite3.connect(db_file)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                date TEXT PRIMARY KEY,
                avg_temp REAL, max_temp REAL, min_temp REAL,
                total_rainfall REAL, avg_wind_speed REAL, max_wind_speed REAL,
                avg_wind_direction REAL, avg_humidity REAL, avg_pressure REAL
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS hourly_data (
                datetime TEXT PRIMARY KEY,
                date TEXT,
                temperature REAL, humidity REAL, pressure REAL,
                wind_speed REAL, wind_direction REAL, rainfall REAL,
                FOREIGN KEY (date) REFERENCES daily_data (date)
            )
        ''')
        self.conn.commit()

    # --------------------------- Temperature Chart Colors ---------------------------

    def __init_temperature_chart__(self):
        """Initialize the color interpolation used for the temperature strip."""
        self.TEMPERATURE_COLOR_MAP = [
            (-9, '#435897'), (-6, '#1d92c1'), (-3, '#60c3c1'), (0, '#7fcebc'),
            (3, '#91d5ba'), (6, '#cfebb2'), (9, '#e3ecab'), (12, '#ffe796'),
            (15, '#ffc96c'), (18, '#ffb34c'), (21, '#f67639'), (24, '#c30031'), (27, '#3a000e')
        ]
        self.setup_temperature_color_interpolation()

    def setup_temperature_color_interpolation(self):
        """Build linear RGB interpolators from the color map."""
        temps = [t for t, _ in self.TEMPERATURE_COLOR_MAP]
        rgb_colors = []
        for _, hex_color in self.TEMPERATURE_COLOR_MAP:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgb_colors.append(rgb)

        r_values = [c[0] for c in rgb_colors]
        g_values = [c[1] for c in rgb_colors]
        b_values = [c[2] for c in rgb_colors]

        self.temp_r_interp = interp1d(temps, r_values, kind='linear', bounds_error=False, fill_value='extrapolate')
        self.temp_g_interp = interp1d(temps, g_values, kind='linear', bounds_error=False, fill_value='extrapolate')
        self.temp_b_interp = interp1d(temps, b_values, kind='linear', bounds_error=False, fill_value='extrapolate')

    def get_temperature_color(self, temp: float) -> str:
        """Return an interpolated hex color for a given temperature (°C)."""
        try:
            r = max(0, min(255, int(self.temp_r_interp(temp))))
            g = max(0, min(255, int(self.temp_g_interp(temp))))
            b = max(0, min(255, int(self.temp_b_interp(temp))))
            return f'#{r:02x}{g:02x}{b:02x}'
        except Exception as e:
            self.logger.warning(f"Error mapping temperature color for {temp}: {e}")
            return '#808080'

    # --------------------------- Ecowitt API ---------------------------

    def test_realtime_connection(self) -> bool:
        """Ping the real-time endpoint to verify credentials are OK."""
        try:
            url = f"{self.config['ecowitt']['base_url']}/device/real_time"
            params = {
                'application_key': self.config['ecowitt']['application_key'],
                'api_key': self.config['ecowitt']['api_key'],
                'mac': self.config['ecowitt']['mac'],
                'call_back': 'all'
            }
            self.logger.info("Testing real-time API connectivity…")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            ok = isinstance(data, dict) and data.get('code') == 0
            self.logger.info("Real-time API: OK" if ok else f"Real-time API: FAILED {data}")
            return ok
        except Exception as e:
            self.logger.error(f"Real-time connection error: {e}")
            return False

    def get_historical_data_all_sensors(self, start_date_str: str, end_date_str: str, cycle_type: str = '30min'):
        """Fetch historical data for several sensor groups between dates."""
        try:
            base_params = {
                'application_key': self.config['ecowitt']['application_key'],
                'api_key': self.config['ecowitt']['api_key'],
                'mac': self.config['ecowitt']['mac'],
                'start_date': start_date_str,
                'end_date': end_date_str,
                'cycle_type': cycle_type
            }
            url = f"{self.config['ecowitt']['base_url']}/device/history"
            sensor_categories = ['outdoor', 'indoor', 'pressure', 'wind', 'rainfall']
            all_sensor_data = {}

            for category in sensor_categories:
                try:
                    params = {**base_params, 'call_back': category}
                    self.logger.info(f"Fetching {category} history {start_date_str}→{end_date_str} ({cycle_type})…")
                    r = requests.get(url, params=params, timeout=60)
                    r.raise_for_status()
                    data = r.json()
                    if data.get('code') == 0 and data.get('data'):
                        all_sensor_data[category] = data['data']
                    else:
                        self.logger.warning(f"No {category} data (msg={data.get('msg')})")
                except Exception as e:
                    self.logger.error(f"{category} fetch error: {e}")

            return all_sensor_data if all_sensor_data else None
        except Exception as e:
            self.logger.error(f"Historical fetch error: {e}")
            return None

    def convert_historical_to_dataframe(self, historical_data) -> Optional[pd.DataFrame]:
        """Flatten the Ecowitt nested dict into a time-indexed DataFrame."""
        try:
            all_records = []

            for category, category_data in historical_data.items():
                if not isinstance(category_data, dict):
                    continue
                sensors = category_data.get(category, category_data)
                for sensor_name, sensor_data in sensors.items():
                    if not isinstance(sensor_data, dict) or 'list' not in sensor_data:
                        continue
                    unit = sensor_data.get('unit', '')
                    for ts_str, val_str in sensor_data['list'].items():
                        try:
                            dt = datetime.fromtimestamp(int(ts_str))
                        except Exception:
                            continue
                        rec = next((r for r in all_records if r['datetime'] == dt), None)
                        if rec is None:
                            rec = {'datetime': dt, 'date': dt.date()}
                            all_records.append(rec)
                        col = f"{category}_{sensor_name}"
                        try:
                            rec[col] = float(val_str)
                            rec[f"{col}_unit"] = unit
                        except Exception:
                            pass

            if not all_records:
                self.logger.warning("No records produced from historical data")
                return None

            df = pd.DataFrame(all_records).sort_values('datetime')

            # Convert Fahrenheit → Celsius if looks like Fahrenheit
            temp_cols = [c for c in df.columns if 'temperature' in c and not c.endswith('_unit')]
            for col in temp_cols:
                if col in df.columns and pd.to_numeric(df[col], errors='coerce').mean() > 50:
                    df[col] = (df[col] - 32) * 5 / 9
            return df
        except Exception as e:
            self.logger.error(f"Historical convert error: {e}")
            return None

    def get_ecowitt_data(self, date: datetime) -> Optional[pd.DataFrame]:
        """Try multiple cycle sizes and return the first usable DataFrame for the date."""
        try:
            if not self.test_realtime_connection():
                return None
            start_date_str = date.strftime('%Y-%m-%d')
            end_date_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            for cycle in ['5min', '30min', '1hour']:
                hist = self.get_historical_data_all_sensors(start_date_str, end_date_str, cycle)
                if hist:
                    df = self.convert_historical_to_dataframe(hist)
                    if df is not None and not df.empty:
                        self.logger.info(f"Got {len(df)} records for {start_date_str} ({cycle})")
                        return df
            self.logger.error("All historical attempts failed")
            return None
        except Exception as e:
            self.logger.error(f"get_ecowitt_data error: {e}")
            return None

    # --------------------------- Daily Stats & Persistence ---------------------------

    def process_daily_stats(self, df: pd.DataFrame, date: datetime) -> dict:
        """Compute daily aggregates from hourly/raw records."""
        try:
            def extract_numeric(series, default=0.0):
                out = []
                for v in series:
                    val = None
                    if isinstance(v, dict):
                        for k in ('value', 'val', 'data', 'current'):
                            if k in v:
                                try:
                                    val = float(v[k])
                                    break
                                except Exception:
                                    pass
                    elif isinstance(v, (int, float)) and not (pd.isna(v) or np.isnan(v)):
                        val = float(v)
                    elif isinstance(v, str):
                        try:
                            val = float(v)
                        except Exception:
                            pass
                    if val is not None and not (pd.isna(val) or np.isnan(val)):
                        out.append(val)
                return out if out else [default]

            def maybe_f_to_c(values):
                if not values:
                    return values
                avg = sum(values) / len(values)
                mn, mx = min(values), max(values)
                if (avg > 40) or (mn > 32) or (mx > 85):  # smells like Fahrenheit
                    return [(t - 32) * 5 / 9 for t in values]
                return values

            # Identify columns
            temp_cols = [c for c in df.columns if any(t in c.lower() for t in ['temperature', 'temp']) and not c.endswith('_unit')]
            outdoor_temp_col = next((c for c in temp_cols if 'outdoor' in c.lower()), temp_cols[0] if temp_cols else None)
            hum_cols = [c for c in df.columns if 'humidity' in c.lower() and not c.endswith('_unit')]
            pres_cols = [c for c in df.columns if 'pressure' in c.lower() and not c.endswith('_unit')]
            ws_cols = [c for c in df.columns if 'wind_speed' in c.lower() and not c.endswith('_unit')]
            wg_cols = [c for c in df.columns if any(x in c.lower() for x in ['wind_gust', 'gust']) and not c.endswith('_unit')]
            wd_cols = [c for c in df.columns if 'wind_direction' in c.lower() and not c.endswith('_unit')]
            rain_cols = [c for c in df.columns if any(x in c.lower() for x in ['rain', 'rainfall']) and not c.endswith('_unit')]

            temps = maybe_f_to_c(extract_numeric(df[outdoor_temp_col])) if outdoor_temp_col else [0]
            hum = extract_numeric(df[hum_cols[0]]) if hum_cols else [0]
            pres = extract_numeric(df[pres_cols[0]]) if pres_cols else [0]
            ws = extract_numeric(df[ws_cols[0]]) if ws_cols else [0]
            wg = extract_numeric(df[wg_cols[0]]) if wg_cols else [0]
            wd = extract_numeric(df[wd_cols[0]]) if wd_cols else [0]

            # Rain handling: store inches for report, show mm in widget
            daily_rain_in = 0.0
            if rain_cols:
                raw_rain = extract_numeric(df[rain_cols[0]])
                max_rain = max(raw_rain) if raw_rain else 0.0
                # If values look like mm totals (e.g., >5), convert to inches. Otherwise assume inches already.
                daily_rain_in = (max_rain / 25.4) if max_rain > 5 else max_rain

            def avg(v): return sum(v)/len(v) if v else 0.0

            return {
                'date': date.strftime('%Y-%m-%d'),
                'avg_temp': avg(temps),
                'max_temp': max(temps) if temps else 0.0,
                'min_temp': min(temps) if temps else 0.0,
                'total_rainfall': daily_rain_in,   # inches
                'avg_wind_speed': avg(ws),
                'max_wind_speed': max(wg) if wg else 0.0,
                'avg_wind_direction': avg(wd),
                'avg_humidity': avg(hum),
                'avg_pressure': avg(pres)
            }
        except Exception as e:
            self.logger.error(f"process_daily_stats error: {e}")
            return {}

    def save_to_database(self, daily_stats: dict, hourly_df: pd.DataFrame):
        """Insert/replace daily and hourly rows for the date."""
        try:
            def v(value):
                if isinstance(value, dict) and 'value' in value:
                    try:
                        return float(value['value'])
                    except Exception:
                        return 0.0
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except Exception:
                        return 0.0
                return 0.0

            cur = self.conn.cursor()

            cur.execute('''
                INSERT OR REPLACE INTO daily_data
                (date, avg_temp, max_temp, min_temp, total_rainfall,
                 avg_wind_speed, max_wind_speed, avg_wind_direction, avg_humidity, avg_pressure)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                daily_stats['date'], daily_stats['avg_temp'], daily_stats['max_temp'],
                daily_stats['min_temp'], daily_stats['total_rainfall'],
                daily_stats['avg_wind_speed'], daily_stats['max_wind_speed'],
                daily_stats['avg_wind_direction'], daily_stats['avg_humidity'],
                daily_stats['avg_pressure']
            ))

            temp_cols = [c for c in hourly_df.columns if 'temperature' in c.lower() and not c.endswith('_unit')]
            hum_cols = [c for c in hourly_df.columns if 'humidity' in c.lower() and not c.endswith('_unit')]
            pres_cols = [c for c in hourly_df.columns if 'pressure' in c.lower() and not c.endswith('_unit')]
            ws_cols = [c for c in hourly_df.columns if 'wind_speed' in c.lower() and not c.endswith('_unit')]
            wd_cols = [c for c in hourly_df.columns if 'wind_direction' in c.lower() and not c.endswith('_unit')]
            rain_cols = [c for c in hourly_df.columns if 'rain' in c.lower() and not c.endswith('_unit')]

            for _, row in hourly_df.iterrows():
                cur.execute('''
                    INSERT OR REPLACE INTO hourly_data
                    (datetime, date, temperature, humidity, pressure, wind_speed, wind_direction, rainfall)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['datetime'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['datetime']) else '',
                    daily_stats['date'],
                    v(row[temp_cols[0]] if temp_cols else 0.0),
                    v(row[hum_cols[0]] if hum_cols else 0.0),
                    v(row[pres_cols[0]] if pres_cols else 0.0),
                    v(row[ws_cols[0]] if ws_cols else 0.0),
                    v(row[wd_cols[0]] if wd_cols else 0.0),
                    v(row[rain_cols[0]] if rain_cols else 0.0),
                ))

            self.conn.commit()
            self.logger.info(f"Saved DB rows for {daily_stats['date']}")
        except Exception as e:
            self.logger.error(f"save_to_database error: {e}")

    # --------------------------- Charts ---------------------------

    def create_minimal_temperature_chart(self, target_date: datetime, days: int = 14) -> Optional[str]:
        """Create a compact color strip showing avg temp for the last `days`."""
        try:
            end_date = target_date
            start_date = end_date - timedelta(days=days - 1)
            df = pd.read_sql_query('''
                SELECT date, avg_temp FROM daily_data
                WHERE date BETWEEN ? AND ? ORDER BY date ASC
            ''', self.conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])

            if df.empty:
                self.logger.warning("No temp data for minimal chart")
                return None

            temps = df['avg_temp'].tolist()
            colors = [self.get_temperature_color(t) for t in temps]

            fig, ax = plt.subplots(figsize=(max(3, len(temps) * 0.3), 4))
            ax.bar(range(len(temps)), [1.0] * len(temps), color=colors, edgecolor='none', width=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlim(-0.5, len(temps) - 0.5)
            ax.set_ylim(0, 1.0)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            charts_dir = Path(self.config['data']['charts_directory']); charts_dir.mkdir(exist_ok=True)
            out = charts_dir / f'minimal_temperature_{target_date.strftime("%Y%m%d")}.png'
            fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
            plt.close(fig)
            return str(out)
        except Exception as e:
            self.logger.error(f"create_minimal_temperature_chart error: {e}")
            return None

    def create_matplotlib_windrose(self, target_date: datetime, days: int = 7,
                               width_px: int = 220, dpi: int = 150) -> Optional[str]:
    ...
    # compute a square figure size from desired pixel width
    figsize = (width_px / dpi, width_px / dpi)  # e.g., 220px @150dpi ≈ 1.47"
    # remove old hard-coded values:
    # dpi = 200
    # figsize = (5, 5)
    ...
    # when creating the figure, use the new figsize/dpi everywhere
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ...
    # (same for fallback path)
    # fig = plt.figure(figsize=figsize, dpi=dpi)

        try:
            end_date = target_date
            start_date = end_date - timedelta(days=days - 1)

            df = pd.read_sql_query('''
                SELECT wind_speed, wind_direction
                FROM hourly_data
                WHERE date BETWEEN ? AND ? AND wind_speed > 0
            ''', self.conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])

            if df.empty:
                self.logger.warning("No wind data for windrose")
                return None

            s = pd.to_numeric(df['wind_speed'], errors='coerce')
            d = pd.to_numeric(df['wind_direction'], errors='coerce')
            mask = s.notna() & d.notna() & (d >= 0) & (d <= 360) & (s > 0)
            s = s[mask].values
            d = d[mask].values
            if s.size == 0:
                self.logger.warning("Windrose: all wind samples invalid/zero")
                return None

            charts_dir = Path(self.config['data']['charts_directory']); charts_dir.mkdir(exist_ok=True)
            out_file = charts_dir / f'windrose_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.png'

            dpi = 200
            figsize = (5, 5)
            speed_bins = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30]  # mph
            cmap = cm.viridis
            norm = mcolors.BoundaryNorm(speed_bins, cmap.N, clip=True)

            used_toolkit = False
            try:
                # Preferred: matplotlib-windrose package (import name is 'windrose')
                from windrose import WindroseAxes  # type: ignore
                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = WindroseAxes.from_ax(fig=fig)
                ax.bar(d, s, normed=True, opening=0.8, edgecolor='none', bins=speed_bins, cmap=cmap)
                ax.set_legend(title="Wind speed (mph)", loc='lower center',
                              bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
                used_toolkit = True
            except Exception as e:
                # Fallback: pure Matplotlib stacked polar bars by sector/speed bin
                self.logger.info(f"Windrose toolkit unavailable ({e}); using fallback.")
                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = plt.subplot(111, projection='polar')
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)

                n_sectors = 16  # 22.5° sectors
                sector_edges_deg = np.linspace(0, 360, n_sectors + 1)
                sector_edges_rad = np.deg2rad(sector_edges_deg)
                width = np.deg2rad(360 / n_sectors)

                sector_idx = np.digitize(d % 360, sector_edges_deg, right=False) - 1
                sector_idx = np.clip(sector_idx, 0, n_sectors - 1)

                counts = np.zeros((len(speed_bins) - 1, n_sectors), dtype=float)
                for i_sec in range(n_sectors):
                    s_in = s[sector_idx == i_sec]
                    if s_in.size == 0:
                        continue
                    hist, _ = np.histogram(s_in, bins=speed_bins)
                    counts[:, i_sec] = hist

                total = counts.sum()
                if total > 0:
                    counts = counts / total * 100.0

                bottoms = np.zeros(n_sectors)
                bin_centers = [(speed_bins[i] + speed_bins[i + 1]) / 2 for i in range(len(speed_bins) - 1)]

                for i_bin in range(len(speed_bins) - 1):
                    radii = counts[i_bin, :]
                    if radii.max() == 0:
                        continue
                    color_val = cmap(norm(bin_centers[i_bin]))
                    ax.bar(sector_edges_rad[:-1] + width / 2.0, radii, width=width,
                           bottom=bottoms, color=color_val, edgecolor='none', align='center')
                    bottoms += radii

                ax.set_rlabel_position(225)
                ax.grid(True, alpha=0.3)
                from matplotlib.patches import Patch
                legend_patches = [Patch(facecolor=cmap(norm((speed_bins[i] + speed_bins[i + 1]) / 2)),
                                        label=f"{speed_bins[i]}–{speed_bins[i + 1]} mph")
                                  for i in range(len(speed_bins) - 1)]
                ax.legend(handles=legend_patches, title="Wind speed (mph)",
                          loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

            fig.suptitle(f"Windrose ({start_date.strftime('%d %b')}–{end_date.strftime('%d %b %Y')})",
                         y=0.98, fontsize=12)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            self.logger.info(f"Windrose created ({'toolkit' if used_toolkit else 'fallback'}) at {out_file}")
            return str(out_file)
        except Exception as e:
            self.logger.error(f"create_matplotlib_windrose error: {e}")
            return None

    # --------------------------- Widgets & Email ---------------------------

    def create_beaker_rainfall_widget(self, rainfall_mm: float, max_scale: float = 25.0) -> str:
        """Return HTML for a simple rainfall 'beaker' visualization."""
        try:
            rainfall_inches = rainfall_mm / 25.4
            rainfall_percentage = min(100, (rainfall_mm / max_scale) * 100)
            if rainfall_mm > max_scale:
                max_scale = rainfall_mm * 1.2
                rainfall_percentage = (rainfall_mm / max_scale) * 100

            filled_cells = 0
            if rainfall_mm > 0:
                cells_float = (rainfall_percentage / 100) * 6
                filled_cells = max(1, round(cells_float))

            beaker_html = f'''
            <div style="text-align: center; margin: 20px 0;">
                <div style="font-size: 16px; color: #333; font-weight: bold; margin-bottom: 15px;">Daily Rainfall</div>
                <table cellpadding="0" cellspacing="0" style="margin: 0 auto; border-collapse: collapse;">
                <tr>
                    <td style="vertical-align: bottom; padding-right: 8px;">
                        <table cellpadding="0" cellspacing="0" style="height: 120px; border-collapse: collapse;">
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right;">{max_scale:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right;">{max_scale*0.8:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right;">{max_scale*0.6:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right;">{max_scale*0.4:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right;">{max_scale*0.2:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right;">0</td></tr>
                        </table>
                    </td>
                    <td style="vertical-align: bottom; padding-right: 5px;">
                        <table cellpadding="0" cellspacing="0" style="height: 120px; border-collapse: collapse;">
                            {''.join(['<tr style="height: 20px;"><td style="border-bottom: 1px solid #999; width: 10px;"></td></tr>' for _ in range(6)])}
                        </table>
                    </td>
                    <td style="vertical-align: bottom;">
                        <div style="width: 120px; height: 120px; border: 3px solid #666; border-top: none; border-radius: 0 0 6px 6px; background: #f8f8f8; overflow: hidden;">
                            <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; width: 100%; height: 120px;">
                                {f'<tr style="height: {20 * (6 - filled_cells)}px;"><td style="background: transparent;"></td></tr>' if filled_cells < 6 else ''}
                                {f'<tr style="height: {20 * filled_cells}px;"><td style="background: linear-gradient(to top, #1976d2 0%, #42a5f5 50%, #64b5f6 100%);"></td></tr>' if filled_cells > 0 else ''}
                            </table>
                        </div>
                    </td>
                </tr>
                </table>
                <div style="font-size: 20px; color: #1976d2; margin: 10px 0 2px 0; font-weight: 600;">{rainfall_mm:.1f} mm</div>
                <div style="font-size: 14px; color: #666; margin-bottom: 15px;">({rainfall_inches:.2f} inches)</div>
            </div>
            '''
            return beaker_html
        except Exception as e:
            self.logger.error(f"beaker widget error: {e}")
            return "<div style='text-align: center; color: #999;'>Rainfall data unavailable</div>"

    def send_email_report(self, daily_stats: dict):
        """Compose and send the HTML email with inline windrose and attached temp strip."""
        try:
            target_date = datetime.strptime(daily_stats['date'], '%Y-%m-%d')
            temp_chart_file = self.create_minimal_temperature_chart(target_date, days=14)
            windrose_file = self.create_matplotlib_windrose(target_date, days=7, width_px=220, dpi=150)

            rainfall_mm = daily_stats['total_rainfall'] * 25.4
            beaker_html = self.create_beaker_rainfall_widget(rainfall_mm, max_scale=25.0)

            receiver_emails = self.config['email']['receiver_email']
            to_addresses = ', '.join(receiver_emails if isinstance(receiver_emails, list) else [receiver_emails])

            msg = MIMEMultipart('related')  # allows inline images
            msg['From'] = self.config['email']['sender_email']
            msg['To'] = to_addresses
            msg['Subject'] = f"Weather Report - {daily_stats['date']}"

            alt = MIMEMultipart('alternative')
            msg.attach(alt)

            text_body = f"""Daily Weather Report - {daily_stats['date']}

Current Day Statistics:
• Average Temperature: {daily_stats['avg_temp']:.1f}°C
• Maximum Temperature: {daily_stats['max_temp']:.1f}°C
• Minimum Temperature: {daily_stats['min_temp']:.1f}°C
• Total Rainfall: {daily_stats['total_rainfall']:.2f} inches ({rainfall_mm:.1f} mm)
• Average Wind Speed: {daily_stats['avg_wind_speed']:.1f} mph
• Maximum Wind Speed: {daily_stats['max_wind_speed']:.1f} mph
• Average Wind Direction: {daily_stats['avg_wind_direction']:.0f}°
• Average Humidity: {daily_stats['avg_humidity']:.1f}%
• Average Pressure: {daily_stats['avg_pressure']:.2f} inHg

Temperature chart attached.
Generated by Ecowitt Weather Processor
"""

            html_body = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Daily Weather Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 640px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }}
  .container {{ background: white; border-radius: 8px; padding: 24px; }}
  .header {{ text-align: center; border-bottom: 2px solid #e9ecef; padding-bottom: 12px; margin-bottom: 18px; }}
  .weather-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
  .weather-item {{ background: #f8f9fa; padding: 10px; border-radius: 6px; text-align: center; }}
  .weather-value {{ font-size: 18px; font-weight: bold; color: #2c3e50; }}
  .weather-label {{ font-size: 12px; color: #666; text-transform: uppercase; margin-top: 2px; }}
  .section {{ text-align: center; margin: 18px 0; padding: 14px; background: #f8f9fa; border-radius: 8px; }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1 style="margin:0;">Daily Weather Report</h1>
      <div style="color:#666;">{daily_stats['date']}</div>
    </div>

    <div class="weather-grid">
      <div class="weather-item"><div class="weather-value">{daily_stats['avg_temp']:.1f}°C</div><div class="weather-label">Average Temp</div></div>
      <div class="weather-item"><div class="weather-value">{daily_stats['max_temp']:.1f}°C</div><div class="weather-label">Max Temp</div></div>
      <div class="weather-item"><div class="weather-value">{daily_stats['min_temp']:.1f}°C</div><div class="weather-label">Min Temp</div></div>
      <div class="weather-item"><div class="weather-value">{daily_stats['avg_humidity']:.1f}%</div><div class="weather-label">Humidity</div></div>
      <div class="weather-item"><div class="weather-value">{daily_stats['avg_wind_speed']:.1f} mph</div><div class="weather-label">Avg Wind</div></div>
      <div class="weather-item"><div class="weather-value">{daily_stats['max_wind_speed']:.1f} mph</div><div class="weather-label">Max Wind</div></div>
      <div class="weather-item"><div class="weather-value">{daily_stats['avg_pressure']:.2f} inHg</div><div class="weather-label">Pressure</div></div>
      <div class="weather-item"><div class="weather-value">{daily_stats['avg_wind_direction']:.0f}°</div><div class="weather-label">Wind Dir</div></div>
    </div>

    <div class="section">
      {beaker_html}
    </div>

    <div class="section">
      <div style="font-size:16px;color:#333;font-weight:bold;margin-bottom:10px;">Weekly Windrose</div>
      {"<img src='cid:windrose' alt='Windrose' style='max-width:100%;height:auto;border:1px solid #eee;border-radius:8px;' />" if windrose_file else "<div style='color:#999;'>Wind data unavailable</div>"}
      <div style="font-size:11px;color:#999;margin-top:6px;">
        Based on 7 days ending {target_date.strftime("%Y-%m-%d")}
      </div>
    </div>

    <div style="text-align: center; margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee; color: #666; font-size: 13px;">
      Temperature chart attached • Generated by Ecowitt Weather Processor
    </div>
  </div>
</body>
</html>
"""
            alt.attach(MIMEText(text_body, 'plain'))
            alt.attach(MIMEText(html_body, 'html'))

            # Attach temperature strip as file attachment
            if temp_chart_file and os.path.exists(temp_chart_file):
                with open(temp_chart_file, 'rb') as f:
                    img_data = f.read()
                image = MIMEImage(img_data)
                image.add_header('Content-Disposition', 'attachment', filename='temperature_overview.png')
                msg.attach(image)

            # Attach windrose inline (Content-ID)
            if windrose_file and os.path.exists(windrose_file):
                with open(windrose_file, 'rb') as f:
                    img_data = f.read()
                wind_img = MIMEImage(img_data)
                wind_img.add_header('Content-ID', '<windrose>')
                wind_img.add_header('Content-Disposition', 'inline', filename='windrose.png')
                msg.attach(wind_img)

            self.logger.info("Sending email…")
            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['sender_email'], self.config['email']['sender_password'])
            server.sendmail(self.config['email']['sender_email'], receiver_emails, msg.as_string())
            server.quit()
            self.logger.info(f"SUCCESS: Email sent to {receiver_emails}")
        except Exception as e:
            self.logger.error(f"send_email_report error: {e}", exc_info=True)

    # --------------------------- Orchestration ---------------------------

    def get_last_processed_date(self) -> Optional[datetime]:
        """Return the most recent date in daily_data."""
        try:
            cur = self.conn.cursor()
            cur.execute('SELECT MAX(date) FROM daily_data')
            r = cur.fetchone()
            return datetime.strptime(r[0], '%Y-%m-%d') if r and r[0] else None
        except Exception as e:
            self.logger.error(f"get_last_processed_date error: {e}")
            return None

    def get_missing_dates(self, end_date: datetime):
        """Return list of dates missing between last processed and end_date inclusive."""
        last = self.get_last_processed_date()
        if last is None:
            return [end_date]
        dates = []
        d = last + timedelta(days=1)
        while d <= end_date:
            dates.append(d)
            d += timedelta(days=1)
        return dates

    def get_daily_data_from_database(self, target_date: datetime) -> Optional[dict]:
        """Load daily stats row for a given date if present."""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            cur = self.conn.cursor()
            cur.execute('''
                SELECT avg_temp, max_temp, min_temp, total_rainfall,
                       avg_wind_speed, max_wind_speed, avg_wind_direction,
                       avg_humidity, avg_pressure
                FROM daily_data WHERE date = ?
            ''', (date_str,))
            r = cur.fetchone()
            if not r:
                return None
            return {
                'date': date_str,
                'avg_temp': r[0], 'max_temp': r[1], 'min_temp': r[2],
                'total_rainfall': r[3], 'avg_wind_speed': r[4],
                'max_wind_speed': r[5], 'avg_wind_direction': r[6],
                'avg_humidity': r[7], 'avg_pressure': r[8]
            }
        except Exception as e:
            self.logger.error(f"get_daily_data_from_database error: {e}")
            return None

    def process_single_date(self, target_date: datetime) -> bool:
        """Fetch, compute, and store data for a single date."""
        try:
            if self.get_daily_data_from_database(target_date):
                self.logger.info(f"{target_date.strftime('%Y-%m-%d')} already processed.")
                return True
            hourly_df = self.get_ecowitt_data(target_date)
            if hourly_df is None or hourly_df.empty:
                self.logger.warning(f"No hourly data for {target_date.strftime('%Y-%m-%d')}")
                return False
            daily_stats = self.process_daily_stats(hourly_df, target_date)
            if not daily_stats:
                self.logger.error("Failed to compute daily stats")
                return False
            self.save_to_database(daily_stats, hourly_df)
            return True
        except Exception as e:
            self.logger.error(f"process_single_date error for {target_date}: {e}")
            return False

    def process_previous_day(self):
        """
        Main entry:
        - Catch up missing dates up to yesterday.
        - Also ensure we have a full 7-day window ending yesterday (for windrose).
        - Send the email for yesterday.
        """
        try:
            yesterday = datetime.now() - timedelta(days=1)
            target_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            self.logger.info(f"Processing up to {target_date.strftime('%Y-%m-%d')}")

            # Catch up any missing dates
            missing = self.get_missing_dates(target_date)
            if missing:
                self.logger.info(f"Catching up {len(missing)} day(s): {[d.strftime('%Y-%m-%d') for d in missing]}")
                for d in missing:
                    self.process_single_date(d)

            # Ensure last 7-day window exists (even if DB is not persisted)
            start_fill = target_date - timedelta(days=6)
            d = start_fill
            while d <= target_date:
                self.process_single_date(d)
                d += timedelta(days=1)

            daily_stats = self.get_daily_data_from_database(target_date)
            if not daily_stats:
                self.logger.error("No daily stats for yesterday; aborting email.")
                return

            self.send_email_report(daily_stats)

        except Exception as e:
            self.logger.error(f"process_previous_day error: {e}")
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()


# --------------------------- main ---------------------------

def main():
    processor = EcowittWeatherProcessor()
    processor.process_previous_day()


if __name__ == "__main__":
    main()
