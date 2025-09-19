#!/usr/bin/env python3
"""
Automated Ecowitt Weather Station Data Processor - Matplotlib Windrose Edition
- Processes daily weather data, stores in SQLite
- Creates a minimal temperature strip chart
- Creates a Matplotlib windrose (toolkit if available; fallback otherwise)
- Sends a styled HTML email with:
    - Metrics grid
    - Rainfall "beaker" widget (HTML/CSS)
    - Inline windrose PNG (cid:windrose)
    - Temperature strip attached (PNG)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sqlite3
import requests
import pandas as pd
import numpy as np

import matplotlib
# Use non-interactive backend in headless
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


class EcowittWeatherProcessor:
    def __init__(self, config_file='weather_config.json'):
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.setup_database()
        self.__init_temperature_chart__()

    # ---------- Setup & Configuration ----------

    def load_config(self, config_file='weather_config.json'):
    """Load configuration from JSON file or environment variables"""
    if os.getenv('GITHUB_ACTIONS') or os.getenv('ECOWITT_API_KEY'):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info("Running in cloud environment - using environment variables")

        # accept either name for the Gmail secret
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
            'ECOWITT_API_KEY': os.getenv('ECOWITT_API_KEY'),
            'ECOWITT_APP_KEY': os.getenv('ECOWITT_APP_KEY'),
            'GMAIL_EMAIL': os.getenv('GMAIL_EMAIL'),
            'GMAIL_PASSWORD/GMAIL_APP_PASSWORD': gmail_password
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            raise ValueError(f"Missing required environment variables: {missing}")

        return cloud_config

    # Local mode: use JSON file if present, otherwise defaults
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('weather_processor.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        db_file = self.config['data']['database_file']
        self.conn = sqlite3.connect(db_file)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                date TEXT PRIMARY KEY, avg_temp REAL, max_temp REAL, min_temp REAL,
                total_rainfall REAL, avg_wind_speed REAL, max_wind_speed REAL,
                avg_wind_direction REAL, avg_humidity REAL, avg_pressure REAL
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS hourly_data (
                datetime TEXT PRIMARY KEY, date TEXT, temperature REAL, humidity REAL,
                pressure REAL, wind_speed REAL, wind_direction REAL, rainfall REAL,
                FOREIGN KEY (date) REFERENCES daily_data (date)
            )
        ''')
        self.conn.commit()

    # ---------- Temperature color map for strip ----------

    def __init_temperature_chart__(self):
        self.TEMPERATURE_COLOR_MAP = [
            (-9, '#435897'), (-6, '#1d92c1'), (-3, '#60c3c1'), (0, '#7fcebc'),
            (3, '#91d5ba'), (6, '#cfebb2'), (9, '#e3ecab'), (12, '#ffe796'),
            (15, '#ffc96c'), (18, '#ffb34c'), (21, '#f67639'), (24, '#c30031'), (27, '#3a000e')
        ]
        self.setup_temperature_color_interpolation()

    def setup_temperature_color_interpolation(self):
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

    def get_temperature_color(self, temp):
        try:
            r = max(0, min(255, int(self.temp_r_interp(temp))))
            g = max(0, min(255, int(self.temp_g_interp(temp))))
            b = max(0, min(255, int(self.temp_b_interp(temp))))
            return f'#{r:02x}{g:02x}{b:02x}'
        except Exception as e:
            self.logger.warning(f"Error mapping temperature color for {temp}: {e}")
            return '#808080'

    # ---------- API & Data Processing ----------

    def test_realtime_connection(self):
        try:
            url = f"{self.config['ecowitt']['base_url']}/device/real_time"
            params = {
                'application_key': self.config['ecowitt']['application_key'],
                'api_key': self.config['ecowitt']['api_key'],
                'mac': self.config['ecowitt']['mac'],
                'call_back': 'all'
            }
            self.logger.info(f"Testing real-time API: {url}")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            ok = isinstance(data, dict) and data.get('code') == 0
            self.logger.info("SUCCESS: Real-time API connection works!" if ok else "Real-time API test failed.")
            return ok
        except Exception as e:
            self.logger.error(f"Error testing real-time connection: {e}")
            return False

    def get_historical_data_all_sensors(self, start_date_str, end_date_str, cycle_type='30min'):
        try:
            base_params = {
                'application_key': self.config['ecowitt']['application_key'],
                'api_key': self.config['ecowitt']['api_key'],
                'mac': self.config['ecowitt']['mac'],
                'start_date': start_date_str,
                'end_date': end_date_str,
                'cycle_type': cycle_type
            }
            sensor_categories = ['outdoor', 'indoor', 'pressure', 'wind', 'rainfall']
            all_sensor_data = {}
            url = f"{self.config['ecowitt']['base_url']}/device/history"

            for category in sensor_categories:
                try:
                    params = {**base_params, 'call_back': category}
                    self.logger.info(f"Fetching {category} historical data...")
                    response = requests.get(url, params=params, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    if data.get('code') == 0 and data.get('data'):
                        all_sensor_data[category] = data['data']
                        self.logger.info(f"Fetched {category} data")
                    else:
                        self.logger.warning(f"{category} data unavailable or error: {data.get('msg')}")
                except Exception as e:
                    self.logger.error(f"Error fetching {category}: {e}")
            return all_sensor_data if all_sensor_data else None
        except Exception as e:
            self.logger.error(f"Error in get_historical_data_all_sensors: {e}")
            return None

    def convert_historical_to_dataframe(self, historical_data):
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
                    data_list = sensor_data['list']
                    for timestamp_str, value_str in data_list.items():
                        try:
                            dt = datetime.fromtimestamp(int(timestamp_str))
                        except Exception:
                            continue
                        existing = next((r for r in all_records if r['datetime'] == dt), None)
                        if existing is None:
                            existing = {'datetime': dt, 'date': dt.date()}
                            all_records.append(existing)
                        col = f"{category}_{sensor_name}"
                        try:
                            existing[col] = float(value_str)
                        except Exception:
                            continue
                        existing[f"{col}_unit"] = unit

            if not all_records:
                self.logger.warning("No records from historical data")
                return None
            df = pd.DataFrame(all_records).sort_values('datetime')

            # Fahrenheit to Celsius if looks like F
            temp_columns = [c for c in df.columns if 'temperature' in c and not c.endswith('_unit')]
            for col in temp_columns:
                if col in df and df[col].mean() > 50:
                    df[col] = (df[col] - 32) * 5 / 9
            return df
        except Exception as e:
            self.logger.error(f"Error converting historical data: {e}")
            return None

    def get_ecowitt_data(self, date):
        try:
            if not self.test_realtime_connection():
                return None
            start_date_str = date.strftime('%Y-%m-%d')
            end_date_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            self.logger.info(f"Fetching historical data for {start_date_str} -> {end_date_str}")
            for cycle in ['5min', '30min', '1hour']:
                hist = self.get_historical_data_all_sensors(start_date_str, end_date_str, cycle)
                if hist:
                    df = self.convert_historical_to_dataframe(hist)
                    if df is not None and not df.empty:
                        self.logger.info(f"Retrieved {len(df)} records ({cycle})")
                        return df
            self.logger.error("Historical data attempts failed")
            return None
        except Exception as e:
            self.logger.error(f"Error downloading Ecowitt data: {e}")
            return None

    def process_daily_stats(self, df, date):
        try:
            def safe_numeric_extract(series, default=0):
                if series.empty:
                    return [default]
                out = []
                for v in series:
                    val = None
                    if isinstance(v, dict):
                        for k in ['value', 'val', 'data', 'current']:
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

            def smart_temp_convert(values):
                if not values:
                    return values
                avg = sum(values) / len(values)
                mn = min(values); mx = max(values)
                if (avg > 40) or (mn > 32) or (mx > 85):
                    return [(t - 32) * 5 / 9 for t in values]
                return values

            # Column discovery
            temp_cols = [c for c in df.columns if any(t in c.lower() for t in ['temperature', 'temp']) and not c.endswith('_unit')]
            outdoor_temp_col = next((c for c in temp_cols if 'outdoor' in c.lower()), temp_cols[0] if temp_cols else None)
            hum_cols = [c for c in df.columns if 'humidity' in c.lower() and not c.endswith('_unit')]
            pres_cols = [c for c in df.columns if 'pressure' in c.lower() and not c.endswith('_unit')]
            ws_cols = [c for c in df.columns if 'wind_speed' in c.lower() and not c.endswith('_unit')]
            wg_cols = [c for c in df.columns if any(x in c.lower() for x in ['wind_gust','gust']) and not c.endswith('_unit')]
            wd_cols = [c for c in df.columns if 'wind_direction' in c.lower() and not c.endswith('_unit')]
            rain_cols = [c for c in df.columns if any(x in c.lower() for x in ['rain','rainfall']) and not c.endswith('_unit')]

            temps = smart_temp_convert(safe_numeric_extract(df[outdoor_temp_col])) if outdoor_temp_col else [0]
            hum = safe_numeric_extract(df[hum_cols[0]]) if hum_cols else [0]
            pres = safe_numeric_extract(df[pres_cols[0]]) if pres_cols else [0]
            ws = safe_numeric_extract(df[ws_cols[0]]) if ws_cols else [0]
            wg = safe_numeric_extract(df[wg_cols[0]]) if wg_cols else [0]
            wd = safe_numeric_extract(df[wd_cols[0]]) if wd_cols else [0]

            # Rain: detect units
            daily_rain_total = 0
            if rain_cols:
                raw_rain = safe_numeric_extract(df[rain_cols[0]])
                max_rain = max(raw_rain) if raw_rain else 0
                # Heuristic: if values look like mm totals, keep; if inches, convert to inches? Your report shows inches & mm
                # We'll store "total_rainfall" in inches (as your email shows inches), convert from mm if necessary
                # If max > 5, assume mm and convert to inches
                rain_in = (max_rain / 25.4) if max_rain > 5 else max_rain
                daily_rain_total = rain_in

            def avg(v): return sum(v)/len(v) if v else 0
            stats = {
                'date': date.strftime('%Y-%m-%d'),
                'avg_temp': avg(temps),
                'max_temp': max(temps) if temps else 0,
                'min_temp': min(temps) if temps else 0,
                'total_rainfall': daily_rain_total,  # inches
                'avg_wind_speed': avg(ws),
                'max_wind_speed': max(wg) if wg else 0,
                'avg_wind_direction': avg(wd),
                'avg_humidity': avg(hum),
                'avg_pressure': avg(pres)
            }
            self.logger.info(f"Daily stats: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Error processing daily stats: {e}")
            return {}

    def save_to_database(self, daily_stats, hourly_df):
        try:
            def safe_db_value(value):
                if isinstance(value, dict) and 'value' in value:
                    try: return float(value['value'])
                    except Exception: return 0
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try: return float(value)
                    except Exception: return 0
                return 0

            cur = self.conn.cursor()
            cur.execute('''
                INSERT OR REPLACE INTO daily_data 
                (date, avg_temp, max_temp, min_temp, total_rainfall, 
                 avg_wind_speed, max_wind_speed, avg_wind_direction, avg_humidity, avg_pressure)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                daily_stats['date'], daily_stats['avg_temp'], daily_stats['max_temp'],
                daily_stats['min_temp'], daily_stats['total_rainfall'], daily_stats['avg_wind_speed'],
                daily_stats['max_wind_speed'], daily_stats['avg_wind_direction'],
                daily_stats['avg_humidity'], daily_stats['avg_pressure']
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
                    safe_db_value(row[temp_cols[0]] if temp_cols else 0),
                    safe_db_value(row[hum_cols[0]] if hum_cols else 0),
                    safe_db_value(row[pres_cols[0]] if pres_cols else 0),
                    safe_db_value(row[ws_cols[0]] if ws_cols else 0),
                    safe_db_value(row[wd_cols[0]] if wd_cols else 0),
                    safe_db_value(row[rain_cols[0]] if rain_cols else 0)
                ))
            self.conn.commit()
            self.logger.info(f"Saved data for {daily_stats['date']}")
        except Exception as e:
            self.logger.error(f"Error saving to DB: {e}")

    # ---------- Charts ----------

    def create_minimal_temperature_chart(self, target_date, days=14):
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
            ax.bar(range(len(temps)), [1.0]*len(temps), color=colors, edgecolor='none', width=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_xlim(-0.5, len(temps)-0.5); ax.set_ylim(0, 1.0)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            charts_dir = Path(self.config['data']['charts_directory']); charts_dir.mkdir(exist_ok=True)
            out = charts_dir / f'minimal_temperature_{target_date.strftime("%Y%m%d")}.png'
            fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
            plt.close(fig)
            self.logger.info(f"Minimal temp chart: {out}")
            return str(out)
        except Exception as e:
            self.logger.error(f"Error creating temp chart: {e}")
            return None

    def create_matplotlib_windrose(self, target_date, days=7):
        """
        Create a Matplotlib windrose PNG for the previous `days` ending at target_date.
        Tries the `matplotlib-windrose` toolkit; falls back to a pure-Matplotlib polar plot.
        Returns the file path or None if no data.
        """
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
                self.logger.warning("Windrose: all wind data invalid/zero")
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
                from windrose import WindroseAxes  # requires: pip install matplotlib-windrose
                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = WindroseAxes.from_ax(fig=fig)
                ax.bar(d, s, normed=True, opening=0.8, edgecolor='none', bins=speed_bins, cmap=cmap)
                ax.set_legend(title="Wind speed (mph)", loc='lower center',
                              bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
                used_toolkit = True
            except Exception as e:
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
                bin_centers = [(speed_bins[i] + speed_bins[i+1]) / 2 for i in range(len(speed_bins)-1)]

                for i_bin in range(len(speed_bins)-1):
                    radii = counts[i_bin, :]
                    if radii.max() == 0:
                        continue
                    color_val = cmap(norm(bin_centers[i_bin]))
                    ax.bar(sector_edges_rad[:-1] + width/2.0, radii, width=width,
                           bottom=bottoms, color=color_val, edgecolor='none', align='center')
                    bottoms += radii

                ax.set_rlabel_position(225)
                ax.grid(True, alpha=0.3)
                from matplotlib.patches import Patch
                legend_patches = [Patch(facecolor=cmap(norm((speed_bins[i]+speed_bins[i+1])/2)),
                                        label=f"{speed_bins[i]}–{speed_bins[i+1]} mph")
                                  for i in range(len(speed_bins)-1)]
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
            self.logger.error(f"Error creating windrose: {e}")
            return None

    # ---------- Widgets & Email ----------

    def create_beaker_rainfall_widget(self, rainfall_mm: float, max_scale: float = 25.0) -> str:
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
            </div>'''
            return beaker_html
        except Exception as e:
            self.logger.error(f"Error creating beaker widget: {e}")
            return "<div style='text-align: center; color: #999;'>Rainfall data unavailable</div>"

    def send_email_report(self, daily_stats, catchup_days=None):
        try:
            target_date = datetime.strptime(daily_stats['date'], '%Y-%m-%d')
            temp_chart_file = self.create_minimal_temperature_chart(target_date, days=14)

            # Beaker widget uses mm for display
            rainfall_mm = daily_stats['total_rainfall'] * 25.4
            beaker_html = self.create_beaker_rainfall_widget(rainfall_mm, max_scale=25.0)

            # Create Matplotlib windrose and attach inline
            windrose_file = self.create_matplotlib_windrose(target_date, days=7)

            receiver_emails = self.config['email']['receiver_email']
            to_addresses = ', '.join(receiver_emails if isinstance(receiver_emails, list) else [receiver_emails])

            msg = MIMEMultipart('related')  # allow inline images
            msg['From'] = self.config['email']['sender_email']
            msg['To'] = to_addresses
            msg['Subject'] = f"Weather Report - {daily_stats['date']}"

            # Create the alternative part (plain and html)
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

            # Attach temperature chart as attachment
            if temp_chart_file and os.path.exists(temp_chart_file):
                with open(temp_chart_file, 'rb') as f:
                    img_data = f.read()
                image = MIMEImage(img_data)
                image.add_header('Content-Disposition', 'attachment', filename='temperature_overview.png')
                msg.attach(image)

            # Attach windrose inline (CID)
            if windrose_file and os.path.exists(windrose_file):
                with open(windrose_file, 'rb') as f:
                    img_data = f.read()
                wind_img = MIMEImage(img_data)
                wind_img.add_header('Content-ID', '<windrose>')
                wind_img.add_header('Content-Disposition', 'inline', filename='windrose.png')
                msg.attach(wind_img)

            self.logger.info("Sending email...")
            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['sender_email'], self.config['email']['sender_password'])
            server.sendmail(self.config['email']['sender_email'], receiver_emails, msg.as_string())
            server.quit()
            self.logger.info(f"SUCCESS: Email sent to {receiver_emails}")
        except Exception as e:
            self.logger.error(f"Error in send_email_report: {e}", exc_info=True)

    # ---------- Orchestration ----------

    def get_last_processed_date(self):
        try:
            cur = self.conn.cursor()
            cur.execute('SELECT MAX(date) FROM daily_data')
            r = cur.fetchone()
            if r and r[0]:
                return datetime.strptime(r[0], '%Y-%m-%d')
            return None
        except Exception as e:
            self.logger.error(f"Error getting last date: {e}")
            return None

    def get_missing_dates(self, end_date):
        last = self.get_last_processed_date()
        if last is None:
            return [end_date]
        dates = []
        d = last + timedelta(days=1)
        while d <= end_date:
            dates.append(d); d += timedelta(days=1)
        return dates

    def get_daily_data_from_database(self, target_date):
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            cur = self.conn.cursor()
            cur.execute('''
                SELECT avg_temp, max_temp, min_temp, total_rainfall, 
                       avg_wind_speed, max_wind_speed, avg_wind_direction, 
                       avg_humidity, avg_pressure
                FROM daily_data WHERE date = ?
            ''', (date_str,))
            res = cur.fetchone()
            if res:
                return {
                    'date': date_str,
                    'avg_temp': res[0], 'max_temp': res[1], 'min_temp': res[2],
                    'total_rainfall': res[3], 'avg_wind_speed': res[4],
                    'max_wind_speed': res[5], 'avg_wind_direction': res[6],
                    'avg_humidity': res[7], 'avg_pressure': res[8]
                }
            return None
        except Exception as e:
            self.logger.error(f"Error reading daily data: {e}")
            return None

    def process_single_date(self, target_date):
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
            self.logger.error(f"Error processing date {target_date}: {e}")
            return False

    def process_previous_day(self):
        try:
            yesterday = datetime.now() - timedelta(days=1)
            target_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            self.logger.info(f"Processing up to {target_date.strftime('%Y-%m-%d')}")

            missing = self.get_missing_dates(target_date)
            if missing:
                self.logger.info(f"Catching up {len(missing)} day(s): {[d.strftime('%Y-%m-%d') for d in missing]}")
                for d in missing:
                    self.process_single_date(d)
            else:
                self.logger.info("No missing days.")

            # Ensure yesterday exists
            daily_stats = self.get_daily_data_from_database(target_date)
            if not daily_stats:
                if not self.process_single_date(target_date):
                    self.logger.error("Failed to process yesterday; aborting email.")
                    return
                daily_stats = self.get_daily_data_from_database(target_date)

            if daily_stats:
                self.send_email_report(daily_stats, missing if missing else None)
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()


def main():
    processor = EcowittWeatherProcessor()
    processor.process_previous_day()


if __name__ == "__main__":
    main()
