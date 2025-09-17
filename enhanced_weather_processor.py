#!/usr/bin/env python3
"""
Automated Ecowitt Weather Station Data Processor with Rainfall Beaker
Processes daily weather data, creates visualizations, and sends HTML email reports
Enhanced with rainfall beaker widget and HTML email template
"""

import requests
import pandas as pd
import numpy as np
import matplotlib
import os
# Set non-interactive backend for cloud environments
if os.getenv('GITHUB_ACTIONS'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import json
from pathlib import Path
import logging
from windrose import WindroseAxes
import sqlite3
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d


class EcowittWeatherProcessor:
    def __init__(self, config_file='weather_config.json'):
        """Initialize the weather processor with configuration"""
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.setup_database()
        self.__init_temperature_chart__()

    def __init_temperature_chart__(self):
        """Initialize temperature chart functionality"""
        self.TEMPERATURE_COLOR_MAP = [
            (-9, '#435897'), (-6, '#1d92c1'), (-3, '#60c3c1'), (0, '#7fcebc'),
            (3, '#91d5ba'), (6, '#cfebb2'), (9, '#e3ecab'), (12, '#ffe796'),
            (15, '#ffc96c'), (18, '#ffb34c'), (21, '#f67639'), (24, '#c30031'), (27, '#3a000e')
        ]
        self.setup_temperature_color_interpolation()

    def setup_temperature_color_interpolation(self):
        """Setup color interpolation for any temperature value"""
        temps = [item[0] for item in self.TEMPERATURE_COLOR_MAP]
        rgb_colors = []
        for _, hex_color in self.TEMPERATURE_COLOR_MAP:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            rgb_colors.append(rgb)

        r_values = [rgb[0] for rgb in rgb_colors]
        g_values = [rgb[1] for rgb in rgb_colors]
        b_values = [rgb[2] for rgb in rgb_colors]

        self.temp_r_interp = interp1d(temps, r_values, kind='linear', bounds_error=False, fill_value='extrapolate')
        self.temp_g_interp = interp1d(temps, g_values, kind='linear', bounds_error=False, fill_value='extrapolate')
        self.temp_b_interp = interp1d(temps, b_values, kind='linear', bounds_error=False, fill_value='extrapolate')

    def get_temperature_color(self, temp):
        """Get the appropriate color for a given temperature"""
        try:
            r = max(0, min(255, int(self.temp_r_interp(temp))))
            g = max(0, min(255, int(self.temp_g_interp(temp))))
            b = max(0, min(255, int(self.temp_b_interp(temp))))
            return f'#{r:02x}{g:02x}{b:02x}'
        except Exception as e:
            self.logger.warning(f"Error getting color for temperature {temp}: {e}")
            return '#808080'

    def create_beaker_rainfall_widget(self, rainfall_mm: float, max_scale: float = 50.0) -> str:
        """Create HTML rainfall beaker widget"""
        try:
            rainfall_inches = rainfall_mm / 25.4
            rainfall_percentage = min(100, (rainfall_mm / max_scale) * 100)
            
            if rainfall_mm > max_scale:
                max_scale = rainfall_mm * 1.2
                rainfall_percentage = (rainfall_mm / max_scale) * 100

            # Calculate how many cells should be filled (out of 6 total cells)
            filled_cells = int((rainfall_percentage / 100) * 6)
            
            beaker_html = f'''
            <div style="text-align: center; margin: 20px 0;">
                <div style="font-size: 16px; color: #333; font-weight: bold; margin-bottom: 15px;">Daily Rainfall</div>
                
                <!-- Use table for reliable email client compatibility -->
                <table cellpadding="0" cellspacing="0" style="margin: 0 auto; border-collapse: collapse;">
                <tr>
                    <!-- Scale column -->
                    <td style="vertical-align: bottom; padding-right: 8px;">
                        <table cellpadding="0" cellspacing="0" style="height: 120px; border-collapse: collapse;">
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right; vertical-align: middle;">{max_scale:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right; vertical-align: middle;">{max_scale*0.8:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right; vertical-align: middle;">{max_scale*0.6:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right; vertical-align: middle;">{max_scale*0.4:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right; vertical-align: middle;">{max_scale*0.2:.0f}</td></tr>
                            <tr style="height: 20px;"><td style="font-size: 11px; color: #666; text-align: right; vertical-align: middle;">0</td></tr>
                        </table>
                    </td>
                    
                    <!-- Tick marks column -->
                    <td style="vertical-align: bottom; padding-right: 5px;">
                        <table cellpadding="0" cellspacing="0" style="height: 120px; border-collapse: collapse;">
                            <tr style="height: 20px;"><td style="border-bottom: 1px solid #999; width: 10px;"></td></tr>
                            <tr style="height: 20px;"><td style="border-bottom: 1px solid #999; width: 10px;"></td></tr>
                            <tr style="height: 20px;"><td style="border-bottom: 1px solid #999; width: 10px;"></td></tr>
                            <tr style="height: 20px;"><td style="border-bottom: 1px solid #999; width: 10px;"></td></tr>
                            <tr style="height: 20px;"><td style="border-bottom: 1px solid #999; width: 10px;"></td></tr>
                            <tr style="height: 20px;"><td style="border-bottom: 1px solid #999; width: 10px;"></td></tr>
                        </table>
                    </td>
                    
                    <!-- Beaker column -->
                    <td style="vertical-align: bottom;">
                        <div style="width: 120px; height: 120px; border: 3px solid #666; border-top: none; border-radius: 0 0 6px 6px; background: #f8f8f8; overflow: hidden;">
                            <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; width: 100%; height: 120px;">
                                <!-- Empty cells at top -->
                                {'' if filled_cells >= 6 else f'<tr style="height: {20 * (6 - filled_cells)}px;"><td style="background: transparent;"></td></tr>'}
                                
                                <!-- Filled cells at bottom -->
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
            self.logger.error(f"Error creating beaker widget: {e}")
            return "<div>Error creating rainfall display</div>"

    def load_config(self, config_file='weather_config.json'):
        """Load configuration from JSON file or environment variables"""
        if os.getenv('GITHUB_ACTIONS') or os.getenv('ECOWITT_API_KEY'):
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger = logging.getLogger(__name__)
            logger.info("Running in cloud environment - using environment variables")
            
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
                    "sender_password": os.getenv('GMAIL_PASSWORD', ''),
                    "receiver_email": os.getenv('RECEIVER_EMAILS', 'revdjem@gmail.com,gillythomp1@gmail.com').split(',')
                },
                "data": {
                    "database_file": "weather_data.db",
                    "charts_directory": "weather_charts"
                }
            }
            
            required_vars = ['ECOWITT_API_KEY', 'ECOWITT_APP_KEY', 'GMAIL_PASSWORD']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {missing_vars}")
                raise ValueError(f"Missing required environment variables: {missing_vars}")
                
            return cloud_config
        
        default_config = {
            "ecowitt": {
                "api_key": "YOUR_ECOWITT_API_KEY",
                "application_key": "YOUR_ECOWITT_APPLICATION_KEY", 
                "mac": "YOUR_STATION_MAC_ADDRESS",
                "base_url": "https://api.ecowitt.net/api/v3"
            },
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "your_email@gmail.com",
                "sender_password": "your_app_password",
                "receiver_email": "recipient@gmail.com"
            },
            "data": {
                "database_file": "weather_data.db",
                "charts_directory": "weather_charts"
            }
        }

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file: {config_file}")
            return default_config

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('weather_processor.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Setup SQLite database for storing weather data"""
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

    def test_realtime_connection(self):
        """Test connection with real-time endpoint first"""
        try:
            url = f"{self.config['ecowitt']['base_url']}/device/real_time"
            params = {
                'application_key': self.config['ecowitt']['application_key'],
                'api_key': self.config['ecowitt']['api_key'],
                'mac': self.config['ecowitt']['mac'],
                'call_back': 'all'
            }

            self.logger.info(f"Testing real-time API: {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and data.get('code') == 0:
                self.logger.info("SUCCESS: Real-time API connection works!")
                return True
            else:
                self.logger.error(f"Real-time API failed with code: {data.get('code')}")
                return False
        except Exception as e:
            self.logger.error(f"Error testing real-time connection: {e}")
            return False

    def get_historical_data_all_sensors(self, start_date_str, end_date_str, cycle_type='30min'):
        """Get historical data from all sensor categories"""
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
                    params = base_params.copy()
                    params['call_back'] = category
                    
                    self.logger.info(f"Fetching {category} historical data...")
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get('code') == 0 and data.get('data'):
                        all_sensor_data[category] = data['data']
                        self.logger.info(f"Successfully fetched {category} data")
                    else:
                        self.logger.warning(f"Failed to fetch {category} data: {data.get('msg')}")
                except Exception as e:
                    self.logger.error(f"Error fetching {category} data: {e}")
                    continue

            return all_sensor_data if all_sensor_data else None
        except Exception as e:
            self.logger.error(f"Error in get_historical_data_all_sensors: {e}")
            return None

    def convert_historical_to_dataframe(self, historical_data):
        """Convert historical sensor data to DataFrame format"""
        try:
            all_records = []
            
            for category, category_data in historical_data.items():
                if not isinstance(category_data, dict):
                    continue
                    
                if category in category_data:
                    sensors = category_data[category]
                else:
                    sensors = category_data
                    
                for sensor_name, sensor_data in sensors.items():
                    if not isinstance(sensor_data, dict) or 'list' not in sensor_data:
                        continue
                        
                    unit = sensor_data.get('unit', '')
                    data_list = sensor_data['list']
                    
                    for timestamp_str, value_str in data_list.items():
                        try:
                            timestamp_int = int(timestamp_str)
                            dt = datetime.fromtimestamp(timestamp_int)
                            
                            existing_record = None
                            for record in all_records:
                                if record['datetime'] == dt:
                                    existing_record = record
                                    break
                                    
                            if existing_record is None:
                                existing_record = {'datetime': dt, 'date': dt.date()}
                                all_records.append(existing_record)
                                
                            column_name = f"{category}_{sensor_name}"
                            existing_record[column_name] = float(value_str)
                            existing_record[f"{column_name}_unit"] = unit
                            
                        except (ValueError, TypeError) as e:
                            continue
                            
            if not all_records:
                self.logger.warning("No records created from historical data")
                return None
                
            df = pd.DataFrame(all_records)
            df = df.sort_values('datetime')
            
            # Convert Fahrenheit temperatures to Celsius
            temp_columns = [col for col in df.columns if 'temperature' in col and not col.endswith('_unit')]
            for temp_col in temp_columns:
                if temp_col in df.columns and df[temp_col].mean() > 50:
                    df[temp_col] = (df[temp_col] - 32) * 5 / 9
            
            self.logger.info(f"Created DataFrame with {len(df)} records and columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting historical data to DataFrame: {e}")
            return None

    def get_ecowitt_data(self, date):
        """Download data from Ecowitt API for a specific date"""
        try:
            if not self.test_realtime_connection():
                self.logger.error("Real-time test failed, aborting historical data request")
                return None

            start_date_str = date.strftime('%Y-%m-%d')
            end_date_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            self.logger.info(f"Fetching historical data for {start_date_str} to {end_date_str}")
            
            cycle_types = ['5min', '30min', '1hour']
            for cycle_type in cycle_types:
                self.logger.info(f"Attempting with cycle_type: {cycle_type}")
                historical_data = self.get_historical_data_all_sensors(start_date_str, end_date_str, cycle_type)
                
                if historical_data:
                    df = self.convert_historical_to_dataframe(historical_data)
                    if df is not None and not df.empty:
                        self.logger.info(f"SUCCESS: Retrieved {len(df)} historical records with {cycle_type}")
                        return df
                        
            self.logger.error("All historical data attempts failed")
            return self.get_realtime_fallback()
            
        except Exception as e:
            self.logger.error(f"Error downloading Ecowitt data: {e}")
            return None

    def get_realtime_fallback(self):
        """Get real-time data as fallback when historical fails"""
        try:
            url = f"{self.config['ecowitt']['base_url']}/device/real_time"
            params = {
                'application_key': self.config['ecowitt']['application_key'],
                'api_key': self.config['ecowitt']['api_key'],
                'mac': self.config['ecowitt']['mac'],
                'call_back': 'all'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('code') == 0 and data.get('data'):
                flattened_data = {}
                for category, values in data['data'].items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            if isinstance(value, dict) and 'value' in value:
                                flattened_data[f"{category}_{key}"] = float(value['value'])
                            else:
                                flattened_data[f"{category}_{key}"] = value
                    else:
                        flattened_data[category] = values

                flattened_data['datetime'] = datetime.now()
                flattened_data['date'] = datetime.now().date()
                df = pd.DataFrame([flattened_data])
                self.logger.info(f"Using real-time data as fallback: {len(df)} record")
                return df
        except Exception as e:
            self.logger.error(f"Error getting real-time fallback: {e}")
            return None

    def process_daily_stats(self, df, date):
        """Calculate daily statistics from weather data"""
        try:
            def safe_numeric_extract(series, default=0):
                if series.empty:
                    return [default]
                numeric_values = []
                for value in series:
                    extracted_value = None
                    if isinstance(value, dict):
                        for key in ['value', 'val', 'data', 'current']:
                            if key in value:
                                try:
                                    extracted_value = float(value[key])
                                    break
                                except (ValueError, TypeError):
                                    continue
                    elif isinstance(value, (int, float)) and not (pd.isna(value) or np.isnan(value)):
                        extracted_value = float(value)
                    elif isinstance(value, str):
                        try:
                            extracted_value = float(value)
                        except (ValueError, TypeError):
                            pass
                    if extracted_value is not None and not (pd.isna(extracted_value) or np.isnan(extracted_value)):
                        numeric_values.append(extracted_value)
                return numeric_values if numeric_values else [default]

            def smart_temp_convert(temp_values, column_name=""):
                if not temp_values:
                    return temp_values
                avg_temp = sum(temp_values) / len(temp_values)
                min_temp = min(temp_values)
                max_temp = max(temp_values)
                is_fahrenheit = (avg_temp > 40) or (min_temp > 32) or (max_temp > 85)
                if is_fahrenheit:
                    converted_values = [(temp - 32) * 5 / 9 for temp in temp_values]
                    self.logger.info(f"Converting {column_name} from Fahrenheit to Celsius")
                    return converted_values
                return temp_values

            # Find temperature columns
            temp_columns = [col for col in df.columns if any(temp_key in col.lower() for temp_key in ['temperature', 'temp']) and not col.endswith('_unit')]
            outdoor_temp_col = None
            for col in temp_columns:
                if 'outdoor' in col.lower():
                    outdoor_temp_col = col
                    break
            if not outdoor_temp_col and temp_columns:
                outdoor_temp_col = temp_columns[0]

            # Extract temperature values
            temp_values = []
            if outdoor_temp_col:
                raw_temp_values = safe_numeric_extract(df[outdoor_temp_col])
                temp_values = smart_temp_convert(raw_temp_values, outdoor_temp_col)

            # Find other sensor columns
            humidity_columns = [col for col in df.columns if 'humidity' in col.lower() and not col.endswith('_unit')]
            pressure_columns = [col for col in df.columns if 'pressure' in col.lower() and not col.endswith('_unit')]
            wind_speed_columns = [col for col in df.columns if 'wind_speed' in col.lower() and not col.endswith('_unit')]
            wind_gust_columns = [col for col in df.columns if any(x in col.lower() for x in ['wind_gust', 'gust']) and not col.endswith('_unit')]
            wind_dir_columns = [col for col in df.columns if 'wind_direction' in col.lower() and not col.endswith('_unit')]
            rain_columns = [col for col in df.columns if any(x in col.lower() for x in ['rain', 'rainfall']) and not col.endswith('_unit')]

            # Extract values
            humidity_values = safe_numeric_extract(df[humidity_columns[0]]) if humidity_columns else [0]
            pressure_values = safe_numeric_extract(df[pressure_columns[0]]) if pressure_columns else [0]
            wind_speed_values = safe_numeric_extract(df[wind_speed_columns[0]]) if wind_speed_columns else [0]
            wind_gust_values = safe_numeric_extract(df[wind_gust_columns[0]]) if wind_gust_columns else [0]
            wind_dir_values = safe_numeric_extract(df[wind_dir_columns[0]]) if wind_dir_columns else [0]
            
            # Process rain data
            daily_rain_total = 0
            if rain_columns:
                raw_rain_values = safe_numeric_extract(df[rain_columns[0]])
                max_rain = max(raw_rain_values) if raw_rain_values else 0
                if max_rain > 5:
                    rain_values = [val / 25.4 for val in raw_rain_values]
                    self.logger.info(f"Converting rain from mm to inches")
                else:
                    rain_values = raw_rain_values
                daily_rain_total = max(rain_values) if rain_values else 0

            def safe_avg(values):
                return sum(values) / len(values) if values and len(values) > 0 else 0
            def safe_max(values):
                return max(values) if values and len(values) > 0 else 0
            def safe_min(values):
                return min(values) if values and len(values) > 0 else 0

            stats = {
                'date': date.strftime('%Y-%m-%d'),
                'avg_temp': safe_avg(temp_values),
                'max_temp': safe_max(temp_values),
                'min_temp': safe_min(temp_values),
                'total_rainfall': daily_rain_total,
                'avg_wind_speed': safe_avg(wind_speed_values),
                'max_wind_speed': safe_max(wind_gust_values),
                'avg_wind_direction': safe_avg(wind_dir_values),
                'avg_humidity': safe_avg(humidity_values),
                'avg_pressure': safe_avg(pressure_values)
            }

            self.logger.info(f"Calculated daily stats: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"Error processing daily stats: {e}")
            return {}

    def save_to_database(self, daily_stats, hourly_df):
        """Save data to SQLite database"""
        try:
            def safe_db_value(value):
                if isinstance(value, dict) and 'value' in value:
                    try:
                        return float(value['value'])
                    except (ValueError, TypeError):
                        return 0
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0
                return 0

            cursor = self.conn.cursor()
            cursor.execute('''
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

            # Find columns for hourly data
            temp_columns = [col for col in hourly_df.columns if 'temperature' in col.lower() and not col.endswith('_unit')]
            humidity_columns = [col for col in hourly_df.columns if 'humidity' in col.lower() and not col.endswith('_unit')]
            pressure_columns = [col for col in hourly_df.columns if 'pressure' in col.lower() and not col.endswith('_unit')]
            wind_speed_columns = [col for col in hourly_df.columns if 'wind_speed' in col.lower() and not col.endswith('_unit')]
            wind_dir_columns = [col for col in hourly_df.columns if 'wind_direction' in col.lower() and not col.endswith('_unit')]
            rain_columns = [col for col in hourly_df.columns if 'rain' in col.lower() and not col.endswith('_unit')]

            for _, row in hourly_df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO hourly_data
                    (datetime, date, temperature, humidity, pressure, wind_speed, wind_direction, rainfall)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['datetime'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['datetime']) else '',
                    daily_stats['date'],
                    safe_db_value(row[temp_columns[0]] if temp_columns else 0),
                    safe_db_value(row[humidity_columns[0]] if humidity_columns else 0),
                    safe_db_value(row[pressure_columns[0]] if pressure_columns else 0),
                    safe_db_value(row[wind_speed_columns[0]] if wind_speed_columns else 0),
                    safe_db_value(row[wind_dir_columns[0]] if wind_dir_columns else 0),
                    safe_db_value(row[rain_columns[0]] if rain_columns else 0)
                ))

            self.conn.commit()
            self.logger.info(f"Data saved to database for {daily_stats['date']}")

        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")

    def create_weekly_windrose(self, target_date):
        """Create weekly windrose chart"""
        try:
            end_date = target_date
            start_date = end_date - timedelta(days=6)

            query = '''
                SELECT wind_speed, wind_direction 
                FROM hourly_data 
                WHERE date BETWEEN ? AND ? AND wind_speed > 0
            '''

            df = pd.read_sql_query(query, self.conn, params=[
                start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            ])

            if df.empty:
                self.logger.warning("No wind data available for windrose, creating placeholder chart")
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No Wind Data Available',
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes, fontsize=16)
                plt.title(f'Weekly Wind Rose\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
            else:
                ax = WindroseAxes.from_ax()
                ax.bar(df['wind_direction'], df['wind_speed'], normed=True, opening=0.8, edgecolor='white')
                ax.set_legend(title='Wind Speed (mph)', loc='upper left', bbox_to_anchor=(1.1, 1))
                plt.title(f'Weekly Wind Rose\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')

            charts_dir = Path(self.config['data']['charts_directory'])
            charts_dir.mkdir(exist_ok=True)

            windrose_file = charts_dir / f'windrose_{target_date.strftime("%Y%m%d")}.png'
            plt.savefig(windrose_file, dpi=300, bbox_inches='tight')
            plt.close()

            return str(windrose_file)

        except Exception as e:
            self.logger.error(f"Error creating windrose: {e}")
            return None

    def create_minimal_temperature_chart(self, target_date, days=14):
        """Create minimal temperature chart for email reports"""
        try:
            end_date = target_date
            start_date = end_date - timedelta(days=days - 1)

            query = '''
                SELECT date, avg_temp
                FROM daily_data 
                WHERE date BETWEEN ? AND ?
                ORDER BY date ASC
            '''

            df = pd.read_sql_query(query, self.conn, params=[
                start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            ])

            if df.empty:
                self.logger.warning("No temperature data available for minimal chart")
                return None

            temps = df['avg_temp'].tolist()
            colors = [self.get_temperature_color(temp) for temp in temps]

            bar_height = 1.0
            bar_heights = [bar_height] * len(temps)

            fig, ax = plt.subplots(figsize=(len(temps) * 0.3, 4))
            bars = ax.bar(range(len(temps)), bar_heights,
                          color=colors, alpha=1.0, edgecolor='none', width=1.0)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.set_xlim(-0.5, len(temps) - 0.5)
            ax.set_ylim(0, bar_height)

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            charts_dir = Path(self.config['data']['charts_directory'])
            charts_dir.mkdir(exist_ok=True)

            temp_chart_file = charts_dir / f'minimal_temperature_{target_date.strftime("%Y%m%d")}.png'

            plt.savefig(temp_chart_file, dpi=300, bbox_inches='tight', pad_inches=0,
                        facecolor='white', edgecolor='none')
            plt.close()

            avg_temp = sum(temps) / len(temps)
            temp_range = f"{min(temps):.1f}°C to {max(temps):.1f}°C"

            self.logger.info(f"Minimal temperature chart created: {temp_chart_file}")
            self.logger.info(f"Temperature range: {temp_range}, Average: {avg_temp:.1f}°C")

            return str(temp_chart_file)

        except Exception as e:
            self.logger.error(f"Error creating minimal temperature chart: {e}")
            return None

    def send_email_report(self, daily_stats, windrose_file, rain_file, catchup_days=None):
        """Send HTML email report with daily statistics, charts, and beaker widget"""
        try:
            target_date = datetime.strptime(daily_stats['date'], '%Y-%m-%d')
            temp_chart_file = self.create_minimal_temperature_chart(target_date, days=14)

            rainfall_mm = daily_stats['total_rainfall'] * 25.4
            beaker_html = self.create_beaker_rainfall_widget(rainfall_mm, max_scale=25.0)

            receiver_emails = self.config['email']['receiver_email']
            if isinstance(receiver_emails, list):
                to_addresses = ', '.join(receiver_emails)
            else:
                to_addresses = receiver_emails
                receiver_emails = [receiver_emails]

            msg = MIMEMultipart('alternative')
            msg['From'] = self.config['email']['sender_email']
            
            if catchup_days and len(catchup_days) > 1:
                msg['To'] = to_addresses
                msg['Subject'] = f"Weather Report - {daily_stats['date']} (+ {len(catchup_days)-1} catch-up days)"
            else:
                msg['To'] = to_addresses
                msg['Subject'] = f"Weather Report - {daily_stats['date']}"

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

Charts attached: Weekly wind patterns and 14-day temperature overview.
Generated by Enhanced Ecowitt Weather Processor
"""

            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Daily Weather Report</title>
                <style>
                    body {{ 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; 
                        padding: 20px; background-color: #f8f9fa;
                    }}
                    .container {{ 
                        background: white; border-radius: 8px; padding: 30px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .header {{ 
                        text-align: center; border-bottom: 2px solid #e9ecef; 
                        padding-bottom: 20px; margin-bottom: 30px;
                    }}
                    .weather-grid {{ 
                        display: grid; grid-template-columns: 1fr 1fr; 
                        gap: 10px; margin-bottom: 20px;
                    }}
                    .weather-item {{ 
                        background: #f8f9fa; padding: 10px; border-radius: 6px; text-align: center;
                    }}
                    .weather-value {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
                    .weather-label {{ font-size: 12px; color: #666; text-transform: uppercase; margin-top: 2px; }}
                    .rainfall-section {{ 
                        text-align: center; margin: 30px 0; padding: 20px; 
                        background: #f8f9fa; border-radius: 8px;
                    }}
                    @media (max-width: 600px) {{
                        .weather-grid {{ grid-template-columns: 1fr; }}
                        body {{ padding: 10px; }}
                        .container {{ padding: 20px; }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Daily Weather Report</h1>
                        <h2>{daily_stats['date']}</h2>
                    </div>

                    <div class="weather-grid">
                        <div class="weather-item">
                            <div class="weather-value">{daily_stats['avg_temp']:.1f}°C</div>
                            <div class="weather-label">Average Temperature</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">{daily_stats['max_temp']:.1f}°C</div>
                            <div class="weather-label">Max Temperature</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">{daily_stats['min_temp']:.1f}°C</div>
                            <div class="weather-label">Min Temperature</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">{daily_stats['avg_humidity']:.1f}%</div>
                            <div class="weather-label">Humidity</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">{daily_stats['avg_wind_speed']:.1f} mph</div>
                            <div class="weather-label">Avg Wind Speed</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">{daily_stats['max_wind_speed']:.1f} mph</div>
                            <div class="weather-label">Max Wind Speed</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">{daily_stats['avg_pressure']:.2f} inHg</div>
                            <div class="weather-label">Pressure</div>
                        </div>
                        <div class="weather-item">
                            <div class="weather-value">{daily_stats['avg_wind_direction']:.0f}°</div>
                            <div class="weather-label">Wind Direction</div>
                        </div>
                    </div>

                    <div class="rainfall-section">
                        {beaker_html}
                    </div>

                    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 14px;">
                        Charts attached: Weekly wind patterns and 14-day temperature overview<br>
                        Generated by Enhanced Ecowitt Weather Processor
                    </div>
                </div>
            </body>
            </html>
            """

            part1 = MIMEText(text_body, 'plain')
            part2 = MIMEText(html_body, 'html')
            
            msg.attach(part1)
            msg.attach(part2)

            if windrose_file and os.path.exists(windrose_file):
                with open(windrose_file, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename='weekly_windrose.png')
                    msg.attach(image)

            if temp_chart_file and os.path.exists(temp_chart_file):
                with open(temp_chart_file, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename='temperature_overview.png')
                    msg.attach(image)

            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['sender_email'], self.config['email']['sender_password'])
            text = msg.as_string()
            server.sendmail(self.config['email']['sender_email'], receiver_emails, text)
            server.quit()

            chart_count = sum(1 for f in [windrose_file, temp_chart_file] if f and os.path.exists(f))
            self.logger.info(f"Enhanced HTML email report sent successfully for {daily_stats['date']} with {chart_count} charts and beaker widget to {len(receiver_emails)} recipient(s)")

        except Exception as e:
            self.logger.error(f"Error sending enhanced email: {e}")

    def get_last_processed_date(self):
        """Get the most recent date that has been processed and saved to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT MAX(date) FROM daily_data')
            result = cursor.fetchone()
            
            if result and result[0]:
                last_date = datetime.strptime(result[0], '%Y-%m-%d')
                self.logger.info(f"Last processed date in database: {last_date.strftime('%Y-%m-%d')}")
                return last_date
            else:
                self.logger.info("No previous data found in database")
                return None
        except Exception as e:
            self.logger.error(f"Error getting last processed date: {e}")
            return None

    def get_missing_dates(self, end_date):
        """Get list of dates that need to be processed between last processed date and end_date"""
        last_processed = self.get_last_processed_date()
        missing_dates = []
        
        if last_processed is None:
            missing_dates = [end_date]
        else:
            current_date = last_processed + timedelta(days=1)
            while current_date <= end_date:
                missing_dates.append(current_date)
                current_date += timedelta(days=1)
        
        if missing_dates:
            date_strings = [d.strftime('%Y-%m-%d') for d in missing_dates]
            self.logger.info(f"Found {len(missing_dates)} missing dates to process: {date_strings}")
        else:
            self.logger.info("No missing dates found - database is up to date")
            
        return missing_dates

    def process_single_date(self, target_date):
        """Process weather data for a single specific date"""
        try:
            self.logger.info(f"Processing data for {target_date.strftime('%Y-%m-%d')}")
            
            existing_daily_stats = self.get_daily_data_from_database(target_date)
            
            if existing_daily_stats:
                self.logger.info(f"Data already exists for {target_date.strftime('%Y-%m-%d')} - skipping")
                return True
            
            hourly_df = self.get_ecowitt_data(target_date)
            if hourly_df is None or hourly_df.empty:
                self.logger.warning(f"Failed to get data for {target_date.strftime('%Y-%m-%d')} - may not be available yet")
                return False
            
            daily_stats = self.process_daily_stats(hourly_df, target_date)
            if not daily_stats:
                self.logger.error(f"Failed to calculate daily statistics for {target_date.strftime('%Y-%m-%d')}")
                return False
            
            self.save_to_database(daily_stats, hourly_df)
            self.logger.info(f"Successfully processed {target_date.strftime('%Y-%m-%d')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {target_date.strftime('%Y-%m-%d')}: {e}")
            return False

    def get_daily_data_from_database(self, target_date):
        """Get daily statistics from the database if available"""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT avg_temp, max_temp, min_temp, total_rainfall, 
                       avg_wind_speed, max_wind_speed, avg_wind_direction, 
                       avg_humidity, avg_pressure
                FROM daily_data 
                WHERE date = ?
            ''', (date_str,))

            result = cursor.fetchone()

            if result:
                daily_stats = {
                    'date': date_str,
                    'avg_temp': result[0], 'max_temp': result[1], 'min_temp': result[2],
                    'total_rainfall': result[3], 'avg_wind_speed': result[4],
                    'max_wind_speed': result[5], 'avg_wind_direction': result[6],
                    'avg_humidity': result[7], 'avg_pressure': result[8]
                }
                self.logger.info(f"Found existing daily data in database for {date_str}")
                return daily_stats
            else:
                self.logger.info(f"No existing daily data found for {date_str}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting daily data from database: {e}")
            return None

    def process_previous_day(self):
        """Main function to process previous day's data with automatic gap-filling"""
        try:
            yesterday = datetime.now() - timedelta(days=1)
            target_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

            self.logger.info(f"Starting weather processing for {target_date.strftime('%Y-%m-%d')}")

            missing_dates = self.get_missing_dates(target_date)
            
            if missing_dates:
                self.logger.info(f"CATCHING UP: Processing {len(missing_dates)} missing days...")
                
                successful_dates = []
                failed_dates = []
                
                for missing_date in missing_dates:
                    if self.process_single_date(missing_date):
                        successful_dates.append(missing_date)
                    else:
                        failed_dates.append(missing_date)
                
                if successful_dates:
                    success_strings = [d.strftime('%Y-%m-%d') for d in successful_dates]
                    self.logger.info(f"SUCCESS: Successfully caught up on {len(successful_dates)} days: {success_strings}")
                
                if failed_dates:
                    failed_strings = [d.strftime('%Y-%m-%d') for d in failed_dates]
                    self.logger.warning(f"FAILED: Failed to process {len(failed_dates)} days: {failed_strings}")
                
            else:
                self.logger.info("SUCCESS: Database is up to date - no missing days to process")

            self.logger.info(f"Processing standard daily report for {target_date.strftime('%Y-%m-%d')}")
            
            daily_stats = self.get_daily_data_from_database(target_date)
            
            if not daily_stats:
                if self.process_single_date(target_date):
                    daily_stats = self.get_daily_data_from_database(target_date)
                else:
                    self.logger.error("Failed to process yesterday's data")
                    return

            if daily_stats:
                self.logger.info(f"Creating charts and sending email report for {daily_stats['date']}")
                windrose_file = self.create_weekly_windrose(target_date)
                self.send_email_report(daily_stats, windrose_file, None, missing_dates if missing_dates else None)

            self.logger.info("Enhanced daily processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error in enhanced daily processing: {e}")
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()


def main():
    """Main function to run the enhanced weather processor"""
    processor = EcowittWeatherProcessor()
    processor.process_previous_day()


if __name__ == "__main__":
    main()
