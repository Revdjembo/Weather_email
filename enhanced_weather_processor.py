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
import base64


class EcowittWeatherProcessor:
    def __init__(self, config_file='weather_config.json'):
        """Initialize the weather processor with configuration"""
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.setup_database()

        # Initialize temperature chart functionality
        self.__init_temperature_chart__()

    def __init_temperature_chart__(self):
        """Initialize temperature chart functionality"""
        # Custom temperature-color mapping from Excel file
        self.TEMPERATURE_COLOR_MAP = [
            (-9, '#435897'),
            (-6, '#1d92c1'),
            (-3, '#60c3c1'),
            (0, '#7fcebc'),
            (3, '#91d5ba'),
            (6, '#cfebb2'),
            (9, '#e3ecab'),
            (12, '#ffe796'),
            (15, '#ffc96c'),
            (18, '#ffb34c'),
            (21, '#f67639'),
            (24, '#c30031'),
            (27, '#3a000e')
        ]

        self.setup_temperature_color_interpolation()

    def setup_temperature_color_interpolation(self):
        """Setup color interpolation for any temperature value"""
        temps = [item[0] for item in self.TEMPERATURE_COLOR_MAP]

        # Convert hex colors to RGB values
        rgb_colors = []
        for _, hex_color in self.TEMPERATURE_COLOR_MAP:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            rgb_colors.append(rgb)

        # Create interpolation functions for R, G, B channels
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
            return '#808080'  # Gray fallback

    def create_minimal_temperature_chart(self, target_date: datetime, days: int = 14) -> str:
        """Create minimal temperature chart for email reports"""
        try:
            # Get temperature data from database
            end_date = target_date
            start_date = end_date - timedelta(days=days - 1)

            query = '''
                SELECT date, avg_temp
                FROM daily_data 
                WHERE date BETWEEN ? AND ?
                ORDER BY date ASC
            '''

            df = pd.read_sql_query(query, self.conn, params=[
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ])

            if df.empty:
                self.logger.warning("No temperature data available for minimal chart")
                return None

            # Prepare data
            temps = df['avg_temp'].tolist()

            # Get colors for each temperature
            colors = [self.get_temperature_color(temp) for temp in temps]

            # All bars have the same height
            bar_height = 1.0
            bar_heights = [bar_height] * len(temps)

            # Create the minimal chart
            fig, ax = plt.subplots(figsize=(len(temps) * 0.3, 4))

            # Create bars - all the same length, colored by temperature
            bars = ax.bar(range(len(temps)), bar_heights,
                          color=colors, alpha=1.0, edgecolor='none', width=1.0)

            # Remove all axes, labels, ticks, and spines
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Remove any padding/margins
            ax.set_xlim(-0.5, len(temps) - 0.5)
            ax.set_ylim(0, bar_height)

            # Remove all whitespace around the plot
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Save the chart
            charts_dir = Path(self.config['data']['charts_directory'])
            charts_dir.mkdir(exist_ok=True)

            temp_chart_file = charts_dir / f'minimal_temperature_{target_date.strftime("%Y%m%d")}.png'

            plt.savefig(temp_chart_file, dpi=300, bbox_inches='tight', pad_inches=0,
                        facecolor='white', edgecolor='none')
            plt.close()

            # Log summary
            avg_temp = sum(temps) / len(temps)
            temp_range = f"{min(temps):.1f}°C to {max(temps):.1f}°C"

            self.logger.info(f"Minimal temperature chart created: {temp_chart_file}")
            self.logger.info(
                f"Chart covers {days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
            self.logger.info(f"Temperature range: {temp_range}, Average: {avg_temp:.1f}°C")

            return str(temp_chart_file)

        except Exception as e:
            self.logger.error(f"Error creating minimal temperature chart: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def create_beaker_rainfall_widget(self, rainfall_mm: float, max_scale: float = 50.0) -> str:
        """Create HTML rainfall beaker widget"""
        try:
            rainfall_inches = rainfall_mm / 25.4
            rainfall_percentage = min(100, (rainfall_mm / max_scale) * 100)
            
            # Auto-adjust scale if rainfall exceeds current max
            if rainfall_mm > max_scale:
                max_scale = rainfall_mm * 1.2
                rainfall_percentage = (rainfall_mm / max_scale) * 100

            beaker_html = f'''
            <div style="text-align: center; margin: 20px 0;">
                <div style="font-size: 18px; color: #333; font-weight: bold; margin-bottom: 10px;">Daily Rainfall</div>
                
                <div style="display: flex; align-items: flex-end; justify-content: center; margin-bottom: 10px;">
                    <!-- Scale labels -->
                    <div style="display: flex; flex-direction: column; justify-content: space-between; height: 120px; font-size: 10px; color: #666; margin-right: 5px; text-align: right; padding-top: 2px;">
                        <div style="display: flex; align-items: center; height: 1px;"><span>{max_scale:.0f}</span><div style="width: 6px; height: 1px; background: #ccc; margin-left: 4px;"></div></div>
                        <div style="display: flex; align-items: center; height: 1px;"><span>{max_scale*0.8:.0f}</span><div style="width: 6px; height: 1px; background: #ccc; margin-left: 4px;"></div></div>
                        <div style="display: flex; align-items: center; height: 1px;"><span>{max_scale*0.6:.0f}</span><div style="width: 6px; height: 1px; background: #ccc; margin-left: 4px;"></div></div>
                        <div style="display: flex; align-items: center; height: 1px;"><span>{max_scale*0.4:.0f}</span><div style="width: 6px; height: 1px; background: #ccc; margin-left: 4px;"></div></div>
                        <div style="display: flex; align-items: center; height: 1px;"><span>{max_scale*0.2:.0f}</span><div style="width: 6px; height: 1px; background: #ccc; margin-left: 4px;"></div></div>
                        <div style="display: flex; align-items: center; height: 1px;"><span>0</span><div style="width: 6px; height: 1px; background: #ccc; margin-left: 4px;"></div></div>
                    </div>
                    
                    <!-- Beaker -->
                    <div style="width: 80px; height: 120px; border: 2px solid #ddd; border-radius: 0 0 12px 12px; position: relative; background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%); overflow: hidden; display: inline-block;">
                        <div style="position: absolute; bottom: 0; left: 0; right: 0; background: linear-gradient(to bottom, #4fc3f7 0%, #29b6f6 50%, #0288d1 100%); border-radius: 0 0 10px 10px; height: {rainfall_percentage}%;"></div>
                    </div>
                </div>
                
                <div style="font-size: 24px; color: #333; margin: 10px 0 5px 0; font-weight: 300;">{rainfall_mm:.1f} mm</div>
                <div style="font-size: 14px; color: #666; margin-bottom: 15px;">({rainfall_inches:.2f} inches)</div>
            </div>
            '''
            
            return beaker_html

        except Exception as e:
            self.logger.error(f"Error creating beaker widget: {e}")
            return "<div>Error creating rainfall display</div>"

    def load_config(self, config_file: str = 'weather_config.json') -> Dict:
        """Load configuration from JSON file or environment variables (for cloud deployment)"""
        import os
        
        # Check if we're running in GitHub Actions or similar cloud environment
        if os.getenv('GITHUB_ACTIONS') or os.getenv('ECOWITT_API_KEY'):
            # Setup basic logging first for cloud environment
            import logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            
            logger.info("Running in cloud environment - using environment variables")
            cloud_config = {
                "ecowitt": {
                    "api_key": os.getenv('ECOWITT_API_KEY', ''),
                    "application_key": os.getenv('ECOWITT_APP_KEY', ''),
                    "mac": os.getenv('ECOWITT_MAC', 'F0:F5:BD:8A:FA:9C'),  # Default to your MAC
                    "base_url": "https://api.ecowitt.net/api/v3"
                },
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": os.getenv('GMAIL_EMAIL', 'revdjem@gmail.com'),  # Default to your email
                    "sender_password": os.getenv('GMAIL_PASSWORD', ''),
                    "receiver_email": os.getenv('RECEIVER_EMAILS', 'revdjem@gmail.com,gillythomp1@gmail.com').split(',')
                },
                "data": {
                    "database_file": "weather_data.db",
                    "charts_directory": "weather_charts"
                }
            }
            
            # Validate required environment variables
            required_vars = ['ECOWITT_API_KEY', 'ECOWITT_APP_KEY', 'GMAIL_PASSWORD']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {missing_vars}")
                raise ValueError(f"Missing required environment variables: {missing_vars}")
                
            return cloud_config
        
        # Local development - use config file
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
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file: {config_file}")
            print("Please update the configuration with your actual credentials!")
            return default_config

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('weather_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Setup SQLite database for storing weather data"""
        db_file = self.config['data']['database_file']
        self.conn = sqlite3.connect(db_file)

        # Create tables if they don't exist
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                date TEXT PRIMARY KEY,
                avg_temp REAL,
                max_temp REAL,
                min_temp REAL,
                total_rainfall REAL,
                avg_wind_speed REAL,
                max_wind_speed REAL,
                avg_wind_direction REAL,
                avg_humidity REAL,
                avg_pressure REAL
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS hourly_data (
                datetime TEXT PRIMARY KEY,
                date TEXT,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                wind_direction REAL,
                rainfall REAL,
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

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('code') == 0 and data.get('data'):
                # Flatten the nested data structure
                flattened_data = {}
                for category, values in data['data'].items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            if isinstance(value, dict) and 'value' in value:
                                # Extract value from nested structure
                                flattened_data[f"{category}_{key}"] = float(value['value'])
                            else:
                                flattened_data[f"{category}_{key}"] = value
                    else:
                        flattened_data[category] = values

                # Create DataFrame with current timestamp
                flattened_data['datetime'] = datetime.now()
                flattened_data['date'] = datetime.now().date()

                df = pd.DataFrame([flattened_data])
                self.logger.info(f"Using real-time data as fallback: {len(df)} record")
                self.logger.info(f"Real-time columns: {list(df.columns)}")
                return df

        except Exception as e:
            self.logger.error(f"Error getting real-time fallback: {e}")
            return None

    def process_daily_stats(self, df: pd.DataFrame, date: datetime) -> Dict:
        """Calculate daily statistics from weather data - FIXED TEMPERATURE CALCULATION"""
        try:
            self.logger.info(f"Processing stats for DataFrame with columns: {list(df.columns)}")

            # Print first few rows to see data structure
            if not df.empty:
                self.logger.info(f"Processing {len(df)} records")
                for col in df.columns:
                    if col not in ['datetime', 'date'] and df[col].dtype in ['float64', 'int64']:
                        sample_value = df[col].iloc[0] if len(df) > 0 else 'N/A'
                        value_type = type(df[col].iloc[0]) if len(df) > 0 else 'N/A'
                        self.logger.info(f"Column '{col}': {sample_value} (type: {value_type})")

            # Improved helper function to safely extract numeric values
            def safe_numeric_extract(series, default=0):
                """Extract numeric values from series, handling all data types"""
                if series.empty:
                    return [default]

                numeric_values = []

                for value in series:
                    extracted_value = None

                    # Handle different data types
                    if isinstance(value, dict):
                        # Try common dictionary keys
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

                    # Only add valid numeric values
                    if extracted_value is not None and not (pd.isna(extracted_value) or np.isnan(extracted_value)):
                        numeric_values.append(extracted_value)

                self.logger.info(f"Extracted {len(numeric_values)} valid values from {len(series)} records")
                return numeric_values if numeric_values else [default]

            # Enhanced temperature conversion with better detection
            def smart_temp_convert(temp_values, column_name=""):
                """Convert temperatures from Fahrenheit to Celsius if needed"""
                if not temp_values:
                    return temp_values

                avg_temp = sum(temp_values) / len(temp_values)
                min_temp = min(temp_values)
                max_temp = max(temp_values)

                # More sophisticated Fahrenheit detection
                # Fahrenheit if: average > 40, or min > 32, or max > 85
                is_fahrenheit = (avg_temp > 40) or (min_temp > 32) or (max_temp > 85)

                if is_fahrenheit:
                    converted_values = [(temp - 32) * 5 / 9 for temp in temp_values]
                    self.logger.info(f"Converting {column_name} from Fahrenheit to Celsius:")
                    self.logger.info(f"  Original range: {min_temp:.1f}°F to {max_temp:.1f}°F (avg: {avg_temp:.1f}°F)")
                    converted_avg = sum(converted_values) / len(converted_values)
                    self.logger.info(
                        f"  Converted range: {min(converted_values):.1f}°C to {max(converted_values):.1f}°C (avg: {converted_avg:.1f}°C)")
                    return converted_values
                else:
                    self.logger.info(
                        f"Temperature data appears to be in Celsius already: {min_temp:.1f}°C to {max_temp:.1f}°C")
                    return temp_values

            # Find temperature columns with better matching
            temp_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(temp_key in col_lower for temp_key in ['temperature', 'temp']) and not col.endswith('_unit'):
                    temp_columns.append(col)

            self.logger.info(f"Found temperature columns: {temp_columns}")

            # Prioritize outdoor temperature columns
            outdoor_temp_col = None
            for col in temp_columns:
                if 'outdoor' in col.lower():
                    outdoor_temp_col = col
                    break

            if not outdoor_temp_col and temp_columns:
                outdoor_temp_col = temp_columns[0]

            self.logger.info(f"Using temperature column: {outdoor_temp_col}")

            # Extract and process temperature values
            temp_values = []
            if outdoor_temp_col:
                raw_temp_values = safe_numeric_extract(df[outdoor_temp_col])
                temp_values = smart_temp_convert(raw_temp_values, outdoor_temp_col)

                if temp_values:
                    self.logger.info(
                        f"Final temperature values: min={min(temp_values):.1f}°C, max={max(temp_values):.1f}°C, avg={sum(temp_values) / len(temp_values):.1f}°C")
                else:
                    self.logger.warning("No valid temperature values extracted!")

            # Find other sensor columns with improved rain handling
            humidity_columns = [col for col in df.columns if 'humidity' in col.lower() and not col.endswith('_unit')]
            pressure_columns = [col for col in df.columns if 'pressure' in col.lower() and not col.endswith('_unit')]
            wind_speed_columns = [col for col in df.columns if
                                  'wind_speed' in col.lower() and not col.endswith('_unit')]
            wind_gust_columns = [col for col in df.columns if
                                 any(x in col.lower() for x in ['wind_gust', 'gust']) and not col.endswith('_unit')]
            wind_dir_columns = [col for col in df.columns if
                                'wind_direction' in col.lower() and not col.endswith('_unit')]
            
            # IMPROVED RAIN HANDLING: Prioritize the best rain columns
            rain_columns = [col for col in df.columns if
                            any(x in col.lower() for x in ['rain', 'rainfall']) and not col.endswith('_unit')]
            
            # Prioritize rain columns in order of preference for daily totals
            priority_rain_columns = []
            for priority in ['daily', 'event', 'hourly']:
                for col in rain_columns:
                    if priority in col.lower():
                        priority_rain_columns.append(col)
                        break
            
            # Fallback to any rain column if none of the priority ones found
            if not priority_rain_columns and rain_columns:
                priority_rain_columns = [rain_columns[0]]
            
            self.logger.info(f"Found rain columns: {rain_columns}")
            self.logger.info(f"Using prioritized rain column: {priority_rain_columns[0] if priority_rain_columns else 'None'}")

            # Extract values for all measurements
            humidity_values = safe_numeric_extract(df[humidity_columns[0]]) if humidity_columns else [0]
            pressure_values = safe_numeric_extract(df[pressure_columns[0]]) if pressure_columns else [0]
            wind_speed_values = safe_numeric_extract(df[wind_speed_columns[0]]) if wind_speed_columns else [0]
            wind_gust_values = safe_numeric_extract(df[wind_gust_columns[0]]) if wind_gust_columns else [0]
            wind_dir_values = safe_numeric_extract(df[wind_dir_columns[0]]) if wind_dir_columns else [0]
            
            # IMPROVED RAIN PROCESSING
            rain_values = []
            if priority_rain_columns:
                raw_rain_values = safe_numeric_extract(df[priority_rain_columns[0]])
                
                # Check if rain values are in mm and convert to inches
                max_rain = max(raw_rain_values) if raw_rain_values else 0
                if max_rain > 5:  # If max rain > 5, assume it's in mm (since 5+ inches would be extreme)
                    rain_values = [val / 25.4 for val in raw_rain_values]  # Convert mm to inches
                    self.logger.info(f"Converting rain from mm to inches:")
                    self.logger.info(f"  Original max: {max_rain:.1f}mm")
                    self.logger.info(f"  Converted max: {max(rain_values):.2f}inches")
                else:
                    rain_values = raw_rain_values
                    self.logger.info(f"Rain appears to be in inches already: max {max_rain:.2f}inches")
                
                # For daily totals, use the maximum value (since daily rain accumulates then resets)
                if 'daily' in priority_rain_columns[0].lower() or 'event' in priority_rain_columns[0].lower():
                    daily_rain_total = max(rain_values) if rain_values else 0
                    self.logger.info(f"Using maximum daily rain total: {daily_rain_total:.2f}inches")
                else:
                    daily_rain_total = sum(rain_values) if rain_values else 0
                    self.logger.info(f"Using summed rain total: {daily_rain_total:.2f}inches")
            else:
                rain_values = [0]
                daily_rain_total = 0
                self.logger.warning("No rain columns found!")

            # Calculate statistics with proper handling of empty lists
            def safe_avg(values):
                return sum(values) / len(values) if values and len(values) > 0 else 0

            def safe_max(values):
                return max(values) if values and len(values) > 0 else 0

            def safe_min(values):
                return min(values) if values and len(values) > 0 else 0

            def safe_sum(values):
                return sum(values) if values and len(values) > 0 else 0

            stats = {
                'date': date.strftime('%Y-%m-%d'),
                'avg_temp': safe_avg(temp_values),
                'max_temp': safe_max(temp_values),
                'min_temp': safe_min(temp_values),
                'total_rainfall': daily_rain_total,  # Use the improved rain calculation
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
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def save_to_database(self, daily_stats: Dict, hourly_df: pd.DataFrame):
        """Save data to SQLite database"""
        try:
            # Helper function to safely extract numeric values for database
            def safe_db_value(value):
                if isinstance(value, dict):
                    if 'value' in value:
                        try:
                            return float(value['value'])
                        except (ValueError, TypeError):
                            return 0
                    # Fallback to other keys
                    for key in ['val', 'data', 'current']:
                        if key in value:
                            try:
                                return float(value[key])
                            except (ValueError, TypeError):
                                continue
                    return 0
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0
                return 0

            # Save daily stats
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

            # Save hourly data
            temp_columns = [col for col in hourly_df.columns if
                            'temperature' in col.lower() and not col.endswith('_unit')]
            humidity_columns = [col for col in hourly_df.columns if
                                'humidity' in col.lower() and not col.endswith('_unit')]
            pressure_columns = [col for col in hourly_df.columns if
                                'pressure' in col.lower() and not col.endswith('_unit')]
            wind_speed_columns = [col for col in hourly_df.columns if
                                  'wind_speed' in col.lower() and not col.endswith('_unit')]
            wind_dir_columns = [col for col in hourly_df.columns if
                                'wind_direction' in col.lower() and not col.endswith('_unit')]
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
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def create_weekly_windrose(self, target_date: datetime) -> str:
        """Create weekly windrose chart"""
        try:
            # Get last 7 days of data
            end_date = target_date
            start_date = end_date - timedelta(days=6)

            query = '''
                SELECT wind_speed, wind_direction 
                FROM hourly_data 
                WHERE date BETWEEN ? AND ?
                AND wind_speed > 0
            '''

            df = pd.read_sql_query(query, self.conn, params=[
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ])

            if df.empty:
                self.logger.warning("No wind data available for windrose, creating sample chart")
                # Create a simple placeholder chart
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No Wind Data Available',
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes, fontsize=16)
                plt.title(f'Weekly Wind Rose\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
            else:
                # Create windrose
                ax = WindroseAxes.from_ax()
                ax.bar(df['wind_direction'], df['wind_speed'], normed=True, opening=0.8, edgecolor='white')
                ax.set_legend(title='Wind Speed (mph)', loc='upper left', bbox_to_anchor=(1.1, 1))
                plt.title(f'Weekly Wind Rose\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')

            # Save chart
            charts_dir = Path(self.config['data']['charts_directory'])
            charts_dir.mkdir(exist_ok=True)

            windrose_file = charts_dir / f'windrose_{target_date.strftime("%Y%m%d")}.png'
            plt.savefig(windrose_file, dpi=300, bbox_inches='tight')
            plt.close()

            return str(windrose_file)

        except Exception as e:
            self.logger.error(f"Error creating windrose: {e}")
            return None

    def send_email_report(self, daily_stats: Dict, windrose_file: str, rain_file: str, catchup_days: List[datetime] = None):
        """Send HTML email report with daily statistics, charts, and beaker widget"""
        try:
            # Create the minimal temperature chart
            target_date = datetime.strptime(daily_stats['date'], '%Y-%m-%d')
            temp_chart_file = self.create_minimal_temperature_chart(target_date, days=14)

            # Create the beaker widget HTML
            rainfall_mm = daily_stats['total_rainfall'] * 25.4  # Convert inches to mm for beaker
            beaker_html = self.create_beaker_rainfall_widget(rainfall_mm)

            # Handle both single email and list of emails
            receiver_emails = self.config['email']['receiver_email']
            if isinstance(receiver_emails, list):
                to_addresses = ', '.join(receiver_emails)
            else:
                to_addresses = receiver_emails
                receiver_emails = [receiver_emails]  # Convert to list for sendmail

            # Create email message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config['email']['sender_email']
            
            # Update subject line if there are catchup days
            if catchup_days and len(catchup_days) > 1:
                msg['To'] = to_addresses
                msg['Subject'] = f"Weather Report - {daily_stats['date']} (+ {len(catchup_days)-1} catch-up days)"
            else:
                msg['To'] = to_addresses
                msg['Subject'] = f"Weather Report - {daily_stats['date']}"

            # Create plain text version
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
"""

            # Add catchup data to text version if available
            if catchup_days and len(catchup_days) > 1:
                text_body += f"""

CATCH-UP DATA - Previously Missing Days:
"""
                # Sort catchup days and exclude the current day
                sorted_catchup = sorted([d for d in catchup_days if d.strftime('%Y-%m-%d') != daily_stats['date']])
                
                for catchup_date in sorted_catchup:
                    catchup_stats = self.get_daily_data_from_database(catchup_date)
                    if catchup_stats:
                        text_body += f"""
{catchup_date.strftime('%Y-%m-%d')}:
• Temperature: {catchup_stats['min_temp']:.1f}°C to {catchup_stats['max_temp']:.1f}°C (avg: {catchup_stats['avg_temp']:.1f}°C)
• Rainfall: {catchup_stats['total_rainfall']:.2f} inches
• Wind: avg {catchup_stats['avg_wind_speed']:.1f} mph, max {catchup_stats['max_wind_speed']:.1f} mph
• Humidity: {catchup_stats['avg_humidity']:.1f}%, Pressure: {catchup_stats['avg_pressure']:.2f} inHg
"""

            text_body += "\n\nCharts attached: Weekly wind patterns, rainfall beaker, and temperature overview."

            # Create HTML version with beaker widget
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
                        line-height: 1.6; 
                        color: #333; 
                        max-width: 600px; 
                        margin: 0 auto; 
                        padding: 20px; 
                        background-color: #f8f9fa;
                    }}
                    .container {{ 
                        background: white; 
                        border-radius: 8px; 
                        padding: 30px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .header {{ 
                        text-align: center; 
                        border-bottom: 2px solid #e9ecef; 
                        padding-bottom: 20px; 
                        margin-bottom: 30px;
                    }}
                    .weather-grid {{ 
                        display: grid; 
                        grid-template-columns: 1fr 1fr; 
                        gap: 20px; 
                        margin-bottom: 30px;
                    }}
                    .weather-item {{ 
                        background: #f8f9fa; 
                        padding: 15px; 
                        border-radius: 6px; 
                        text-align: center;
                    }}
                    .weather-value {{ 
                        font-size: 24px; 
                        font-weight: bold; 
                        color: #2c3e50;
                    }}
                    .weather-label {{ 
                        font-size: 12px; 
                        color: #666; 
                        text-transform: uppercase;
                    }}
                    .rainfall-section {{ 
                        text-align: center; 
                        margin: 30px 0; 
                        padding: 20px; 
                        background: #f8f9fa; 
                        border-radius: 8px;
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

            # Attach both text and HTML parts
            part1 = MIMEText(text_body, 'plain')
            part2 = MIMEText(html_body, 'html')
            
            msg.attach(part1)
            msg.attach(part2)

            # Attach windrose chart
            if windrose_file and os.path.exists(windrose_file):
                with open(windrose_file, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename='weekly_windrose.png')
                    msg.attach(image)

            # Attach the minimal temperature chart
            if temp_chart_file and os.path.exists(temp_chart_file):
                with open(temp_chart_file, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename='temperature_overview.png')
                    msg.attach(image)

            # Send email
            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['sender_email'], self.config['email']['sender_password'])
            text = msg.as_string()
            server.sendmail(self.config['email']['sender_email'], receiver_emails, text)
            server.quit()

            chart_count = sum(1 for f in [windrose_file, temp_chart_file] if f and os.path.exists(f))
            self.logger.info(
                f"Enhanced HTML email report sent successfully for {daily_stats['date']} with {chart_count} charts and beaker widget to {len(receiver_emails)} recipient(s)")

        except Exception as e:
            self.logger.error(f"Error sending enhanced email: {e}")

    def get_last_processed_date(self) -> Optional[datetime]:
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
                
                'api_key': self.config['ecowitt']['api_key'],
                'mac': self.config['ecowitt']['mac'],
                'call_back': 'all'
            }

            self.logger.info(f"Testing real-time API: {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            self.logger.info(
                f"Real-time response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

            if isinstance(data, dict) and data.get('code') == 0:
                self.logger.info("SUCCESS: Real-time API connection works!")
                self.logger.info(
                    f"Real-time data structure: {list(data.get('data', {}).keys()) if 'data' in data else 'No data'}")
                return True
            else:
                self.logger.error(f"Real-time API failed with code: {data.get('code')}, message: {data.get('msg')}")
                return False

        except Exception as e:
            self.logger.error(f"Error testing real-time connection: {e}")
            return False

    def get_historical_data_all_sensors(self, start_date_str: str, end_date_str: str, cycle_type: str = '30min') -> Optional[Dict]:
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

            # Historical API requires separate calls for each sensor category
            sensor_categories = ['outdoor', 'indoor', 'pressure', 'wind', 'rainfall']
            all_sensor_data = {}
            url = f"{self.config['ecowitt']['base_url']}/device/history"

            for category in sensor_categories:
                try:
                    params = base_params.copy()
                    params['call_back'] = category

                    self.logger.info(f"Fetching {category} historical data...")
                    
                    # ADD DEBUGGING: Log the actual API request
                    self.logger.info(f"DEBUGGING API REQUEST: {url}")
                    self.logger.info(f"DEBUGGING PARAMS: {params}")
                    
                    response = requests.get(url, params=params)
                    response.raise_for_status()

                    data = response.json()
                    
                    # ADD DEBUGGING: Log API response details
                    self.logger.info(f"DEBUGGING API RESPONSE for {category}:")
                    self.logger.info(f"  Response code: {data.get('code')}")
                    self.logger.info(f"  Response message: {data.get('msg')}")
                    self.logger.info(f"  Response data keys: {list(data.get('data', {}).keys()) if data.get('data') else 'None'}")

                    if data.get('code') == 0 and data.get('data'):
                        all_sensor_data[category] = data['data']
                        self.logger.info(f"Successfully fetched {category} data")

                        # Log the data structure for debugging
                        if category in data['data']:
                            category_data = data['data'][category]
                            self.logger.info(f"{category} sensors: {list(category_data.keys())}")

                            # Show sample data points
                            for sensor_name, sensor_data in category_data.items():
                                if isinstance(sensor_data, dict) and 'list' in sensor_data:
                                    data_points = len(sensor_data['list'])
                                    self.logger.info(f"  {sensor_name}: {data_points} data points")
                                    
                                    # ADD DEBUGGING: Show first few data points
                                    if data_points > 0:
                                        sample_data = list(sensor_data['list'].items())[:3]
                                        self.logger.info(f"    Sample data: {sample_data}")
                    else:
                        self.logger.warning(f"Failed to fetch {category} data: {data.get('msg')}")

                except Exception as e:
                    self.logger.error(f"Error fetching {category} data: {e}")
                    continue

            if all_sensor_data:
                self.logger.info(f"Successfully fetched data from {len(all_sensor_data)} sensor categories")
                return all_sensor_data
            else:
                self.logger.error("Failed to fetch any historical sensor data")
                return None

        except Exception as e:
            self.logger.error(f"Error in get_historical_data_all_sensors: {e}")
            return None

    def convert_historical_to_dataframe(self, historical_data: Dict) -> Optional[pd.DataFrame]:
        """Convert historical sensor data to DataFrame format - FIXED VERSION"""
        try:
            all_records = []
            
            self.logger.info("DEBUGGING: Starting DataFrame conversion...")
            
            # Process each sensor category
            for category, category_data in historical_data.items():
                self.logger.info(f"DEBUGGING: Processing category {category}")
                
                if not isinstance(category_data, dict):
                    continue
                    
                # FIX: The data structure is historical_data['outdoor']['outdoor']['temperature']
                # We need to access the inner category data
                if category in category_data:
                    sensors = category_data[category]  # Get the actual sensor data
                else:
                    sensors = category_data
                    
                self.logger.info(f"DEBUGGING: Found sensors: {list(sensors.keys()) if isinstance(sensors, dict) else 'Not a dict'}")
                    
                for sensor_name, sensor_data in sensors.items():
                    if not isinstance(sensor_data, dict) or 'list' not in sensor_data:
                        continue
                        
                    unit = sensor_data.get('unit', '')
                    data_list = sensor_data['list']
                    
                    self.logger.info(f"DEBUGGING: Processing {category}_{sensor_name} with {len(data_list)} points")
                    
                    # Convert each timestamp entry
                    for timestamp_str, value_str in data_list.items():
                        try:
                            # Convert timestamp to datetime
                            timestamp_int = int(timestamp_str)
                            dt = datetime.fromtimestamp(timestamp_int)
                            
                            # Find or create record for this timestamp
                            existing_record = None
                            for record in all_records:
                                if record['datetime'] == dt:
                                    existing_record = record
                                    break
                                    
                            if existing_record is None:
                                existing_record = {
                                    'datetime': dt,
                                    'date': dt.date()
                                }
                                all_records.append(existing_record)
                                
                            # Add sensor data
                            column_name = f"{category}_{sensor_name}"
                            existing_record[column_name] = float(value_str)
                            existing_record[f"{column_name}_unit"] = unit
                            
                        except (ValueError, TypeError) as e:
                            self.logger.error(f"DEBUGGING: Error processing timestamp {timestamp_str}: {e}")
                            continue
                            
            self.logger.info(f"DEBUGGING: Created {len(all_records)} total records")
            
            if not all_records:
                self.logger.warning("No records created from historical data")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(all_records)
            df = df.sort_values('datetime')
            
            # Convert Fahrenheit temperatures to Celsius
            temp_columns = [col for col in df.columns if 'temperature' in col and not col.endswith('_unit')]
            for temp_col in temp_columns:
                if temp_col in df.columns:
                    # Check if it's in Fahrenheit (assume if values are typically > 50)
                    if df[temp_col].mean() > 50:
                        df[temp_col] = (df[temp_col] - 32) * 5 / 9
            
            self.logger.info(f"Created DataFrame with {len(df)} records and columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting historical data to DataFrame: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_ecowitt_data(self, date: datetime) -> Optional[pd.DataFrame]:
        """Download data from Ecowitt API for a specific date - FIXED VERSION WITH DEBUGGING"""
        try:
            # ADD DEBUGGING: Log what date we're processing
            self.logger.info(f"DEBUGGING: Input date parameter: {date}")
            self.logger.info(f"DEBUGGING: Current datetime.now(): {datetime.now()}")
            
            # First test if our credentials work with real-time
            if not self.test_realtime_connection():
                self.logger.error("Real-time test failed, aborting historical data request")
                return None

            # Create date strings for API call
            start_date_str = date.strftime('%Y-%m-%d')
            end_date_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')

            # ADD DEBUGGING: Log the exact dates being requested
            self.logger.info(f"DEBUGGING: Requesting start_date={start_date_str}, end_date={end_date_str}")
            self.logger.info(f"DEBUGGING: This should get data for {date.strftime('%Y-%m-%d')} from 00:00 to 23:59")

            self.logger.info(f"Fetching historical data for {start_date_str} to {end_date_str}")

            # Try different cycle types in order of preference
            cycle_types = ['5min', '30min', '1hour']

            for cycle_type in cycle_types:
                self.logger.info(f"Attempting with cycle_type: {cycle_type}")

                # Use the corrected historical API call
                historical_data = self.get_historical_data_all_sensors(
                    start_date_str, end_date_str, cycle_type
                )

                if historical_data:
                    # Convert to DataFrame
                    df = self.convert_historical_to_dataframe(historical_data)

                    if df is not None and not df.empty:
                        self.logger.info(f"SUCCESS: Retrieved {len(df)} historical records with {cycle_type}")
                        return df
                    else:
                        self.logger.warning(f"Empty DataFrame with {cycle_type}")
                else:
                    self.logger.warning(f"No historical data returned with {cycle_type}")

            self.logger.error("All historical data attempts failed")

            # As a fallback, try to use real-time data for current processing
            self.logger.info("Attempting to use real-time data as fallback...")
            return self.get_realtime_fallback()

        except Exception as e:
            self.logger.error(f"Error downloading Ecowitt data: {e}")
            return None

    def get_realtime_fallback(self) -> Optional[pd.DataFrame]:
        """Get real-time data as fallback when historical fails"""
        try:
            url = f"{self.config['ecowitt']['base_url']}/device/real_time"
            params = {
                'application_key': self.config['ecowitt']['application_key'],
                'api
