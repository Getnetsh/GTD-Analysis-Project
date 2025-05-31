import pandas as pd
import numpy as np
import os
import re
import logging

# Script version
SCRIPT_VERSION = "2025-05-21-v6.1"

# Setup logging
log_dir = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'cleaning_log.txt'), level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logging.info(f"Starting cleaning with script version: {SCRIPT_VERSION}")
logging.info(f"Script path: {os.path.abspath(__file__)}")
print(f"Running clean_gtd_data.py version {SCRIPT_VERSION}")

# Define paths
input_path = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\raw\globalterrorismdb_0718dist.csv'
output_dir = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed'
output_path = os.path.join(output_dir, 'gtd_cleaned_enhanced_v5.csv')
dropped_dates_path = os.path.join(output_dir, 'dropped_invalid_dates.csv')

# Read dataset
try:
    df = pd.read_csv(input_path, encoding='ISO-8859-1', low_memory=False)
    logging.info(f"Loaded raw dataset: {len(df)} rows, {len(df.columns)} columns")
except FileNotFoundError:
    logging.error(f"Input file not found: {input_path}")
    raise
except Exception as e:
    logging.error(f"Failed to read input file: {str(e)}")
    raise

# Check for nwoundter
wound_cols = [col for col in df.columns if 'wound' in col.lower() and 'ter' in col.lower()]
if not any('nwoundter' in col.lower() for col in df.columns):
    logging.warning("Column 'nwoundter' not found. Possible matches: %s", wound_cols)
    df['nwoundter'] = 0

# Columns to select
columns = [
    'eventid', 'iyear', 'imonth', 'iday', 'approxdate', 'extended', 'success',
    'nkill', 'nwound', 'nkillter', 'nwoundter', 'propvalue', 'propextent_txt', 'property',
    'region_txt', 'country_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity',
    'attacktype1_txt', 'weaptype1_txt', 'weapsubtype1_txt',
    'targtype1_txt', 'targsubtype1_txt', 'natlty1_txt',
    'gname', 'gsubname', 'guncertain1',
    'nkillus', 'nwoundus', 'summary', 'motive'
]
existing_columns = [col for col in columns if col in df.columns or col == 'nwoundter']
missing_columns = [col for col in columns if col not in df.columns and col != 'nwoundter']
if missing_columns:
    logging.warning("Missing columns: %s", missing_columns)

df = df[existing_columns]

# Rename columns
rename_map = {
    'eventid': 'eventid', 'iyear': 'year', 'imonth': 'month', 'iday': 'day', 'approxdate': 'approxdate',
    'extended': 'extended', 'success': 'success',
    'nkill': 'nkill', 'nwound': 'nwound', 'nkillter': 'nkillter', 'nwoundter': 'nwoundter',
    'propvalue': 'propvalue', 'propextent_txt': 'extent', 'property': 'property',
    'region_txt': 'region', 'country_txt': 'country', 'provstate': 'provstate', 'city': 'city',
    'latitude': 'latitude', 'longitude': 'longitude', 'specificity': 'specificity', 'vicinity': 'vicinity',
    'attacktype1_txt': 'attack_type', 'weaptype1_txt': 'weapon_type', 'weapsubtype1_txt': 'weapon_subtype',
    'targtype1_txt': 'target_type', 'targsubtype1_txt': 'target_subtype', 'natlty1_txt': 'nationality',
    'gname': 'group_name', 'gsubname': 'subgroup_name', 'guncertain1': 'uncertainty',
    'nkillus': 'nkillus', 'nwoundus': 'nwoundus', 'summary': 'summary', 'motive': 'motive'
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# 1. Handle duplicates
initial_rows = len(df)
eventid_counts = df['eventid'].value_counts()
duplicate_eventids = eventid_counts[eventid_counts > 1]
if not duplicate_eventids.empty:
    logging.warning(f"Found {len(duplicate_eventids)} duplicate eventid in raw data:\n{duplicate_eventids.head().to_string()}")
df = df.sort_values(by=['eventid']).drop_duplicates(subset=['eventid'], keep='first')
logging.info("Removed %d duplicate eventid rows", initial_rows - len(df))

# Validate no duplicates
eventid_counts = df['eventid'].value_counts()
duplicate_eventids = eventid_counts[eventid_counts > 1]
if not duplicate_eventids.empty:
    logging.error(f"Duplicate eventid remain after deduplication:\n{duplicate_eventids.head().to_string()}")
    raise ValueError("Duplicate eventid remain after deduplication")
logging.info("Validated: No duplicate eventid in cleaned data")

# 2. Handle missing and invalid values
initial_rows = len(df)
# Drop rows missing eventid
df = df.dropna(subset=['eventid'])
logging.info("Dropped %d rows missing eventid", initial_rows - len(df))
initial_rows = len(df)
# Ensure date columns are integers before checking for zeros
for col in ['year', 'month', 'day']:
    if col in df.columns:
        try:
            df[col] = df[col].astype('int32')
        except Exception as e:
            logging.error(f"Failed to convert {col} to int32: {str(e)}")
            raise
# Log and drop rows where year, month, or day is 0
year_zero = (df['year'] == 0).sum()
month_zero = (df['month'] == 0).sum()
day_zero = (df['day'] == 0).sum()
year_only = ((df['year'] == 0) & (df['month'] != 0) & (df['day'] != 0)).sum()
month_only = ((df['month'] == 0) & (df['year'] != 0) & (df['day'] != 0)).sum()
day_only = ((df['day'] == 0) & (df['year'] != 0) & (df['month'] != 0)).sum()
year_month = ((df['year'] == 0) & (df['month'] == 0) & (df['day'] != 0)).sum()
year_day = ((df['year'] == 0) & (df['day'] == 0) & (df['month'] != 0)).sum()
month_day = ((df['month'] == 0) & (df['day'] == 0) & (df['year'] != 0)).sum()
all_zero = ((df['year'] == 0) & (df['month'] == 0) & (df['day'] == 0)).sum()
logging.info(f"Rows with year = 0: {year_zero}")
logging.info(f"Rows with month = 0: {month_zero}")
logging.info(f"Rows with day = 0: {day_zero}")
logging.info(f"Breakdown: year only = {year_only}, month only = {month_only}, day only = {day_only}, "
             f"year+month = {year_month}, year+day = {year_day}, month+day = {month_day}, all zero = {all_zero}")
invalid_date_rows = df[(df['year'] == 0) | (df['month'] == 0) | (df['day'] == 0)]
if not invalid_date_rows.empty:
    unique_eventids = invalid_date_rows['eventid'].nunique()
    logging.info(f"Found {len(invalid_date_rows)} rows with year, month, or day = 0 (unique eventid: {unique_eventids})")
    logging.info(f"Sample dropped rows:\n{invalid_date_rows[['eventid', 'year', 'month', 'day', 'approxdate']].to_string(index=False)}")
    invalid_date_rows[['eventid', 'year', 'month', 'day', 'approxdate']].to_csv(dropped_dates_path, index=False)
    logging.info(f"Saved dropped rows to: {dropped_dates_path}")
    print(f"📌 Found {len(invalid_date_rows)} rows with year, month, or day = 0")
    print(f"📌 Rows with year = 0: {year_zero}, month = 0: {month_zero}, day = 0: {day_zero}")
    print(f"📌 Dropped rows saved to: {dropped_dates_path}")
df = df[(df['year'] != 0) & (df['month'] != 0) & (df['day'] != 0)]
dropped_count = initial_rows - len(df)
logging.info("Dropped %d rows with year, month, or day = 0", dropped_count)
print(f"📌 Dropped {dropped_count} rows with year, month, or day = 0")
# Validate no zero dates remain
post_drop_zero = df[(df['year'] == 0) | (df['month'] == 0) | (df['day'] == 0)]
if not post_drop_zero.empty:
    logging.error(f"Found {len(post_drop_zero)} rows with year, month, or day = 0 after dropping:\n{post_drop_zero[['eventid', 'year', 'month', 'day']].to_string()}")
    raise ValueError("Zero date rows remain after dropping")
logging.info("Validated: No rows with year, month, or day = 0 remain")

# Drop rows outside valid year range
initial_rows = len(df)
df = df[(df['year'] >= 1970) & (df['year'] <= 2017)]
logging.info("Dropped %d rows with year < 1970 or > 2017", initial_rows - len(df))

# Measures
for col in ['nkill', 'nwound', 'nkillter', 'nwoundter', 'propvalue', 'success', 'property', 'nkillus', 'nwoundus']:
    if col in df.columns:
        missing = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        logging.info("Imputed %d missing %s with 0", missing, col)

# Text dimensions
for col in ['region', 'country', 'provstate', 'city', 'extent', 'attack_type', 'weapon_type',
            'weapon_subtype', 'target_type', 'target_subtype', 'nationality', 'group_name']:
    if col in df.columns:
        missing = df[col].isna().sum()
        df[col] = df[col].fillna('unknown')
        logging.info("Imputed %d missing %s with 'unknown'", missing, col)

# Numeric dimensions
for col in ['specificity', 'vicinity', 'uncertainty']:
    if col in df.columns:
        missing = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        logging.info("Imputed %d missing %s with 0", missing, col)

# Optional text
for col in ['approxdate', 'subgroup_name', 'summary', 'motive']:
    if col in df.columns:
        missing = df[col].isna().sum()
        df[col] = df[col].fillna('')
        logging.info("Imputed %d missing %s with ''", missing, col)

# 3. Ensure data types
type_map = {
    'eventid': 'int64', 'year': 'int32', 'month': 'int32', 'day': 'int32',
    'extended': 'int32', 'success': 'int32', 'nkill': 'int32', 'nwound': 'int32',
    'nkillter': 'int32', 'nwoundter': 'int32', 'propvalue': 'float32',
    'property': 'int32', 'specificity': 'int32', 'vicinity': 'int32', 'uncertainty': 'int32',
    'latitude': 'float64', 'longitude': 'float64', 'nkillus': 'int32', 'nwoundus': 'int32'
}
for col, dtype in type_map.items():
    if col in df.columns:
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            logging.error(f"Failed to convert {col} to {dtype}: {str(e)}")
            raise

# 4. Normalize text
def normalize_text(text):
    if pd.isna(text) or text == '':
        return text
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    mappings = {
        'usa': 'united states', 'uk': 'united kingdom',
        'yunnan': 'yunnan province', 'yunnan prov': 'yunnan province'
    }
    return mappings.get(text, text)

for col in ['region', 'country', 'provstate', 'city', 'extent', 'attack_type', 'weapon_type',
            'weapon_subtype', 'target_type', 'target_subtype', 'nationality', 'group_name', 'summary', 'motive']:
    if col in df.columns:
        df[col] = df[col].apply(normalize_text)
        logging.info("Normalized text in %s", col)

# 5. Validate and impute coordinates
def format_coord(x):
    if pd.isna(x) or x == 0:
        return 0.0
    return round(x, 4)

if 'latitude' in df.columns and 'longitude' in df.columns:
    df['latitude'] = df['latitude'].clip(lower=-90, upper=90).fillna(0).apply(format_coord)
    df['longitude'] = df['longitude'].clip(lower=-180, upper=180).fillna(0).apply(format_coord)
    
    country_centroids = {
        'united states': (39.8283, -98.5795),
        'united kingdom': (51.5074, -0.1278),
        'india': (20.5937, 78.9629),
        'china': (35.8617, 104.1954),
        'unknown': (0.0, 0.0)
    }
    region_centroids = {
        'east asia': (35.8617, 104.1954),
        'south asia': (20.5937, 78.9629),
        'unknown': (0.0, 0.0)
    }
    city_centroids = {
        'kunming': (25.0389, 102.7183)
    }
    
    missing_coords = df['latitude'].isna() | df['longitude'].isna()
    invalid_coords = (df['latitude'] == 0) & (df['longitude'] == 0)
    logging.info("Found %d rows with missing coordinates", missing_coords.sum())
    logging.info("Found %d rows with invalid (0,0) coordinates", invalid_coords.sum())
    
    for idx in df.index:
        if missing_coords[idx] or invalid_coords[idx]:
            city = df.loc[idx, 'city']
            country = df.loc[idx, 'country']
            region = df.loc[idx, 'region']
            if city in city_centroids:
                lat, lon = city_centroids[city]
                df.loc[idx, ['latitude', 'longitude']] = (format_coord(lat), format_coord(lon))
            elif country in country_centroids:
                lat, lon = country_centroids[country]
                df.loc[idx, ['latitude', 'longitude']] = (format_coord(lat), format_coord(lon))
            elif region in region_centroids:
                lat, lon = region_centroids[region]
                df.loc[idx, ['latitude', 'longitude']] = (format_coord(lat), format_coord(lon))
            else:
                df.loc[idx, ['latitude', 'longitude']] = (0.0, 0.0)
                logging.info("Set default (0,0) for row %d (city: %s, country: %s, region: %s)", idx, city, country, region)
    
    empty_coords = df['latitude'].isna() | df['longitude'].isna()
    logging.info("After imputation, %d rows have empty coordinates", empty_coords.sum())
    logging.info(f"Sample coordinates:\n{df[['provstate', 'city', 'latitude', 'longitude']].head().to_string()}")

# 6. Validate attack types
valid_attack_types = [
    'assassination', 'armed assault', 'bombing/explosion', 'hijacking', 
    'hostage taking (kidnapping)', 'hostage taking (barricade incident)', 
    'facility/infrastructure attack', 'unarmed assault', 'unknown'
]
if 'attack_type' in df.columns:
    invalid_attacks = df[~df['attack_type'].isin(valid_attack_types)]['attack_type'].unique()
    if invalid_attacks.size > 0:
        logging.warning(f"Found invalid attack_type values: {invalid_attacks}")
        df.loc[~df['attack_type'].isin(valid_attack_types), 'attack_type'] = 'unknown'
    logging.info("Validated attack_type values")

# 7. Handle outliers (IQR)
def cap_outliers(series, col_name):
    if series.dtype not in ['int32', 'float32', 'float64']:
        return series
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((series < lower) | (series > upper)).sum()
    capped = series.clip(lower=lower, upper=upper)
    logging.info("Capped %d outliers in %s (lower: %.2f, upper: %.2f)", outliers, col_name, lower, upper)
    return capped

for col in ['nkill', 'nwound', 'propvalue', 'nkillus', 'nwoundus']:
    if col in df.columns:
        df[col] = cap_outliers(df[col], col)

# 8. Logical consistency and Unknown flags
if 'success' in df.columns and 'nkill' in df.columns:
    inconsistent = ((df['success'] == 0) & (df['nkill'] > 100)).sum()
    logging.info("Found %d rows with success=0 but nkill>100", inconsistent)
    df['consistency_flag'] = 0
    df.loc[(df['success'] == 0) & (df['nkill'] > 100), 'consistency_flag'] = 1

df['unknown_location_flag'] = 0
df.loc[(df['city'] == 'unknown') | ((df['latitude'] == 0) & (df['longitude'] == 0)), 'unknown_location_flag'] = 1
logging.info("Flagged %d rows with unknown_location_flag (city=unknown or lat/lon=0)", (df['unknown_location_flag'] == 1).sum())

df['unknown_group_flag'] = 0
df.loc[df['group_name'] == 'unknown', 'unknown_group_flag'] = 1
logging.info("Flagged %d rows with unknown_group_flag (group_name=unknown)", (df['unknown_group_flag'] == 1).sum())

# 9. Validate ranges
binary_clip = {
    'success': (0, 1), 'extended': (0, 1), 'property': (-1, 1),
    'specificity': (1, 5), 'vicinity': (0, 1), 'uncertainty': (0, 1)
}
for col, (low, high) in binary_clip.items():
    if col in df.columns:
        df[col] = df[col].clip(lower=low, upper=high)

# 10. Validate measures
for col in ['nkill', 'nwound', 'nkillter', 'nwoundter', 'propvalue', 'nkillus', 'nwoundus']:
    if col in df.columns:
        df[col] = df[col].clip(lower=0)

# 11. Check for location duplicates
location_cols = ['region', 'country', 'provstate', 'city', 'latitude', 'longitude']
location_duplicates = df[location_cols].duplicated().sum()
if location_duplicates > 0:
    logging.warning(f"Found {location_duplicates} duplicate location combinations:\n{df[location_cols][df[location_cols].duplicated()].head().to_string()}")

# Save cleaned dataset
df.to_csv(output_path, index=False, encoding='utf-8')
logging.info("Saved cleaned dataset: %s, %d rows, %d columns", output_path, len(df), len(df.columns))
print(f"✅ Cleaned dataset saved to: {output_path}")
print(f"📊 Rows after cleaning: {len(df)}")
print(f"🧾 Columns: {list(df.columns)}")
print(f"📜 Check {os.path.join(log_dir, 'cleaning_log.txt')} for details")
print(f"📌 Dropped invalid date rows saved to: {dropped_dates_path}")