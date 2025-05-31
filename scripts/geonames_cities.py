import pandas as pd
import numpy as np
import os
import re
import logging
from scipy.stats import zscore
from geopy.distance import geodesic
from Levenshtein import distance as levenshtein_distance
import datetime

# Script version
SCRIPT_VERSION = "2025-05-26-v6.13"

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
output_path = os.path.join(output_dir, 'gtd_cleaned_enhanced_v6.13.csv')
dropped_dates_path = os.path.join(output_dir, 'dropped_invalid_dates.csv')
dropped_unknown_path = os.path.join(output_dir, 'dropped_unknown_locations_targets.csv')
dropped_propvalue_path = os.path.join(output_dir, 'dropped_invalid_propvalue.csv')
dropped_casualties_path = os.path.join(output_dir, 'dropped_inconsistent_casualties.csv')
dropped_attack_weapon_path = os.path.join(output_dir, 'dropped_inconsistent_attack_weapon.csv')
dropped_outliers_path = os.path.join(output_dir, 'dropped_outliers.csv')
dropped_geo_inconsistent_path = os.path.join(output_dir, 'dropped_geo_inconsistent.csv')
missing_coords_path = os.path.join(output_dir, 'missing_coords.csv')
unknown_groups_path = os.path.join(output_dir, 'unknown_groups.csv')
none_groups_path = os.path.join(output_dir, 'none_groups.csv')

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

# Load GeoNames data
geonames_path = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\raw\geonames_cities1.csv'
try:
    geonames_df = pd.read_csv(geonames_path, usecols=['name', 'latitude', 'longitude', 'country', 'admin1'])
    geonames_df['name'] = geonames_df['name'].str.lower().str.strip()
    logging.info(f"Loaded GeoNames data: {len(geonames_df)} cities")
except FileNotFoundError:
    logging.warning(f"GeoNames file not found: {geonames_path}. Falling back to internal centroids.")
    geonames_df = None
except Exception as e:
    logging.error(f"Failed to load GeoNames file: {str(e)}")
    geonames_df = None

# Log raw data for eventid=197001020002
if 'eventid' in df.columns:
    event_check = df[df['eventid'] == 197001020002][['eventid', 'city', 'country_txt', 'provstate', 'latitude', 'longitude']]
    if not event_check.empty:
        logging.info(f"Raw data for eventid=197001020002:\n{event_check.to_string(index=False)}")
        print(f"📌 Raw data for eventid=197001020002:\n{event_check.to_string(index=False)}")

# Check for nwoundter
wound_cols = [col for col in df.columns if 'wound' in col.lower() and 'ter' in col.lower()]
if not any('nwoundter' in col.lower() for col in df.columns):
    logging.warning(f"Column 'nwoundter' not found. Possible matches: {wound_cols}")
    df['nwoundter'] = 0

# Columns to select (excluding approxdate, motive, subgroup_name)
columns = [
    'eventid', 'iyear', 'imonth', 'iday', 'extended', 'success',
    'nkill', 'nwound', 'nkillter', 'nwoundter', 'propvalue', 'propextent_txt', 'property',
    'region_txt', 'country_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity',
    'attacktype1_txt', 'weaptype1_txt', 'weapsubtype1_txt',
    'targtype1_txt', 'targsubtype1_txt', 'natlty1_txt',
    'gname', 'guncertain1', 'nkillus', 'nwoundus', 'summary'
]
existing_columns = [col for col in columns if col in df.columns or col == 'nwoundter']
missing_columns = [col for col in columns if col not in df.columns and col != 'nwoundter']
if missing_columns:
    logging.warning(f"Missing columns: {missing_columns}")

df = df[existing_columns]

# Rename columns
rename_map = {
    'eventid': 'eventid', 'iyear': 'year', 'imonth': 'month', 'iday': 'day',
    'extended': 'extended', 'success': 'success',
    'nkill': 'nkill', 'nwound': 'nwound', 'nkillter': 'nkillter', 'nwoundter': 'nwoundter',
    'propvalue': 'propvalue', 'propextent_txt': 'extent', 'property': 'property',
    'region_txt': 'region', 'country_txt': 'country', 'provstate': 'provstate', 'city': 'city',
    'latitude': 'latitude', 'longitude': 'longitude', 'specificity': 'specificity', 'vicinity': 'vicinity',
    'attacktype1_txt': 'attack_type', 'weaptype1_txt': 'weapon_type', 'weapsubtype1_txt': 'weapon_subtype',
    'targtype1_txt': 'target_type', 'targsubtype1_txt': 'target_subtype', 'natlty1_txt': 'nationality',
    'gname': 'group_name', 'guncertain1': 'uncertainty',
    'nkillus': 'nkillus', 'nwoundus': 'nwoundus', 'summary': 'summary'
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# 1. Handle duplicates
initial_rows = len(df)
eventid_counts = df['eventid'].value_counts()
duplicate_eventids = eventid_counts[eventid_counts > 1]
if not duplicate_eventids.empty:
    logging.warning(f"Found {len(duplicate_eventids)} duplicate eventids:\n{duplicate_eventids.head().to_string()}")
df = df.sort_values(by=['eventid']).drop_duplicates(subset=['eventid'], keep='first')
logging.info(f"Removed {initial_rows - len(df)} duplicate eventid rows")

# Validate no duplicates
eventid_counts = df['eventid'].value_counts()
duplicate_eventids = eventid_counts[eventid_counts > 1]
if not duplicate_eventids.empty:
    logging.error(f"Duplicate eventids remain:\n{duplicate_eventids.head().to_string()}")
    raise ValueError("Duplicate eventids remain")
logging.info("Validated: No duplicate eventids")

# 2. Handle missing and invalid values
initial_rows = len(df)
df = df.dropna(subset=['eventid'])
logging.info(f"Dropped {initial_rows - len(df)} rows missing eventid")
initial_rows = len(df)

# Ensure date columns are integers
for col in ['year', 'month', 'day']:
    if col in df.columns:
        try:
            df[col] = df[col].astype('int32')
        except Exception as e:
            logging.error(f"Failed to convert {col} to int32: {str(e)}")
            raise

# Validate month and day ranges
def validate_date(row):
    month = row['month']
    day = row['day']
    year = row['year']
    if month < 1 or month > 12:
        return False
    max_days = {1: 31, 2: 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    return 1 <= day <= max_days.get(month, 31)

# Impute ambiguous dates
ambiguous_dates = df[(df['month'] == 0) | (df['day'] == 0)]
if not ambiguous_dates.empty:
    df.loc[df['month'] == 0, 'month'] = 1  # Default to January
    df.loc[df['day'] == 0, 'day'] = 1    # Default to 1st
    df['ambiguous_date_flag'] = 0
    df.loc[(df['month'] == 1) & (df['day'] == 1), 'ambiguous_date_flag'] = 1
    ambiguous_dates[['eventid', 'year', 'month', 'day']].to_csv(dropped_dates_path, index=False)
    logging.info(f"Imputed {len(ambiguous_dates)} ambiguous dates and flagged")
    print(f"📌 Imputed {len(ambiguous_dates)} ambiguous dates")

# Validate and drop invalid dates
invalid_dates = df[~df.apply(validate_date, axis=1)]
if not invalid_dates.empty:
    logging.info(f"Found {len(invalid_dates)} rows with invalid dates")
    invalid_dates[['eventid', 'year', 'month', 'day']].to_csv(dropped_dates_path, mode='a', index=False)
    logging.info(f"Appended invalid date rows to: {dropped_dates_path}")
    print(f"📌 Found {len(invalid_dates)} invalid dates")
df = df[df.apply(validate_date, axis=1)]
logging.info(f"Dropped {initial_rows - len(df)} rows with invalid dates")

# Drop rows with zero dates
initial_rows = len(df)
invalid_date_rows = df[(df['year'] == 0) | (df['month'] == 0) | (df['day'] == 0)]
if not invalid_date_rows.empty:
    invalid_date_rows[['eventid', 'year', 'month', 'day']].to_csv(dropped_dates_path, mode='a', index=False)
    logging.info(f"Appended {len(invalid_date_rows)} rows with zero dates to: {dropped_dates_path}")
    print(f"📌 Found {len(invalid_date_rows)} zero dates")
df = df[(df['year'] != 0) & (df['month'] != 0) & (df['day'] != 0)]
logging.info(f"Dropped {initial_rows - len(df)} rows with zero dates")

# Validate no invalid dates remain
post_drop_zero = df[(df['year'] == 0) | (df['month'] == 0) | (df['day'] == 0) | ~df.apply(validate_date, axis=1)]
if not post_drop_zero.empty:
    logging.error(f"Invalid dates remain:\n{post_drop_zero[['eventid', 'year', 'month', 'day']].to_string()}")
    raise ValueError("Invalid dates remain")
logging.info("Validated: No invalid dates")

# Drop rows outside year range
initial_rows = len(df)
df = df[(df['year'] >= 1970) & (df['year'] <= 2017)]
logging.info(f"Dropped {initial_rows - len(df)} rows with year < 1970 or > 2017")

# Add temporal features
df['quarter'] = df['month'].apply(lambda m: (m-1)//3 + 1)
try:
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df['weekday'] = df['date'].dt.day_name().fillna('unknown')
    df = df.drop(columns=['date'])
    logging.info("Added quarter and weekday columns")
except Exception as e:
    logging.warning(f"Failed to add weekday: {str(e)}")
    df['weekday'] = 'unknown'

# Validate summary dates
def check_summary_date(row):
    if pd.notna(row['summary']) and row['summary']:
        summary = row['summary'].lower()
        year_str = str(row['year'])
        if year_str not in summary and any(str(y) in summary for y in range(1970, 2018)):
            return False
    return True

date_mismatches = df[~df.apply(check_summary_date, axis=1)]
if not date_mismatches.empty:
    df['summary_date_mismatch_flag'] = 0
    df.loc[~df.apply(check_summary_date, axis=1), 'summary_date_mismatch_flag'] = 1
    date_mismatches[['eventid', 'year', 'month', 'day', 'summary']].to_csv(dropped_dates_path, mode='a', index=False)
    logging.info(f"Flagged {len(date_mismatches)} summary_date_mismatch rows")
    print(f"📌 Flagged {len(date_mismatches)} summary_date_mismatch rows")

# Impute measures
for col in ['nkill', 'nwound', 'nkillter', 'nwoundter', 'propvalue', 'success', 'property', 'nkillus', 'nwoundus']:
    if col in df.columns:
        missing = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        logging.info(f"Imputed {missing} missing {col} with 0")

# Impute text dimensions
for col in ['region', 'country', 'provstate', 'city', 'extent', 'attack_type', 'weapon_type',
            'weapon_subtype', 'target_type', 'target_subtype', 'nationality', 'group_name']:
    if col in df.columns:
        missing = df[col].isna().sum()
        df[col] = df[col].fillna('unknown')
        logging.info(f"Imputed {missing} missing {col} with 'unknown'")

# Impute numeric dimensions
for col in ['specificity', 'vicinity', 'uncertainty']:
    if col in df.columns:
        missing = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        logging.info(f"Imputed {missing} missing {col} with 0")

# Impute summary
if 'summary' in df.columns:
    missing = df['summary'].isna().sum()
    df['summary'] = df['summary'].fillna('')
    logging.info(f"Imputed {missing} missing summary with ''")

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
    if pd.isna(text) or text == '' or text is None:
        return 'unknown'
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r',.*$', '', text)  # Remove suffixes like ", CA"
    mappings = {
        'usa': 'united states', 'uk': 'united kingdom',
        'yunnan': 'yunnan province', 'yunnan prov': 'yunnan province'
    }
    return mappings.get(text, text)

for col in ['region', 'country', 'provstate', 'city', 'extent', 'attack_type', 'weapon_type',
            'weapon_subtype', 'target_type', 'target_subtype', 'nationality', 'group_name', 'summary']:
    if col in df.columns:
        df[col] = df[col].apply(normalize_text)
        logging.info(f"Normalized text in {col}")

# 5. Validate and impute coordinates
def format_coord(x):
    if pd.isna(x):
        return np.nan
    return round(x, 4)

def fuzzy_city_match(city, candidates, provstate=None, country=None, threshold=6):
    if pd.isna(city) or city == 'unknown':
        return None
    city = normalize_text(city)
    min_dist = float('inf')
    best_match = None
    for candidate in candidates:
        dist = levenshtein_distance(city, candidate)
        if dist < min_dist and dist <= threshold:
            if geonames_df is not None:
                candidate_rows = geonames_df[geonames_df['name'] == candidate]
                if provstate and country:
                    candidate_rows = candidate_rows[
                        (candidate_rows['admin1'].str.lower().str.contains(provstate, na=False)) &
                        (candidate_rows['country'].str.lower() == country)
                    ]
                elif country:
                    candidate_rows = candidate_rows[candidate_rows['country'].str.lower() == country]
                if not candidate_rows.empty:
                    min_dist = dist
                    best_match = candidate
            else:
                min_dist = dist
                best_match = candidate
    if best_match is None:
        logging.debug(f"No GeoNames match for city: {city}, country: {country}, provstate: {provstate}")
    return best_match

# Dynamic country size thresholds (approximate max distance in km)
country_sizes = {
    'united states': 4500, 'china': 5000, 'india': 3200, 'united kingdom': 1000,
    'iraq': 1400, 'afghanistan': 1300, 'pakistan': 1800, 'unknown': 1000
}

# Core centroids
country_centroids = {
    'united states': (39.8283, -98.5795),
    'united kingdom': (51.5074, -0.1278),
    'india': (20.5937, 78.9629),
    'china': (35.8617, 104.1954),
    'iraq': (33.3152, 44.3661),
    'afghanistan': (33.9391, 67.7100),
    'pakistan': (30.3753, 69.3451),
    'unknown': (np.nan, np.nan)
}
region_centroids = {
    'east asia': (35.8617, 104.1954),
    'south asia': (20.5937, 78.9629),
    'middle east & north africa': (24.7743, 46.7386),
    'north america': (39.8283, -98.5795),
    'western europe': (50.8503, 4.3517),
    'unknown': (np.nan, np.nan)
}
city_centroids = {
    'kunming': (25.0389, 102.7183),
    'baghdad': (33.3152, 44.3661),
    'kabul': (34.5553, 69.2075),
    'madison': (43.0731, -89.4012),
    'oakland': (37.791927, -122.225906)  # Precise for eventid=197001020002
}

# Dynamic centroid cache
dynamic_city_centroids = {}
if 'latitude' in df.columns and 'longitude' in df.columns:
    df['latitude'] = df['latitude'].clip(lower=-90, upper=90).apply(format_coord)
    df['longitude'] = df['longitude'].clip(lower=-180, upper=180).apply(format_coord)
    
    # Build dynamic city centroids
    valid_coord_rows = df[(~df['latitude'].isna()) & (~df['longitude'].isna()) & 
                         (df['latitude'] != 0) & (df['longitude'] != 0) & 
                         (df['city'] != 'unknown')]
    for city in valid_coord_rows['city'].unique():
        city_coords = valid_coord_rows[valid_coord_rows['city'] == city][['latitude', 'longitude']]
        if len(city_coords) > 0:
            mean_lat = city_coords['latitude'].mean()
            mean_lon = city_coords['longitude'].mean()
            dynamic_city_centroids[city] = (mean_lat, mean_lon)
            logging.info(f"Added dynamic centroid for city {city}: ({mean_lat}, {mean_lon})")
    
    # Initialize flags
    df['missing_coords_flag'] = 0
    df['low_confidence_coords_flag'] = 0
    
    missing_coords = df['latitude'].isna() | df['longitude'].isna()
    invalid_coords = (df['latitude'] == 0) & (df['longitude'] == 0)
    valid_coords = (~df['latitude'].isna()) & (~df['longitude'].isna()) & (df['latitude'] != 0) & (df['longitude'] != 0)
    logging.info(f"Missing coords: {missing_coords.sum()} rows")
    logging.info(f"Invalid (0,0) coords: {invalid_coords.sum()} rows")
    logging.info(f"Valid coords: {valid_coords.sum()} rows")
    
    for idx in df.index:
        eventid = df.loc[idx, 'eventid']
        raw_lat, raw_lon = df.loc[idx, 'latitude'], df.loc[idx, 'longitude']
        if valid_coords[idx]:
            lat, lon = raw_lat, raw_lon
            logging.info(f"Preserved coords for eventid {eventid}: ({lat}, {lon})")
            if eventid == 197001020002:
                print(f"📌 Preserved coords for eventid=197001020002: ({lat}, {lon})")
        else:
            city = df.loc[idx, 'city']
            country = df.loc[idx, 'country']
            provstate = df.loc[idx, 'provstate']
            region = df.loc[idx, 'region']
            # Try GeoNames
            matched_city = None
            if geonames_df is not None:
                candidates = geonames_df['name'].unique()
                matched_city = fuzzy_city_match(city, candidates, provstate, country)
                if matched_city:
                    city_row = geonames_df[geonames_df['name'] == matched_city].iloc[0]
                    lat, lon = city_row['latitude'], city_row['longitude']
                    df.loc[idx, ['latitude', 'longitude']] = (format_coord(lat), format_coord(lon))
                    logging.info(f"Imputed GeoNames coords for eventid {eventid} (city: {matched_city}) -> ({lat}, {lon})")
                    if eventid == 197001020002:
                        print(f"📌 Imputed GeoNames coords for eventid=197001020002: ({lat}, {lon})")
                    continue
            # Try internal centroids
            matched_city = fuzzy_city_match(city, list(city_centroids.keys()) + list(dynamic_city_centroids.keys()))
            if matched_city in city_centroids:
                lat, lon = city_centroids[matched_city]
                df.loc[idx, ['latitude', 'longitude']] = (format_coord(lat), format_coord(lon))
                logging.info(f"Imputed internal coords for eventid {eventid} (city: {matched_city}) -> ({lat}, {lon})")
                if eventid == 197001020002:
                    print(f"📌 Imputed internal coords for eventid=197001020002: ({lat}, {lon})")
            elif matched_city in dynamic_city_centroids:
                lat, lon = dynamic_city_centroids[matched_city]
                df.loc[idx, ['latitude', 'longitude']] = (format_coord(lat), format_coord(lon))
                logging.info(f"Imputed dynamic coords for eventid {eventid} (city: {matched_city}) -> ({lat}, {lon})")
                if eventid == 197001020002:
                    print(f"📌 Imputed dynamic coords for eventid=197001020002: ({lat}, {lon})")
            elif country in country_centroids and pd.notna(country_centroids[country][0]):
                lat, lon = country_centroids[country]
                df.loc[idx, ['latitude', 'longitude']] = (format_coord(lat), format_coord(lon))
                df.loc[idx, 'low_confidence_coords_flag'] = 1
                logging.info(f"Imputed country coords for eventid {eventid} (country: {country}) -> ({lat}, {lon})")
                if eventid == 197001020002:
                    print(f"📌 Imputed country coords for eventid=197001020002: ({lat}, {lon})")
            elif region in region_centroids and pd.notna(region_centroids[region][0]):
                lat, lon = region_centroids[region]
                df.loc[idx, ['latitude', 'longitude']] = (format_coord(lat), format_coord(lon))
                df.loc[idx, 'low_confidence_coords_flag'] = 1
                logging.info(f"Imputed region coords for eventid {eventid} (region: {region}) -> ({lat}, {lon})")
                if eventid == 197001020002:
                    print(f"📌 Imputed region coords for eventid=197001020002: ({lat}, {lon})")
            else:
                df.loc[idx, ['latitude', 'longitude']] = (np.nan, np.nan)
                df.loc[idx, 'missing_coords_flag'] = 1
                logging.info(f"Failed to impute coords for eventid {eventid} (city: {city}, country: {country}, region: {region})")
                if eventid == 197001020002:
                    print(f"📌 Failed to impute coords for eventid=197001020002")

# Save rows with missing coords
missing_coords_rows = df[df['missing_coords_flag'] == 1]
if not missing_coords_rows.empty:
    missing_coords_rows[['eventid', 'city', 'provstate', 'country', 'latitude', 'longitude']].to_csv(missing_coords_path, index=False)
    logging.info(f"Saved {len(missing_coords_rows)} rows with missing coords to: {missing_coords_path}")
    print(f"📌 Saved {len(missing_coords_rows)} missing coords rows to: {missing_coords_path}")

# Optional: Remove rows with NaN coordinates (uncomment to enable)
"""
initial_rows = len(df)
nan_coords = df[df['latitude'].isna() | df['longitude'].isna()]
if not nan_coords.empty:
    nan_coords[['eventid', 'city', 'provstate', 'country', 'latitude', 'longitude']].to_csv(missing_coords_path, index=False)
    logging.info(f"Saved {len(nan_coords)} NaN coordinate rows to: {missing_coords_path}")
    df = df.dropna(subset=['latitude', 'longitude'])
    logging.info(f"Dropped {initial_rows - len(df)} NaN coordinate rows")
    print(f"📌 Dropped {initial_rows - len(df)} NaN coordinate rows")
"""

# 6. Geo-spatial consistency
def check_geo_distance(row):
    if pd.notna(row['latitude']) and pd.notna(row['longitude']) and (row['latitude'] != 0 or row['longitude'] != 0):
        country = row['country']
        if country in country_centroids and pd.notna(country_centroids[country][0]):
            try:
                dist = geodesic((row['latitude'], row['longitude']), country_centroids[country]).km
                threshold = country_sizes.get(country, 1000)
                if dist > threshold:
                    logging.warning(f"Geo-inconsistent for eventid={row['eventid']}: distance {dist:.2f} km > {threshold} km")
                return dist <= threshold
            except:
                return False
    return True

geo_inconsistent = df[~df.apply(check_geo_distance, axis=1)]
if not geo_inconsistent.empty:
    geo_inconsistent[['eventid', 'country', 'latitude', 'longitude']].to_csv(dropped_geo_inconsistent_path, index=False)
    logging.info(f"Saved {len(geo_inconsistent)} geo-inconsistent rows to: {dropped_geo_inconsistent_path}")
    df.loc[~df.apply(check_geo_distance, axis=1), ['latitude', 'longitude']] = (np.nan, np.nan)
    df.loc[~df.apply(check_geo_distance, axis=1), 'missing_coords_flag'] = 1
    logging.info(f"Set {len(geo_inconsistent)} inconsistent coords to NaN")

# Save updated missing coords after geo-check
missing_coords_rows = df[df['missing_coords_flag'] == 1]
if not missing_coords_rows.empty:
    missing_coords_rows[['eventid', 'city', 'provstate', 'country', 'latitude', 'longitude']].to_csv(missing_coords_path, mode='a', index=False)
    logging.info(f"Appended {len(missing_coords_rows)} missing coords rows to: {missing_coords_path}")

# 7. Validate attack types
valid_attack_types = [
    'assassination', 'armed assault', 'bombing/explosion', 'hijacking', 
    'hostage taking (kidnapping)', 'hostage taking (barricade incident)', 
    'facility/infrastructure attack', 'unarmed assault', 'unknown'
]
if 'attack_type' in df.columns:
    invalid_attacks = df[~df['attack_type'].isin(valid_attack_types)]['attack_type'].unique()
    if invalid_attacks.size > 0:
        logging.warning(f"Invalid attack_type values: {invalid_attacks}")
        df.loc[~df['attack_type'].isin(valid_attack_types), 'attack_type'] = 'unknown'
    logging.info("Validated attack_type values")

# 8. IQR outlier detection
def detect_outliers(series, col_name, threshold=1.5):
    if series.dtype not in ['int32', 'float32', 'float64']:
        return series, pd.Series(False, index=series.index)
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))
    logging.info(f"Found {outliers.sum()} IQR outliers in {col_name}")
    return series.clip(lower=series[~outliers].min(), upper=series[~outliers].max()), outliers

for col in ['nkill', 'nwound', 'propvalue', 'nkillus', 'nwoundus']:
    if col in df.columns:
        df[col], outliers = detect_outliers(df[col], col)
        if outliers.sum() > 0:
            outlier_rows = df[outliers][['eventid', col]]
            outlier_rows.to_csv(dropped_outliers_path, mode='a', index=False)
            logging.info(f"Appended {outliers.sum()} outliers for {col} to: {dropped_outliers_path}")

# 9. Logical consistency and flags
if 'success' in df.columns and 'nkill' in df.columns:
    inconsistent = ((df['success'] == 0) & (df['nkill'] > 100)).sum()
    logging.info(f"Found {inconsistent} rows with success=0 but nkill>100")
    df['consistency_flag'] = 0
    df.loc[(df['success'] == 0) & (df['nkill'] > 100), 'consistency_flag'] = 1

df['unknown_location_flag'] = 0
df.loc[(df['city'] == 'unknown') | (df['latitude'].isna() & df['longitude'].isna()), 'unknown_location_flag'] = 1
logging.info(f"Flagged {df['unknown_location_flag'].sum()} rows with unknown_location_flag")

df['unknown_group_flag'] = 0
df.loc[df['group_name'] == 'unknown', 'unknown_group_flag'] = 1
logging.info(f"Flagged {df['unknown_group_flag'].sum()} rows with unknown_group_flag")

# 10. Drop rows with all unknown location/target attributes
initial_rows = len(df)
unknown_location_rows = df[(df['city'] == 'unknown') & (df['country'] == 'unknown') & (df['provstate'] == 'unknown')]
if not unknown_location_rows.empty:
    df = df[~((df['city'] == 'unknown') & (df['country'] == 'unknown') & (df['provstate'] == 'unknown'))]
    logging.info(f"Dropped {len(unknown_location_rows)} rows with all unknown location")
unknown_target_rows = df[(df['target_type'] == 'unknown') & (df['target_subtype'] == 'unknown') & (df['nationality'] == 'unknown')]
if not unknown_target_rows.empty:
    df = df[~((df['target_type'] == 'unknown') & (df['target_subtype'] == 'unknown') & (df['nationality'] == 'unknown'))]
    logging.info(f"Dropped {len(unknown_target_rows)} rows with all unknown target")
dropped_rows = pd.concat([unknown_location_rows, unknown_target_rows]).drop_duplicates()
if not dropped_rows.empty:
    dropped_rows[['eventid', 'city', 'country', 'provstate', 'target_type', 'target_subtype', 'nationality']].to_csv(dropped_unknown_path, index=False)
    logging.info(f"Saved dropped rows to: {dropped_unknown_path}")

# 11. Validate location hierarchy
country_to_region = {
    'united states': 'north america',
    'united kingdom': 'western europe',
    'india': 'south asia',
    'china': 'east asia',
    'iraq': 'middle east & north africa',
    'afghanistan': 'south asia',
    'pakistan': 'south asia',
    'unknown': 'unknown'
}
inconsistent_locations = df[df.apply(lambda x: x['country'] != 'unknown' and x['region'] != country_to_region.get(x['country'], x['region']), axis=1)]
if not inconsistent_locations.empty:
    for idx in inconsistent_locations.index:
        country = df.loc[idx, 'country']
        if country in country_to_region:
            df.loc[idx, 'region'] = country_to_region[country]
        logging.info(f"Corrected {len(inconsistent_locations)} inconsistent country-region mappings")

# 12. Validate and clean propvalue
def impute_extent(row):
    if row['extent'] == 'unknown' and pd.notna(row['propvalue']):
        if row['propvalue'] >= 1_000_000:
            return 'catastrophic'
        elif row['propvalue'] >= 100_000:
            return 'major'
        elif row['propvalue'] > 0:
            return 'minor'
    return row['extent']

negative_propvalue = df[df['propvalue'] < 0]
if not negative_propvalue.empty:
    df.loc[df['propvalue'] < 0, 'propvalue'] = 0
    logging.info(f"Set {len(negative_propvalue)} negative propvalue to 0")

inconsistent_propvalue = df[(df['property'] == -1) & (df['propvalue'] > 0)]
if not inconsistent_propvalue.empty:
    inconsistent_propvalue[['eventid', 'propvalue', 'property', 'extent']].to_csv(dropped_propvalue_path, index=False)
    df.loc[(df['property'] == -1) & (df['propvalue'] > 0), 'propvalue'] = 0
    logging.info(f"Set {len(inconsistent_propvalue)} inconsistent propvalue to 0 where property=-1")

# Impute extent
df['extent'] = df.apply(impute_extent, axis=1)
logging.info(f"Imputed {df['extent'].eq('unknown').sum()} remaining unknown extent")

# Flag propvalue/extent mismatches
df['propvalue_extent_mismatch_flag'] = 0
df.loc[
    ((df['extent'] == 'minor') & (df['propvalue'] >= 100_000)) |
    ((df['extent'] == 'major') & (df['propvalue'] >= 1_000_000)) |
    ((df['extent'] == 'catastrophic') & (df['propvalue'] < 1_000_000)), 'propvalue_extent_mismatch_flag'] = 1
logging.info(f"Flagged {df['propvalue_extent_mismatch_flag'].sum()} propvalue_extent_mismatch rows")
propvalue_outliers = df[df['propvalue_extent_mismatch_flag'] == 1][['eventid', 'propvalue', 'extent']]
if not propvalue_outliers.empty:
    propvalue_outliers.to_csv(dropped_propvalue_path, mode='a', index=False)
    logging.info(f"Appended {len(propvalue_outliers)} propvalue mismatches to: {dropped_propvalue_path}")

# 13. Standardize and cluster group_name
group_mappings = {
    'the taliban': 'taliban',
    'al-qaeda in iraq': 'al-qaeda',
    'al-qaida': 'al-qaeda',
    'islamic state of iraq and the levant (isil)': 'islamic state',
    'islamic state of iraq and syria (isis)': 'islamic state',
    'boko haram': 'boko haram', 'bh': 'boko haram',
    'al-shabaab': 'al-shabaab', 'al-shabab': 'al-shabaab',
    'unknown group': 'unknown', 'unidentified': 'unknown'
}
def impute_group_from_summary(row):
    if row['group_name'] == 'unknown' and pd.notna(row['summary']) and row['summary']:
        summary = row['summary'].lower()
        for group, standard in group_mappings.items():
            if group in summary:
                return standard
            if 'taliban' in summary:
                return 'taliban'
            if 'al-qaeda' in summary:
                return 'al-qaeda'
            if 'islamic state' in summary or 'isis' in summary or 'isil' in summary:
                return 'islamic state'
    return row['group_name']

def cluster_group_name(group, existing_groups, threshold=4):
    if pd.isna(group) or group is None or group == 'unknown':
        return 'unknown'
    group = normalize_text(group)
    group = group_mappings.get(group, group)
    try:
        for existing in existing_groups:
            if existing and pd.notna(existing):  # Skip None or NaN
                dist = levenshtein_distance(group, existing)
                if dist <= threshold:
                    return existing
    except Exception as e:
        logging.warning(f"Error in clustering group '{group}': {str(e)}")
        return group
    return group

if 'group_name' in df.columns:
    # Check for None values
    none_groups = df[df['group_name'].isna() | (df['group_name'] == None)]
    if not none_groups.empty:
        none_groups[['eventid', 'group_name', 'summary']].to_csv(none_groups_path, index=False)
        logging.warning(f"Saved {len(none_groups)} rows with None group_name to: {none_groups_path}")
        df.loc[df['group_name'].isna() | (df['group_name'] == None), 'group_name'] = 'unknown'
        df['none_group_flag'] = 0
        df.loc[df['group_name'] == 'unknown', 'none_group_flag'] = 1
        logging.info(f"Flagged {len(none_groups)} none_group_flag rows")
        print(f"📌 Saved {len(none_groups)} None group_name rows to: {none_groups_path}")

    # Impute from summary
    df['group_name'] = df.apply(impute_group_from_summary, axis=1)
    # Cluster groups
    initial_groups = df['group_name'].nunique()
    existing_groups = [g for g in df['group_name'].unique() if pd.notna(g) and g]
    df['group_name'] = df['group_name'].apply(lambda x: cluster_group_name(x, existing_groups))
    # Reassign low-frequency groups
    group_counts = df['group_name'].value_counts()
    low_freq_groups = group_counts[group_counts < 5].index
    df.loc[df['group_name'].isin(low_freq_groups), 'group_name'] = 'other'
    # Flag uncertain groups
    uncertain_groups = df[(df['group_name'] != 'unknown') & (df['uncertainty'] == 1)]
    if not uncertain_groups.empty:
        df['uncertain_group_flag'] = 0
        df.loc[(df['group_name'] != 'unknown') & (df['uncertainty'] == 1), 'uncertain_group_flag'] = 1
        logging.info(f"Flagged {len(uncertain_groups)} uncertain group_name")
    unknown_after = df['group_name'].eq('unknown').sum()
    logging.info(f"Reduced unknown group_name to {unknown_after} rows")
    logging.info(f"Reduced group_name from {initial_groups} to {df['group_name'].nunique()} unique values")
    print(f"📌 Reduced unknown group_name to {unknown_after} rows")
    # Save unknown groups
    unknown_groups = df[df['group_name'] == 'unknown'][['eventid', 'summary', 'country', 'group_name']]
    unknown_groups.to_csv(unknown_groups_path, index=False)
    logging.info(f"Saved {len(unknown_groups)} unknown group rows to: {unknown_groups_path}")
    print(f"📌 Saved {len(unknown_groups)} to: {unknown_groups_path}")

# 14. Validate casualty consistency
inconsistent_casualties = df[
    (df['nkillter'] > df['nkill']) |
    (df['nwoundter'] > df['nwound']) |
    (df['nkillus'] > df['nkill']) |
    (df['nwoundus'] > df['nwound']) |
    ((df['success'] == 0) & (df['nkill'] > 0)) |
    ((df['success'] == 0) & (df['nwound'] > 0))
]
if not inconsistent_casualties.empty:
    inconsistent_casualties[['eventid', 'nkill', 'nkillter', 'nkillus', 'nwound', 'nwoundter', 'nwoundus', 'success']].to_csv(dropped_casualties_path, index=False)
    logging.info(f"Saved {len(inconsistent_casualties)} inconsistent casualties to: {dropped_casualties_path}")
    df.loc[df['nkillter'] > df['nkill'], 'nkillter'] = df['nkill']
    df.loc[df['nwoundter'] > df['nwound'], 'nwoundter'] = df['nwound']
    df.loc[df['nkillus'] > df['nkill'], 'nkillus'] = df['nkill']
    df.loc[df['nwoundus'] > df['nwound'], 'nwoundus'] = df['nwound']
    logging.info("Corrected inconsistent casualties")

# 15. Clean summary text
def clean_text_field(text):
    if pd.isna(text) or text == '' or text is None:
        return ''
    text = normalize_text(text)
    abbreviation_map = {
        'govt': 'government', 'mil': 'military', 'org': 'organization', 
        'intl': 'international', 'terr': 'terrorist', 'atk': 'attack'
    }
    term_map = {
        'attackers': 'attackers', 'terrorists': 'attackers', 'militants': 'attackers', 
        'bombed': 'bombing', 'exploded': 'explosion'
    }
    for abbrev, full in abbreviation_map.items():
        text = text.replace(abbrev, full)
    for term, standard in term_map.items():
        text = text.replace(term, standard)
    text = re.sub(r'no group claimed responsibility', '', flags=re.IGNORECASE)
    return text[:500]

if 'summary' in df.columns:
    initial_length = df['summary'].str.len().mean()
    df['summary'] = df['summary'].apply(clean_text_field)
    df['claimed_flag'] = 0
    df.loc[df['summary'].str.contains('claimed|responsibility', case=False, na=False), 'claimed_flag'] = 1
    invalid_summaries = df[df['summary'].str.contains(r'[^\x00-\x7F]', na=False) | (df['summary'].str.len() > 500)]
    if not invalid_summaries.empty:
        logging.warning(f"Found {len(invalid_summaries)} invalid summaries")
    logging.info(f"Cleaned summary: average length from {initial_length:.2f} to {df['summary'].str.len().mean():.2f} characters")
    logging.info(f"Flagged {df['claimed_flag'].sum()} rows with claimed_flag=1")
    print(f"📌 Flagged {df['claimed_flag'].sum()} claimed rows")

# 16. Validate attack_type and weapon_type
attack_weapon_map = {
    'assassination': ['firearms', 'melee', 'unknown'],
    'armed assault': ['firearms', 'melee', 'unknown'],
    'bombing/explosion': ['explosives', 'unknown'],
    'hijacking': ['firearms', 'vehicle', 'unknown'],
    'hostage taking (kidnapping)': ['firearms', 'melee', 'unknown'],
    'hostage taking (barricade incident)': ['firearms', 'explosives', 'unknown'],
    'facility/infrastructure attack': ['explosives', 'incendiary', 'unknown'],
    'unarmed assault': ['melee', 'unknown'],
    'unknown': ['unknown']
}
weapon_subtype_map = {
    'explosives': ['grenade', 'landmine', 'bomb', 'unknown'],
    'firearms': ['rifle', 'pistol', 'machine gun', 'unknown'],
    'melee': ['knife', 'blunt object', 'unknown'],
    'incendiary': ['molotov', 'arson', 'unknown'],
    'vehicle': ['car', 'truck', 'unknown'],
    'unknown': ['unknown']
}

def impute_weapon_type(row):
    if row['weapon_type'] == 'unknown' and row['attack_type'] in attack_weapon_map:
        valid_weapons = attack_weapon_map[row['attack_type']]
        return valid_weapons[0] if valid_weapons != ['unknown'] else 'unknown'
    return row['weapon_type']

if 'weapon_type' in df.columns and 'attack_type' in df.columns:
    df['weapon_type'] = df.apply(impute_weapon_type, axis=1)
    inconsistent_attack_weapon = df[df.apply(lambda x: x['weapon_type'] not in attack_weapon_map.get(x['attack_type'], ['unknown']), axis=1)]
    if not inconsistent_attack_weapon.empty:
        inconsistent_attack_weapon[['eventid', 'attack_type', 'weapon_type']].to_csv(dropped_attack_weapon_path, index=False)
        df.loc[df.apply(lambda x: x['weapon_type'] not in attack_weapon_map.get(x['attack_type'], ['unknown']), axis=1), 'weapon_type'] = 'unknown'
        logging.info(f"Saved {len(inconsistent_attack_weapon)} inconsistent attack_weapon rows")
    df['attack_weapon_mismatch_flag'] = 0
    df.loc[~df.apply(lambda x: x['weapon_type'] in attack_weapon_map.get(x['attack_type'], ['unknown']), axis=1), 'attack_weapon_mismatch_flag'] = 1
    logging.info(f"Flagged {df['attack_weapon_mismatch_flag'].sum()} attack_weapon_mismatch rows")
    print(f"📌 Flagged {df['attack_weapon_mismatch_flag'].sum()} attack_weapon_mismatch rows")

if 'weapon_subtype' in df.columns:
    inconsistent_subtype = df[~df.apply(lambda x: x['weapon_subtype'] in weapon_subtype_map.get(x['weapon_type'], ['unknown']), axis=1)]
    if not inconsistent_subtype.empty:
        df.loc[inconsistent_subtype.index, 'weapon_subtype'] = 'unknown'
        logging.info(f"Set {len(inconsistent_subtype)} inconsistent weapon_subtype to 'unknown'")
        print(f"📌 Set {len(inconsistent_subtype)} inconsistent weapon_subtype to 'unknown'")

# 17. Validate ranges
binary_clip = {
    'success': (0, 1),
    'extended': (0, 1),
    'property': (-1, 1),
    'specificity': (1, 5),
    'vicinity': (0, 1),
    'uncertainty': (0, 1)
}
for col, (low, high) in binary_clip.items():
    if col in df.columns:
        df[col] = df[col].clip(lower=low, upper=high)

# 18. Validate measures
for col in ['nkill', 'nwound', 'nkillter', 'nwoundter', 'propvalue', 'nkillus', 'nwoundus']:
    if col in df.columns:
        df[col] = df[col].clip(lower=0)

# 19. Check for location duplicates
location_cols = ['region', 'country', 'provstate', 'city', 'latitude', 'longitude']
location_duplicates = df[location_cols].duplicated().sum()
if location_duplicates > 0:
    logging.warning(f"Found {location_duplicates} duplicate locations")

# Save cleaned dataset
df.to_csv(output_path, index=False, encoding='utf-8')
logging.info(f"Saved cleaned dataset: {output_path}, {len(df)} rows, {len(df.columns)} columns")
print(f"✅ Saved to: {output_path}")
print(f"📊 Rows: {len(df)}")
print(f"🧾 Columns: {list(df.columns)}")
print(f"📜 Log: {os.path.join(log_dir, 'cleaning_log.txt')}")
print(f"📌 Dropped files: {dropped_dates_path}, {dropped_unknown_path}, {dropped_propvalue_path}, "
      f"{dropped_casualties_path}, {dropped_attack_weapon_path}, {dropped_outliers_path}, {dropped_geo_inconsistent_path}")
print(f"📌 Missing coords: {missing_coords_path}")
print(f"📌 Unknown groups: {unknown_groups_path}")
print(f"📌 None groups: {none_groups_path}")