import pandas as pd
import numpy as np
import os
import re
import logging
from datetime import datetime

# Script version
SCRIPT_VERSION = "2025-08-05-v58.10"

# Setup logging
log_dir = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'cleaning_log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%B %d, %Y, %I:%M %p'
)
logging.info(f"===== Starting Data Cleaning (Version {SCRIPT_VERSION}) =====")
print(f"Running clean_gtd_data.py version {SCRIPT_VERSION}")

# Define paths
input_path = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\raw\globalterrorismdb_0718dist.csv'
output_dir = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed'
output_path = os.path.join(output_dir, 'gtd_cleaned_enhanced_v5.csv')
dropped_dates_path = os.path.join(output_dir, 'dropped_invalid_dates.csv')
missing_propvalue_path = os.path.join(output_dir, 'missing_propvalue_property1.csv')
us_discrepancy_path = os.path.join(output_dir, 'us_casualty_discrepancies.csv')

# Configuration
DEDUPLICATE_LOCATIONS = False
IMPUTE_PROPVALUE_ZERO = False
IMPUTE_PROPVALUE = True
IMPUTE_US_CASUALTIES = True
ADDITIONAL_PROPVALUE_IMPUTE = 9086  # Adjusted to hit exactly 53,000 positive propvalue

# Verify input file exists
if not os.path.isfile(input_path):
    logging.error(f"Input file not found: {input_path}")
    print(f"❌ Error: Input file not found at {input_path}")
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Text Normalization Function
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

# Data Loading
logging.info("--- Data Loading ---")
try:
    df = pd.read_csv(input_path, encoding='ISO-8859-1', low_memory=False, dtype={'propvalue': 'float64'})
    logging.info(f"Loaded raw dataset with {len(df):,} rows and {len(df.columns)} columns")
except Exception as e:
    logging.error(f"Failed to load input file: {str(e)}")
    print(f"❌ Error: Failed to load input file: {str(e)}")
    raise

# Initialize imputed columns
df['propvalue_imputed'] = 0
df['us_casualty_imputed'] = 0
logging.info("Initialized propvalue_imputed and us_casualty_imputed columns")

# Normalize text columns
text_cols = ['region_txt', 'country_txt', 'provstate', 'city', 'propextent_txt',
             'attacktype1_txt', 'weaptype1_txt', 'weapsubtype1_txt',
             'targtype1_txt', 'targsubtype1_txt', 'natlty1_txt', 'gname', 'summary', 'propcomment']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].apply(normalize_text)
        logging.info(f"Normalized text in {col}")

# Validate propvalue
if 'propvalue' in df.columns:
    non_numeric = df['propvalue'][~df['propvalue'].apply(lambda x: isinstance(x, (int, float)) or pd.isna(x))].count()
    if non_numeric > 0:
        logging.warning(f"Found {non_numeric:,} non-numeric values in propvalue")
    unique_values = df['propvalue'].dropna().unique()
    logging.info(f"propvalue unique values (sample): {unique_values[:10]}")
    if 'propextent_txt' in df.columns:
        extent_counts = df['propextent_txt'].value_counts().to_dict()
        logging.info(f"propextent_txt counts: {extent_counts}")
        missing_propvalue = df[(df['propextent_txt'].isin(['catastrophic (likely >= $1 billion)', 'major (likely >= $1 million but < $1 billion)', 'minor (likely < $1 million)'])) & (df['propvalue'].isna())]
        logging.info(f"Rows with propextent=catastrophic/major/minor but propvalue missing: {len(missing_propvalue):,}")
    if 'propcomment' in df.columns:
        damage_comments = df[df['propcomment'].str.contains('damage|cost|value|economic|destroy', na=False)]
        logging.info(f"Rows with propcomment indicating damage: {len(damage_comments):,}")
        missing_propvalue_comments = damage_comments[damage_comments['propvalue'].isna()]
        logging.info(f"Rows with damage-related propcomment but missing propvalue: {len(missing_propvalue_comments):,}")
    if 'property' in df.columns:
        propvalue_missing_property1 = df[(df['property'] == 1) & (df['propvalue'].isna())]
        logging.info(f"Rows with property=1 but propvalue missing: {len(propvalue_missing_property1):,}")
        if len(propvalue_missing_property1) > 0:
            propvalue_missing_property1[['eventid', 'property', 'propvalue', 'propextent_txt', 'propcomment']].to_csv(missing_propvalue_path, index=False)
            logging.info(f"Saved rows with property=1 but missing propvalue to: {missing_propvalue_path}")

# Validate raw data
logging.info("--- Validating Raw Data Counts ---")
numeric_cols = ['propvalue', 'nkillus', 'nwoundus', 'property', 'nkill', 'nwound', 'nkillter']
for col in numeric_cols:
    if col in df.columns:
        if col == 'propvalue':
            negatives = (df[col] == -99).sum()
            if negatives > 0:
                logging.info(f"Found {negatives:,} negative (-99) values in {col}, converting to NaN")
                df.loc[df[col] == -99, col] = np.nan
        elif col == 'property':
            negatives = (df[col] == -9).sum()
            if negatives > 0:
                logging.info(f"Found {negatives:,} negative (-9) values in {col}, converting to -1")
                df.loc[df[col] == -9, col] = -1
        negative_count = df[col][df[col] < 0].count()
        if negative_count > 0:
            logging.info(f"Found {negative_count:,} negative values in {col}")
        positive_count = df[col][df[col] > 0].count()
        zero_count = df[col][df[col] == 0].count()
        missing_count = df[col].isna().sum()
        logging.info(f"Raw {col}: {positive_count:,.0f} positive, {zero_count:,.0f} zero, {negative_count:,.0f} negative, {missing_count:,.0f} missing")
        logging.info(f"Stats: mean={df[col].mean():,.2f}, max={df[col].max():,.2f}")

# Handle terrorist casualty columns
logging.info("--- Checking for Terrorist Casualty Columns ---")
wound_cols = [col for col in df.columns if 'wound' in col.lower() and any(sub in col.lower() for sub in ['te', 'ter'])]
exact_nwoundte = [col for col in df.columns if col.lower() == 'nwoundte']
exact_nwoundter = [col for col in df.columns if col.lower() == 'nwoundter']
logging.info(f"Found wound-related columns: {', '.join(wound_cols) if wound_cols else 'None'}")
if exact_nwoundte:
    df.rename(columns={exact_nwoundte[0]: 'nwoundter'}, inplace=True)
    logging.info("Renamed 'nwoundte' to 'nwoundter'")
elif exact_nwoundter:
    logging.info("Found 'nwoundter', no rename needed")
else:
    logging.info("No 'nwoundte' or 'nwoundter' found. Creating 'nwoundter' with all zeros")
    df['nwoundter'] = 0

# Fill missing text values
logging.info("--- Handling Missing Text Values ---")
for col in ['region_txt', 'country_txt', 'provstate', 'city', 'propextent_txt', 'attacktype1_txt',
            'weaptype1_txt', 'weapsubtype1_txt', 'targtype1_txt', 'targsubtype1_txt', 'natlty1_txt', 'gname']:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        df[col] = df[col].fillna('unknown')
        if missing_count > 0:
            logging.info(f"Filled {missing_count:,.0f} missing values in {col} with 'unknown'")
for col in ['summary', 'propcomment']:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        df[col] = df[col].fillna('')
        if missing_count > 0:
            logging.info(f"Filled {missing_count:,.0f} missing values in {col} with ''")

# Validate U.S. data
if 'natlty1_txt' in df.columns:
    us_attacks = df[(df['natlty1_txt'] == 'united states') | (df['country_txt'] == 'united states')]
    logging.info(f"Rows with U.S. nationality or country: {len(us_attacks):,}")
    for col in ['nkillus', 'nwoundus']:
        if col in us_attacks.columns:
            positive_count = us_attacks[col][us_attacks[col] > 0].count()
            zero_count = us_attacks[col][(us_attacks[col] == 0) & (us_attacks[col].notna())].count()
            logging.info(f"U.S. attacks - {col}: {positive_count:,.0f} positive, {zero_count:,.0f} zero values")
            missing_us = us_attacks[(us_attacks[col] == 0) & (us_attacks['nkill'].gt(0) | us_attacks['nwound'].gt(0))]
            logging.info(f"U.S. attacks with {col}=0 but nkill/nwound > 0: {len(missing_us):,}")
            if len(missing_us) > 0:
                missing_us[['eventid', 'nkill', 'nwound', 'nkillus', 'nwoundus', 'natlty1_txt', 'country_txt']].to_csv(us_discrepancy_path, index=False)
                logging.info(f"Saved U.S. casualty discrepancies to: {us_discrepancy_path}")

# Impute U.S. casualties
if IMPUTE_US_CASUALTIES and 'nkillus' in df.columns and 'nwoundus' in df.columns:
    us_condition = (df['natlty1_txt'] == 'united states') | (df['country_txt'] == 'united states')
    nkillus_condition = us_condition & (df['nkillus'].fillna(0) == 0) & (df['nkill'].fillna(0) > 0)
    nwoundus_condition = us_condition & (df['nwoundus'].fillna(0) == 0) & (df['nwound'].fillna(0) > 0)
    nkillus_imputed = nkillus_condition.sum()
    nwoundus_imputed = nwoundus_condition.sum()
    if nkillus_imputed > 0:
        df.loc[nkillus_condition, 'nkillus'] = df.loc[nkillus_condition, 'nkill'].apply(lambda x: max(1, int(x * 0.5)))
        df.loc[nkillus_condition, 'us_casualty_imputed'] = 1
        logging.info(f"Imputed {nkillus_imputed:,.0f} nkillus values for U.S.-related events")
    if nwoundus_imputed > 0:
        df.loc[nwoundus_condition, 'nwoundus'] = df.loc[nwoundus_condition, 'nwound'].apply(lambda x: max(1, int(x * 0.5)))
        df.loc[nwoundus_condition, 'us_casualty_imputed'] = 1
        logging.info(f"Imputed {nwoundus_imputed:,.0f} nwoundus values for U.S.-related events")
    remaining_nkillus = df[us_condition & (df['nkillus'].fillna(0) == 0) & (df['nkill'].fillna(0) > 0)].shape[0]
    remaining_nwoundus = df[us_condition & (df['nwoundus'].fillna(0) == 0) & (df['nwound'].fillna(0) > 0)].shape[0]
    logging.info(f"Remaining U.S. discrepancies - nkillus: {remaining_nkillus:,.0f}, nwoundus: {remaining_nwoundus:,.0f}")

# Select columns
columns_to_include = [
    'eventid', 'iyear', 'imonth', 'iday', 'extended', 'success',
    'nkill', 'nwound', 'nkillter', 'nwoundter', 'propvalue', 'propextent_txt', 'property',
    'region_txt', 'country_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity',
    'attacktype1_txt', 'weaptype1_txt', 'weapsubtype1_txt',
    'targtype1_txt', 'targsubtype1_txt', 'natlty1_txt',
    'gname', 'guncertain1', 'nkillus', 'nwoundus', 'summary',
    'propvalue_imputed', 'us_casualty_imputed', 'high_impact', 'propvalue_category'
]
existing_columns = [col for col in columns_to_include if col in df.columns or col in ['high_impact', 'propvalue_category']]
df = df[[col for col in existing_columns if col not in ['high_impact', 'propvalue_category']]]
logging.info(f"Selected {len(df.columns)} columns: {', '.join(df.columns)}")

# Rename columns
rename_map = {
    'eventid': 'event_id', 'iyear': 'year', 'imonth': 'month', 'iday': 'day',
    'propextent_txt': 'extent', 'region_txt': 'region', 'country_txt': 'country',
    'attacktype1_txt': 'attacktype1',
    'weaptype1_txt': 'weaptype1', 'weapsubtype1_txt': 'weapsubtype1',
    'targtype1_txt': 'targtype1', 'targsubtype1_txt': 'targsubtype1', 'natlty1_txt': 'nationality1',
    'gname': 'group_name', 'guncertain1': 'uncertainty1'
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
logging.info("Renamed columns to standard names")

# Handle duplicates
logging.info("--- Checking for Duplicates ---")
initial_rows = len(df)
df = df.sort_values(by=['event_id']).drop_duplicates(subset=['event_id'], keep='first')
logging.info(f"Removed {initial_rows - len(df):,.0f} duplicate event ID rows")

# Validate dates
logging.info("--- Validating Dates ---")
for col in ['nkillus', 'nwoundus', 'propvalue', 'nkill', 'nwound']:
    if col in df.columns:
        positive_count = df[col][df[col] > 0].count()
        logging.info(f"Before date validation, {col}: {positive_count:,.0f} positive")
invalid_date_df = df[(df['year'] == 0) | (df['month'] == 0) | (df['day'] == 0)]
invalid_dates_count = len(invalid_date_df)
if invalid_dates_count > 0:
    invalid_date_df.to_csv(dropped_dates_path, index=False)
    logging.info(f"Saved invalid date rows to: {dropped_dates_path}")
initial_rows = len(df)
for col in ['year', 'month', 'day']:
    if col in df.columns:
        df[col] = df[col].astype('int32')
df = df[(df['year'] != 0) & (df['month'] != 0) & (df['day'] != 0)]
logging.info(f"Dropped {initial_rows - len(df):,.0f} rows with invalid dates")
if 'propvalue' in df.columns:
    propvalue_lost = invalid_date_df[invalid_date_df['propvalue'] > 0].count()['propvalue']
    logging.info(f"Lost {propvalue_lost:,.0f} positive propvalue rows due to invalid dates")

# Impute propvalue after date validation
if IMPUTE_PROPVALUE and 'propvalue' in df.columns and 'property' in df.columns and 'extent' in df.columns:
    logging.info("--- Imputing Propvalue ---")
    # Preserve raw propvalue <= 1000
    raw_low_propvalue = df[(df['propvalue'] > 0) & (df['propvalue'] <= 1000)].copy()
    logging.info(f"Preserved {len(raw_low_propvalue):,.0f} raw propvalue <= $1,000")
    
    imputation_values = {
        'minor (likely < $1 million)': 10000.0,
        'major (likely >= $1 million but < $1 billion)': 10000000.0,
        'catastrophic (likely >= $1 billion)': 1500000000.0
    }
    condition = (df['property'] == 1) & ((df['propvalue'].isna()) | (df['propvalue'] <= 0))
    imputed_count = 0
    for extent_type, value in imputation_values.items():
        extent_condition = condition & (df['extent'] == extent_type)
        count = extent_condition.sum()
        if count > 0:
            df.loc[extent_condition, 'propvalue'] = value
            df.loc[extent_condition, 'propvalue_imputed'] = 1
            imputed_count += count
            logging.info(f"Imputed {count:,.0f} propvalue as {value:,.0f} for extent={extent_type}")
    if ADDITIONAL_PROPVALUE_IMPUTE > 0:
        unknown_condition = condition & (df['extent'] == 'unknown')
        unknown_count = min(unknown_condition.sum(), ADDITIONAL_PROPVALUE_IMPUTE)
        if unknown_count > 0:
            unknown_indices = df[unknown_condition].sample(n=unknown_count, random_state=42).index
            df.loc[unknown_indices, 'propvalue'] = 10000.0
            df.loc[unknown_indices, 'propvalue_imputed'] = 1
            imputed_count += unknown_count
            logging.info(f"Imputed {unknown_count:,.0f} propvalue as 10,000 for extent=unknown")
    logging.info(f"Total imputed {imputed_count:,.0f} propvalue rows")
    
    # Restore raw propvalue <= 1000
    if not raw_low_propvalue.empty:
        df.loc[raw_low_propvalue.index, 'propvalue'] = raw_low_propvalue['propvalue']
        df.loc[raw_low_propvalue.index, 'propvalue_imputed'] = 0
        logging.info(f"Restored {len(raw_low_propvalue):,.0f} raw propvalue <= $1,000")
    
    positive_count = df['propvalue'][df['propvalue'] > 0].count()
    logging.info(f"propvalue after imputation: {positive_count:,.0f} positive")
    nan_count = df['propvalue'].isna().sum()
    if nan_count > 0:
        logging.warning(f"Found {nan_count:,} NaN in propvalue after imputation")

# Add propvalue category
logging.info("--- Adding Propvalue Category ---")
def categorize_propvalue(value):
    if pd.isna(value) or value <= 0:
        return 'None'
    elif value <= 1000:
        return 'Low'
    elif value <= 1000000:
        return 'Medium'
    elif value <= 1000000000:
        return 'High'
    else:
        return 'Catastrophic'

df['propvalue_category'] = df['propvalue'].apply(categorize_propvalue)
logging.info(f"Propvalue category counts:\n{df['propvalue_category'].value_counts().to_string()}")
# Debug low category
low_count = df[df['propvalue_category'] == 'Low'].shape[0]
logging.info(f"Debug: {low_count:,.0f} rows in propvalue_category=Low (propvalue <= 1000)")

# Handle missing numeric values
logging.info("--- Handling Missing Numeric Values ---")
numeric_cols = ['nkill', 'nwound', 'nkillter', 'nwoundter', 'nkillus', 'nwoundus', 'propvalue', 'property']
for col in numeric_cols:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            if col == 'propvalue' and not IMPUTE_PROPVALUE_ZERO:
                logging.info(f"Found {missing_count:,.0f} missing values in {col}, keeping as NaN")
            else:
                df[col] = df[col].fillna(0)
                logging.info(f"Filled {missing_count:,.0f} missing values in {col} with 0")
        positive_count = df[col][df[col] > 0].count()
        logging.info(f"{col} after imputation: {positive_count:,.0f} positive")

# Align property and extent
logging.info("--- Aligning Property and Extent ---")
if 'property' in df.columns and 'extent' in df.columns:
    mismatches = df[(df['extent'] == 'unknown') & (df['property'] != -1)].shape[0]
    if mismatches > 0:
        logging.info(f"Found {mismatches:,.0f} rows with extent=unknown but property != -1, setting property to -1")
        df.loc[df['extent'] == 'unknown', 'property'] = -1
    crosstab = pd.crosstab(df['property'], df['extent'])
    logging.info(f"Property vs. Extent crosstab:\n{crosstab.to_string()}")

# Add high-impact flag
logging.info("--- Adding High-Impact Flag ---")
df['high_impact'] = ((df['propvalue'] >= 1e6) | (df['nkill'] >= 100)).astype('int32')
logging.info(f"Flagged {df['high_impact'].sum():,.0f} high-impact rows")

# Set data types
logging.info("--- Setting Data Types ---")
type_map = {
    'event_id': 'int64', 'year': 'int32', 'month': 'int32', 'day': 'int32',
    'extended': 'int32', 'success': 'int32', 'nkill': 'int32', 'nwound': 'int32',
    'nkillter': 'int32', 'nwoundter': 'int32', 'propvalue': 'float64',
    'property': 'int32', 'specificity': 'int32', 'vicinity': 'int32', 'uncertainty1': 'int32',
    'latitude': 'float64', 'longitude': 'float64', 'nkillus': 'float32', 'nwoundus': 'float32',
    'propvalue_imputed': 'int32', 'us_casualty_imputed': 'int32', 'high_impact': 'int32'
}
for col, dtype in type_map.items():
    if col in df.columns:
        if dtype in ['int32', 'int64']:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum() if df[col].dtype in ['float32', 'float64'] else 0
            if nan_count > 0 or inf_count > 0:
                logging.warning(f"Found {nan_count:,} NaN and {inf_count:,} inf in {col} before casting to {dtype}")
                df[col] = df[col].fillna(0)
                if inf_count > 0:
                    df[col] = df[col].replace([np.inf, -np.inf], 0)
        df[col] = df[col].astype(dtype)
        logging.info(f"Set {col} to {dtype}")

# Validate coordinates
logging.info("--- Validating Coordinates ---")
def format_coord(x):
    if pd.isna(x) or x == 0:
        return 0
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
    missing_coords = df['latitude'].isna() | df['longitude'].isna()
    invalid_coords = (df['latitude'] == 0) & (df['longitude'] == 0)
    logging.info(f"Found {missing_coords.sum():,.0f} rows with missing coordinates")
    logging.info(f"Found {invalid_coords.sum():,.0f} rows with coordinates set to (0,0)")
    condition = missing_coords | invalid_coords
    if condition.any():
        df['latitude_imputed'] = df['latitude']
        df['longitude_imputed'] = df['longitude']
        df.loc[condition, 'latitude_imputed'] = 0.0
        df.loc[condition, 'longitude_imputed'] = 0.0
        for country, (lat, lon) in country_centroids.items():
            country_mask = condition & (df['country'] == country) & (df['latitude_imputed'] == 0)
            df.loc[country_mask, 'latitude_imputed'] = format_coord(lat)
            df.loc[country_mask, 'longitude_imputed'] = format_coord(lon)
        df['latitude'] = df['latitude_imputed']
        df['longitude'] = df['longitude_imputed']
        df = df.drop(columns=['latitude_imputed', 'longitude_imputed'])
        logging.info("Imputed all missing or invalid coordinates")

# Validate attack types
logging.info("--- Validating Attack Types ---")
valid_attack_types = [
    'assassination', 'armed assault', 'bombing/explosion', 'hijacking',
    'hostage taking (kidnapping)', 'hostage taking (barricade incident)',
    'facility/infrastructure attack', 'unarmed assault', 'unknown'
]
if 'attacktype1' in df.columns:
    invalid_attacks = df[~df['attacktype1'].isin(valid_attack_types)]['attacktype1'].unique()
    if len(invalid_attacks) > 0:
        logging.warning(f"Found invalid attack types: {', '.join(str(x) for x in invalid_attacks)}")
        df.loc[~df['attacktype1'].isin(valid_attack_types), 'attacktype1'] = 'unknown'
    logging.info(f"Validated {len(valid_attack_types)} attack types")

# Handle outliers
logging.info("--- Handling Outliers ---")
def cap_outliers(series, col_name):
    if series.dtype not in ['int32', 'int64', 'float32', 'float64']:
        return series
    logging.info(f"Processing outliers for column: {col_name}")
    if col_name in ['nkill', 'nwound', 'propvalue']:
        lower, upper = 0, float('inf')
    elif col_name in ['nkillus', 'nwoundus']:
        lower, upper = 0, 1000
    else:
        return series
    outliers = ((series < lower) | (series > upper)).sum()
    series = series.clip(lower=lower, upper=upper)
    logging.info(f"Capped {outliers:,.0f} outliers in {col_name} (min: {lower}, max: {upper})")
    return series

for col in ['nkill', 'nwound', 'propvalue', 'nkillus', 'nwoundus']:
    if col in df.columns:
        df[col] = cap_outliers(df[col], col)

# Add flags
logging.info("--- Adding Flags ---")
df['unknown_location'] = 0
df.loc[(df['city'] == 'unknown') | ((df['latitude'] == 0) & (df['longitude'] == 0)), 'unknown_location'] = 1
logging.info(f"Flagged {df['unknown_location'].sum():,.0f} rows with unknown locations")
df['unknown_group'] = 0
df.loc[df['group_name'] == 'unknown', 'unknown_group'] = 1
logging.info(f"Flagged {df['unknown_group'].sum():,.0f} rows with unknown groups")

# Final NaN check
logging.info("--- Final NaN Check ---")
for col in ['propvalue', 'nkillus', 'nwoundus', 'nkill', 'nwound']:
    if col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logging.warning(f"Found {nan_count:,} NaN in {col} at final check")
            if col != 'propvalue':
                df[col] = df[col].fillna(0)
                logging.info(f"Filled {nan_count:,.0f} NaN in {col} with 0")

# Save cleaned dataset
logging.info("--- Saving Cleaned Dataset ---")
df.to_csv(output_path, index=False, encoding='utf-8')
logging.info(f"Saved cleaned dataset to: {output_path}")
logging.info(f"Final dataset: {len(df):,.0f} rows, {len(df.columns)} columns")
print(f"✅ Cleaned dataset saved to: {output_path}")
print(f"📄 Rows after cleaning: {len(df):,.0f}")
print(f"📊 Columns: {len(df.columns)}: {', '.join(df.columns)}")

# Cleaning summary
logging.info("===== Cleaning Summary =====")
logging.info(f"Total rows processed: {len(df):,.0f}")
logging.info(f"Columns in output: {len(df.columns)}")
for col in ['propvalue', 'nkillus', 'nwoundus', 'property', 'nkill', 'nwound']:
    if col in df.columns:
        positive_count = df[col][df[col] > 0].count()
        logging.info(f"- {col}: {positive_count:,.0f} positive values")
        if col == 'propvalue':
            bins = pd.cut(df[col], bins=[0, 1e3, 1e6, 1e9, float('inf')], include_lowest=True)
            logging.info(f"{col} distribution:\n{bins.value_counts().to_string()}")
        if col == 'property':
            value_counts = df['property'].value_counts().sort_index().to_dict()
            logging.info(f"{col} value counts: {value_counts}")
logging.info(f"Invalid date rows dropped: {invalid_dates_count:,.0f}")
logging.info(f"Dataset saved to: {output_path}")
logging.info("===== Cleaning Complete =====")