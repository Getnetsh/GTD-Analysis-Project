import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
import os
import re
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from rapidfuzz import process, fuzz

# Script version
SCRIPT_VERSION = "2025-05-21-v25"

# Setup logging
log_dir = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'etl_log.txt')
backup_log_file = os.path.join(log_dir, f'etl_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# Test log file writability
try:
    with open(log_file, 'a') as f:
        f.write('')
    with open(backup_log_file, 'a') as f:
        f.write('')
except Exception as e:
    print(f"Error: Cannot write to log file: {str(e)}")
    raise

# Custom filter for console to show only important INFO messages
class ConsoleFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno == logging.INFO and getattr(record, 'important', False):
            return True
        return False

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

backup_file_handler = logging.FileHandler(backup_log_file)
backup_file_handler.setLevel(logging.INFO)
backup_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(backup_file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.addFilter(ConsoleFilter())
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

logging.info(f"Starting ETL with script version: {SCRIPT_VERSION}", extra={'important': True})
logging.info(f"Script path: {os.path.abspath(__file__)}")
print(f"Running etl_gtd_data.py version {SCRIPT_VERSION}")

# Database connection
db_user = 'root'
db_password = '1234'
db_host = 'localhost'
terrorist_db_engine = create_engine(f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/Terrorist_db')
terrorist_dw_engine = create_engine(f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/Terrorist_dw')

# Input path
input_path = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed\gtd_cleaned_enhanced_v5.csv'
if not os.path.exists(input_path):
    logging.error(f"Input file not found: {input_path}")
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Read data
try:
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    logging.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns", extra={'important': True})
except Exception as e:
    logging.error(f"Failed to read input file: {str(e)}")
    raise

# Validate input
expected_cols = ['eventid', 'year', 'month', 'day', 'region', 'country', 'attack_type', 'target_type', 'group_name', 'extent']
missing_cols = [col for col in expected_cols if col not in df.columns]
if missing_cols:
    logging.error(f"Missing columns: {missing_cols}")
    raise ValueError(f"Missing columns: {missing_cols}")
if len(df) == 0:
    logging.error("Input file is empty")
    raise ValueError("Input file is empty")

# Validate no invalid dates
invalid_dates = df[(df['year'] == 0) | (df['month'] == 0) | (df['day'] == 0) | 
                  (df['year'] < 1970) | (df['month'] < 1) | (df['day'] < 1)]
if not invalid_dates.empty:
    logging.error(f"Found {len(invalid_dates)} rows with invalid dates:\n{invalid_dates[['eventid', 'year', 'month', 'day']].to_string()}")
    raise ValueError("Input data contains invalid dates")

# Validate eventid uniqueness
eventid_counts = df['eventid'].value_counts()
duplicate_eventids = eventid_counts[eventid_counts > 1]
if not duplicate_eventids.empty:
    logging.error(f"Found {len(duplicate_eventids)} duplicate eventid:\n{duplicate_eventids.head().to_string()}")
    raise ValueError("Duplicate eventid in input data")

# Clean location columns
def clean_string(s):
    if pd.isna(s):
        return 'unknown'
    s = str(s).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\x00-\x7F]+', '', s)
    mappings = {'yunnan': 'yunnan province', 'yunnan prov': 'yunnan province'}
    return mappings.get(s, s)

def format_coord(x):
    if pd.isna(x) or x == 0:
        return 0.0
    return round(x, 3)

for col in ['region', 'country', 'provstate', 'city']:
    df[col] = df[col].apply(clean_string)
df['latitude'] = df['latitude'].fillna(0).apply(format_coord)
df['longitude'] = df['longitude'].fillna(0).apply(format_coord)
df['specificity'] = df['specificity'].fillna(0).astype(int)
df['vicinity'] = df['vicinity'].fillna(0).astype(int)
df['unknown_location_flag'] = df['unknown_location_flag'].fillna(0).astype(int)
df['group_name'] = df['group_name'].apply(clean_string)
df['approxdate'] = df['approxdate'].apply(clean_string)

# Optimized fuzzy matching for city
def fuzzy_dedupe(df, col, threshold=90, min_freq=5):
    value_counts = df[col].value_counts()
    frequent_values = value_counts[value_counts >= min_freq].index.tolist()
    unique_values = [v for v in df[col].unique() if v in frequent_values and v != 'unknown']
    matches = {}
    for value in unique_values:
        close_matches = process.extract(value, unique_values, scorer=fuzz.ratio, limit=None)
        matches[value] = [m[0] for m in close_matches if m[1] >= threshold and m[0] != value]
    for main_val, dupes in matches.items():
        for dupe in dupes:
            df.loc[df[col] == dupe, col] = main_val
    logging.info(f"Deduplicated {col}: {len(unique_values)} frequent values processed")
    return df

df = fuzzy_dedupe(df, 'city', threshold=90, min_freq=5)

# Clear existing data
with terrorist_db_engine.connect() as conn:
    trans = conn.begin()
    try:
        for table in ['properties', 'terroristgroups', 'casualties', 'targets', 'attacks', 'locations', 'incidents']:
            conn.execute(text(f"DELETE FROM {table}"))
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn)['count'].iloc[0]
            logging.info(f"Deleted all rows from Terrorist_db.{table}, row count: {count}")
        trans.commit()
    except Exception as e:
        trans.rollback()
        logging.error(f"Failed to delete rows in Terrorist_db: {str(e)}")
        raise

with terrorist_dw_engine.connect() as conn:
    trans = conn.begin()
    try:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for table in ['fact_incidents', 'dim_propertydamage', 'dim_terroristgroup', 'dim_targettype', 'dim_attacktype', 'dim_time', 'dim_location']:
            conn.execute(text(f"TRUNCATE TABLE {table}"))
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn)['count'].iloc[0]
            logging.info(f"Truncated Terrorist_dw.{table}, row count: {count}")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        trans.commit()
    except Exception as e:
        trans.rollback()
        logging.error(f"Failed to truncate tables in Terrorist_dw: {str(e)}")
        raise
logging.info("Cleared existing data in Terrorist_db and Terrorist_dw")

# --- Terrorist_db ETL ---
logging.info("Starting ETL for Terrorist_db")
for table, cols in [
    ('incidents', ['eventid', 'year', 'month', 'day', 'approxdate', 'extended', 'success', 'summary', 'motive']),
    ('locations', ['eventid', 'region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location_flag']),
    ('attacks', ['eventid', 'attack_type', 'weapon_type', 'weapon_subtype']),
    ('targets', ['eventid', 'target_type', 'target_subtype', 'nationality']),
    ('casualties', ['eventid', 'nkill', 'nwound', 'nkillter', 'nwoundter', 'nkillus', 'nwoundus']),
    ('terroristgroups', ['eventid', 'group_name', 'subgroup_name', 'uncertainty', 'unknown_group_flag']),
    ('properties', ['eventid', 'propvalue', 'extent', 'property'])
]:
    df_table = df[cols].drop_duplicates()
    try:
        df_table.to_sql(table, terrorist_db_engine, if_exists='append', index=False)
        logging.info(f"Loaded {len(df_table)} rows into Terrorist_db.{table}")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load Terrorist_db.{table}: {str(e)}")
        raise

# --- Terrorist_dw ETL ---
logging.info("Starting ETL for Terrorist_dw", extra={'important': True})

# dim_time
default_time = df[['year', 'month', 'day', 'approxdate', 'extended']].drop_duplicates().reset_index(drop=True)
logging.info(f"Initial dim_time rows: {len(default_time)}")
# Filter valid dates
default_time = default_time[
    (default_time['year'] >= 1970) &
    (default_time['month'].between(1, 12)) &
    (default_time['day'].between(1, 31)) &
    (default_time['year'].notna()) &
    (default_time['month'].notna()) &
    (default_time['day'].notna())
]
logging.info(f"dim_time rows after filtering invalid dates: {len(default_time)}")
# Validate no invalid dates
invalid_time_rows = default_time[
    (default_time['year'] < 1970) | 
    (default_time['month'] < 1) | 
    (default_time['month'] > 12) | 
    (default_time['day'] < 1) | 
    (default_time['day'] > 31)
]
if not invalid_time_rows.empty:
    logging.error(f"Found {len(invalid_time_rows)} invalid date rows in dim_time:\n{invalid_time_rows.to_string()}")
    raise ValueError("Invalid dates in dim_time")
default_time['time_id'] = range(1, len(default_time) + 1)
default_time = default_time[['time_id', 'year', 'month', 'day', 'approxdate', 'extended']]
try:
    default_time.to_sql('dim_time', terrorist_dw_engine, if_exists='append', index=False)
    logging.info(f"Loaded {len(default_time)} rows into dim_time")
except SQLAlchemyError as e:
    logging.error(f"Failed to load dim_time: {str(e)}")
    raise

# dim_terroristgroup (insert default 'unknown' group first)
default_group = pd.DataFrame({
    'group_name': ['unknown'], 
    'subgroup_name': ['unknown'], 
    'uncertainty': [1], 
    'unknown_group_flag': [1]
})
default_group['group_id'] = 1
try:
    default_group.to_sql('dim_terroristgroup', terrorist_dw_engine, if_exists='append', index=False)
    logging.info("Inserted default terrorist group record")
except SQLAlchemyError as e:
    logging.error(f"Failed to insert default terrorist group: {str(e)}")
    raise

# dim_location
default_location = df[[
    'region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location_flag'
]].drop_duplicates().reset_index(drop=True)
default_location['latitude'] = default_location['latitude'].apply(format_coord)
default_location['longitude'] = default_location['longitude'].apply(format_coord)
default_location['location_id'] = range(1, len(default_location) + 1)
default_location = default_location[[
    'location_id', 'region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location_flag'
]]
logging.info(f"Yunnan province rows:\n{default_location[default_location['provstate'] == 'yunnan province'][['city', 'latitude', 'longitude']].to_string()}")
try:
    default_location.to_sql('dim_location', terrorist_dw_engine, if_exists='append', index=False)
    logging.info(f"Loaded {len(default_location)} rows into dim_location", extra={'important': True})
except SQLAlchemyError as e:
    logging.error(f"Failed to load dim_location: {str(e)}")
    raise

# dim_attacktype
default_attacktype = df[['attack_type', 'weapon_type', 'weapon_subtype']].drop_duplicates().reset_index(drop=True)
default_attacktype['attack_type_id'] = range(1, len(default_attacktype) + 1)
default_attacktype = default_attacktype[['attack_type_id', 'attack_type', 'weapon_type', 'weapon_subtype']]
try:
    default_attacktype.to_sql('dim_attacktype', terrorist_dw_engine, if_exists='append', index=False)
    logging.info(f"Loaded {len(default_attacktype)} rows into dim_attacktype")
except SQLAlchemyError as e:
    logging.error(f"Failed to load dim_attacktype: {str(e)}")
    raise

# dim_targettype
default_targettype = df[['target_type', 'target_subtype', 'nationality']].drop_duplicates().reset_index(drop=True)
default_targettype['target_type_id'] = range(1, len(default_targettype) + 1)
default_targettype = default_targettype[['target_type_id', 'target_type', 'target_subtype', 'nationality']]
try:
    default_targettype.to_sql('dim_targettype', terrorist_dw_engine, if_exists='append', index=False)
    logging.info(f"Loaded {len(default_targettype)} rows into dim_targettype")
except SQLAlchemyError as e:
    logging.error(f"Failed to load dim_targettype: {str(e)}")
    raise

# dim_terroristgroup (remaining groups)
default_terroristgroup = df[['group_name', 'subgroup_name', 'uncertainty', 'unknown_group_flag']].drop_duplicates().reset_index(drop=True)
default_terroristgroup['group_id'] = range(2, len(default_terroristgroup) + 2)  # Start after default group_id=1
default_terroristgroup = default_terroristgroup[['group_id', 'group_name', 'subgroup_name', 'uncertainty', 'unknown_group_flag']]
try:
    default_terroristgroup.to_sql('dim_terroristgroup', terrorist_dw_engine, if_exists='append', index=False)
    logging.info(f"Loaded {len(default_terroristgroup)} rows into dim_terroristgroup")
except SQLAlchemyError as e:
    logging.error(f"Failed to load dim_terroristgroup: {str(e)}")
    raise

# dim_propertydamage
default_propertydamage = df[['extent', 'property']].drop_duplicates().reset_index(drop=True)
default_propertydamage['property_id'] = range(1, len(default_propertydamage) + 1)
default_propertydamage = default_propertydamage[['property_id', 'extent', 'property']]
try:
    default_propertydamage.to_sql('dim_propertydamage', terrorist_dw_engine, if_exists='append', index=False)
    logging.info(f"Loaded {len(default_propertydamage)} rows into dim_propertydamage")
except SQLAlchemyError as e:
    logging.error(f"Failed to load dim_propertydamage: {str(e)}")
    raise

# Fact_Incidents
df_fact = df.copy()
logging.info(f"Initial df_fact rows: {len(df_fact)}", extra={'important': True})

# Fetch dimension IDs
dim_tables = {
    'dim_time': ['time_id', 'year', 'month', 'day', 'approxdate', 'extended'],
    'dim_location': ['location_id', 'region', 'country', 'provstate', 'city', 'latitude', 'longitude'],
    'dim_attacktype': ['attack_type_id', 'attack_type', 'weapon_type', 'weapon_subtype'],
    'dim_targettype': ['target_type_id', 'target_type', 'target_subtype', 'nationality'],
    'dim_terroristgroup': ['group_id', 'group_name', 'subgroup_name', 'uncertainty'],
    'dim_propertydamage': ['property_id', 'extent', 'property']
}
dim_dfs = {}
for table, cols in dim_tables.items():
    try:
        dim_dfs[table] = pd.read_sql(f"SELECT {', '.join(cols)} FROM {table}", terrorist_dw_engine)
        dim_dfs[table][cols[1:]] = dim_dfs[table][cols[1:]].apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        if table == 'dim_location':
            dim_dfs[table]['latitude'] = dim_dfs[table]['latitude'].apply(format_coord)
            dim_dfs[table]['longitude'] = dim_dfs[table]['longitude'].apply(format_coord)
        logging.info(f"Loaded {len(dim_dfs[table])} rows from {table}, columns: {dim_dfs[table].columns.tolist()}")
    except SQLAlchemyError as e:
        logging.error(f"Failed to fetch {table}: {str(e)}")
        raise

# Create location_key
df_fact['location_key'] = (
    df_fact['region'] + '|' + df_fact['country'] + '|' + df_fact['provstate'] + '|' + 
    df_fact['city'] + '|' + df_fact['latitude'].apply(lambda x: f"{x:.3f}") + '|' + 
    df_fact['longitude'].apply(lambda x: f"{x:.3f}")
)
if not dim_dfs['dim_location'].empty:
    dim_dfs['dim_location']['location_key'] = (
        dim_dfs['dim_location']['region'] + '|' + dim_dfs['dim_location']['country'] + '|' + 
        dim_dfs['dim_location']['provstate'] + '|' + dim_dfs['dim_location']['city'] + '|' + 
        dim_dfs['dim_location']['latitude'].apply(lambda x: f"{x:.3f}") + '|' + 
        dim_dfs['dim_location']['longitude'].apply(lambda x: f"{x:.3f}")
    )

# Merge dimensions
for table, cols in dim_tables.items():
    key_cols = cols[1:]
    id_col = cols[0]
    df_fact = df_fact.merge(dim_dfs[table][[id_col] + key_cols], on=key_cols, how='left')
    logging.info(f"{table} merge resulted in {len(df_fact)} rows")
    
    # Deduplicate after each merge
    duplicates = df_fact[df_fact.duplicated(subset=['eventid'])]
    if not duplicates.empty:
        logging.warning(f"Found {len(duplicates)} duplicate eventid after {table} merge:\n{duplicates[['eventid', id_col]].head().to_string()}")
        df_fact = df_fact.sort_values(by=['eventid', id_col], na_position='last').drop_duplicates(subset=['eventid'], keep='first')
        logging.info(f"After deduplication post-{table} merge, df_fact has {len(df_fact)} rows")

# Handle missing group_id
missing_group = df_fact[df_fact['group_id'].isna()]
if not missing_group.empty:
    logging.info(f"Found {len(missing_group)} rows with missing group_id", extra={'important': True})
    default_group_id = pd.read_sql("SELECT group_id FROM dim_terroristgroup WHERE group_name = 'unknown'", terrorist_dw_engine)['group_id'].iloc[0]
    df_fact.loc[df_fact['group_id'].isna(), 'group_id'] = default_group_id
    logging.info(f"Assigned default group_id {default_group_id} to {len(missing_group)} rows")

# Handle missing location_id
missing_loc = df_fact[df_fact['location_id'].isna()]
if not missing_loc.empty:
    logging.info(f"Found {len(missing_loc)} rows with missing location_id:\n{missing_loc[['region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'location_key']].head().to_string()}")
    
    if not dim_dfs['dim_location'].empty and 'location_id' in dim_dfs['dim_location'].columns:
        # Merge on region, country, provstate, city
        fallback_cols = ['region', 'country', 'provstate', 'city']
        missing_loc = missing_loc.merge(
            dim_dfs['dim_location'][['location_id'] + fallback_cols],
            on=fallback_cols,
            how='left'
        )
        matched_rows = len(missing_loc[~missing_loc['location_id'].isna()])
        logging.info(f"Fallback merge matched {matched_rows} rows")
        
        # Try location_key for remaining
        still_missing = missing_loc[missing_loc['location_id'].isna()]
        if not still_missing.empty:
            still_missing = still_missing.merge(
                dim_dfs['dim_location'][['location_id', 'location_key']],
                on='location_key',
                how='left'
            )
            matched_rows = len(still_missing[~still_missing['location_id'].isna()])
            logging.info(f"Location_key merge matched {matched_rows} rows")
            missing_loc = missing_loc.combine_first(still_missing[['eventid', 'location_id']])
    
    # Assign temporary location_id for unmatched
    still_missing = missing_loc[missing_loc['location_id'].isna()]
    if not still_missing.empty:
        logging.info(f"Assigning temporary location_id for {len(still_missing)} unmatched rows")
        new_locations = still_missing[['region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location_flag']].drop_duplicates()
        new_locations['latitude'] = new_locations['latitude'].apply(format_coord)
        new_locations['longitude'] = new_locations['longitude'].apply(format_coord)
        new_locations = new_locations.drop_duplicates(subset=['region', 'country', 'provstate', 'city', 'latitude', 'longitude'])
        try:
            new_locations.to_sql('dim_location', terrorist_dw_engine, if_exists='append', index=False)
            logging.info(f"Inserted {len(new_locations)} new locations")
        except SQLAlchemyError as e:
            logging.error(f"Failed to insert new locations: {str(e)}")
            raise
        
        # Fetch new location_id values
        dim_dfs['dim_location'] = pd.read_sql("SELECT location_id, region, country, provstate, city, latitude, longitude, specificity, vicinity, unknown_location_flag FROM dim_location", terrorist_dw_engine)
        dim_dfs['dim_location'][['region', 'country', 'provstate', 'city']] = dim_dfs['dim_location'][['region', 'country', 'provstate', 'city']].apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        dim_dfs['dim_location']['latitude'] = dim_dfs['dim_location']['latitude'].apply(format_coord)
        dim_dfs['dim_location']['longitude'] = dim_dfs['dim_location']['longitude'].apply(format_coord)
        dim_dfs['dim_location']['location_key'] = (
            dim_dfs['dim_location']['region'] + '|' + dim_dfs['dim_location']['country'] + '|' + 
            dim_dfs['dim_location']['provstate'] + '|' + dim_dfs['dim_location']['city'] + '|' + 
            dim_dfs['dim_location']['latitude'].apply(lambda x: f"{x:.3f}") + '|' + 
            dim_dfs['dim_location']['longitude'].apply(lambda x: f"{x:.3f}")
        )
        missing_loc = missing_loc.merge(dim_dfs['dim_location'][['location_id', 'location_key']], on='location_key', how='left')
        matched_rows = len(missing_loc[~missing_loc['location_id'].isna()])
        logging.info(f"Final merge matched {matched_rows} rows")
    
    # Update df_fact
    if not missing_loc.empty:
        missing_loc = missing_loc[['eventid', 'location_id']].rename(columns={'location_id': 'location_id_temp'})
        df_fact = df_fact.merge(missing_loc, on='eventid', how='left')
        df_fact['location_id'] = df_fact['location_id'].combine_first(df_fact['location_id_temp'])
        df_fact = df_fact.drop(columns=['location_id_temp'], errors='ignore')

# Final deduplication
duplicates = df_fact[df_fact.duplicated(subset=['eventid'])]
if not duplicates.empty:
    logging.warning(f"Found {len(duplicates)} duplicate eventid before final load:\n{duplicates[['eventid']].head().to_string()}")
    df_fact = df_fact.drop_duplicates(subset=['eventid'], keep='first')
    logging.info(f"After final deduplication, df_fact has {len(df_fact)} rows")

# Log missing IDs
missing_ids = {col: df_fact[col].isna().sum() for col in ['time_id', 'location_id', 'attack_type_id', 'target_type_id', 'group_id', 'property_id']}
logging.info(f"Missing dimension IDs: {missing_ids}", extra={'important': True})
if missing_ids['location_id'] > 0:
    logging.warning(f"Sample rows with missing location_id:\n{df_fact[df_fact['location_id'].isna()][['eventid', 'region', 'country', 'provstate', 'city', 'latitude', 'longitude']].head().to_string()}")

# Load fact_incidents
df_fact = df_fact[[
    'eventid', 'time_id', 'location_id', 'attack_type_id', 'target_type_id', 'group_id', 'property_id',
    'nkill', 'nwound', 'nkillter', 'nwoundter', 'nkillus', 'nwoundus', 'propvalue', 'success', 'consistency_flag'
]].drop_duplicates()
try:
    df_fact.to_sql('fact_incidents', terrorist_dw_engine, if_exists='append', index=False)
    logging.info(f"Loaded {len(df_fact)} rows into fact_incidents", extra={'important': True})
except SQLAlchemyError as e:
    logging.error(f"Failed to load fact_incidents: {str(e)}")
    raise

print(f"✅ ETL completed. Check {log_file} for details.")
logging.info("ETL completed successfully", extra={'important': True})