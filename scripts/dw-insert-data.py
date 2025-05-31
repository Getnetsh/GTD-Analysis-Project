import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

# Script version
SCRIPT_VERSION = "2025-05-30-v10"

# Setup logging
log_dir = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'etl_terrorist_db_dw_log.txt')
backup_log_file = os.path.join(log_dir, f'etl_terrorist_db_dw_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# Test log file writability
try:
    with open(log_file, 'a') as f:
        f.write('')
    with open(backup_log_file, 'a') as f:
        f.write('')
except Exception as e:
    print(f"Error: Cannot write to log file: {str(e)}")
    raise

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
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

logging.info(f"Starting Terrorist_db and Terrorist_dw insertion with script version: {SCRIPT_VERSION}", extra={'important': True})
logging.info(f"Script path: {os.path.abspath(__file__)}")
print(f"Running Terrorist_db and Terrorist_dw insertion version {SCRIPT_VERSION}")

# Database connections
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

# Load CSV in chunks
chunk_size = 20000
chunks = pd.read_csv(input_path, encoding='utf-8', chunksize=chunk_size)

# Define table columns for Terrorist_db
terrorist_db_table_columns = {
    'incidents': ['eventid', 'year', 'month', 'day', 'extended', 'success', 'summary'],
    'locations': ['eventid', 'region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location'],
    'attacks': ['eventid', 'attacktype1', 'weaptype1', 'weapsubtype1'],
    'targets': ['eventid', 'targtype1', 'targsubtype1', 'nationality1'],
    'casualties': ['eventid', 'nkill', 'nwound', 'nkillter', 'nwoundter', 'nkillus', 'nwoundus'],
    'terroristgroups': ['eventid', 'group_name', 'uncertainty1', 'unknown_group'],
    'properties': ['eventid', 'propvalue', 'extent', 'property', 'propvalue_category']
}

# Clear existing data in Terrorist_db
with terrorist_db_engine.connect() as conn:
    trans = conn.begin()
    try:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        logging.info("Disabled foreign key checks for Terrorist_db")
        for table in terrorist_db_table_columns.keys():
            conn.execute(text(f"DELETE FROM {table}"))
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn)['count'].iloc[0]
            logging.info(f"Deleted all rows from Terrorist_db.{table}, row count: {count}")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        logging.info("Re-enabled foreign key checks for Terrorist_db")
        trans.commit()
    except Exception as e:
        trans.rollback()
        logging.error(f"Failed to delete rows in Terrorist_db: {str(e)}")
        raise
logging.info("Cleared existing data in Terrorist_db")

# Process chunks and insert into Terrorist_db
for i, chunk in enumerate(chunks):
    logging.info(f"Processing chunk {i+1} with {len(chunk)} rows", extra={'important': True})

    # Rename column to match schema
    chunk.rename(columns={'event_id': 'eventid'}, inplace=True)

    # Insert into each Terrorist_db table
    for table, cols in terrorist_db_table_columns.items():
        try:
            df_table = chunk[cols].copy()
            if table == 'incidents':
                df_table = df_table[
                    (df_table['year'].between(1970, 2017)) &
                    (df_table['month'].between(1, 12)) &
                    (df_table['day'].between(1, 31)) &
                    (df_table['extended'].isin([0, 1])) &
                    (df_table['success'].isin([0, 1]))
                ]
            elif table == 'locations':
                df_table = df_table[
                    (df_table['latitude'].between(-90, 90)) &
                    (df_table['longitude'].between(-180, 180)) &
                    (df_table['specificity'].between(1, 5) | df_table['specificity'].isna()) &
                    (df_table['vicinity'].isin([0, 1]) | df_table['vicinity'].isna()) &
                    (df_table['unknown_location'].isin([0, 1]))
                ]
            elif table in ['attacks', 'targets', 'terroristgroups', 'properties']:
                df_table = df_table.dropna(subset=[col for col in cols if col != 'eventid'])

            if table == 'casualties':
                df_table = df_table.fillna(0)

            df_table.to_sql(table, terrorist_db_engine, if_exists='append', index=False)
            logging.info(f"Loaded {len(df_table)} rows into Terrorist_db.{table} for chunk {i+1}")
        except SQLAlchemyError as e:
            logging.error(f"Failed to load Terrorist_db.{table} for chunk {i+1}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading Terrorist_db.{table} for chunk {i+1}: {str(e)}")
            raise

# Populate Terrorist_dw dimension tables
logging.info("Populating Terrorist_dw dimension tables")

# Clear existing data in Terrorist_dw
with terrorist_dw_engine.connect() as conn:
    trans = conn.begin()
    try:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        logging.info("Disabled foreign key checks for Terrorist_dw")
        for table in ['dim_time', 'dim_location', 'dim_attacktype', 'dim_targettype', 'dim_terroristgroup', 'dim_propertydamage', 'fact_incidents']:
            conn.execute(text(f"DROP TABLE IF EXISTS Terrorist_dw.{table}"))
            logging.info(f"Dropped table Terrorist_dw.{table}")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        logging.info("Re-enabled foreign key checks for Terrorist_dw")
        trans.commit()
    except Exception as e:
        trans.rollback()
        logging.error(f"Failed to drop tables in Terrorist_dw: {str(e)}")
        raise
logging.info("Cleared existing data in Terrorist_dw")

# Define and populate dimension tables with explicit schema
def populate_dimension_table(engine, schema, table_name, select_query, default_df, unique_cols, schema_def):
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            # Create table with explicit schema
            conn.execute(text(f"CREATE TABLE {schema}.{table_name} {schema_def}"))
            logging.info(f"Created table {schema}.{table_name}")
            # Load initial data
            df = pd.read_sql(select_query, terrorist_db_engine)
            df.to_sql(table_name, engine, schema=schema, if_exists='append', index=False)
            logging.info(f"Loaded initial data into {schema}.{table_name}")
            # Add default row
            default_df.to_sql(table_name, engine, schema=schema, if_exists='append', index=False)
            logging.info(f"Added default row to {schema}.{table_name}")
            # Read back and deduplicate
            df = pd.read_sql(f"SELECT * FROM {schema}.{table_name}", engine)
            df = df.drop_duplicates(subset=unique_cols, keep='first')
            # Truncate and reload deduplicated data
            conn.execute(text(f"TRUNCATE TABLE {schema}.{table_name}"))
            df.to_sql(table_name, engine, schema=schema, if_exists='append', index=False)
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            trans.commit()
            logging.info(f"Loaded {len(df)} rows into {schema}.{table_name} after deduplication")
        except Exception as e:
            trans.rollback()
            logging.error(f"Failed to populate {schema}.{table_name}: {str(e)}")
            raise

# Define schemas for dimension tables
dim_time_schema = """
    (
        time_id INT AUTO_INCREMENT PRIMARY KEY,
        year INT,
        month INT,
        day INT,
        extended INT
    )
"""
dim_location_schema = """
    (
        location_id INT AUTO_INCREMENT PRIMARY KEY,
        region VARCHAR(100),
        country VARCHAR(100),
        provstate VARCHAR(100),
        city VARCHAR(100),
        latitude FLOAT,
        longitude FLOAT,
        specificity INT,
        vicinity INT,
        unknown_location_flag INT
    )
"""
dim_attacktype_schema = """
    (
        attack_type_id INT AUTO_INCREMENT PRIMARY KEY,
        attack_type VARCHAR(100),
        weapon_type VARCHAR(100),
        weapon_subtype VARCHAR(100),
        UNIQUE (attack_type, weapon_type, weapon_subtype)
    )
"""
dim_targettype_schema = """
    (
        target_type_id INT AUTO_INCREMENT PRIMARY KEY,
        target_type VARCHAR(100),
        target_subtype VARCHAR(100),
        nationality VARCHAR(100),
        UNIQUE (target_type, target_subtype, nationality)
    )
"""
dim_terroristgroup_schema = """
    (
        group_id INT AUTO_INCREMENT PRIMARY KEY,
        group_name VARCHAR(255),
        uncertainty INT,
        unknown_group_flag INT,
        UNIQUE (group_name, uncertainty, unknown_group_flag)
    )
"""
dim_propertydamage_schema = """
    (
        property_id INT AUTO_INCREMENT PRIMARY KEY,
        extent VARCHAR(100),
        property INT,
        propvalue_category VARCHAR(100)
    )
"""
fact_incidents_schema = """
    (
        eventid BIGINT PRIMARY KEY,
        time_id INT,
        location_id INT,
        attack_type_id INT,
        target_type_id INT,
        group_id INT,
        property_id INT,
        nkill FLOAT,
        nwound FLOAT,
        nkillter FLOAT,
        nwoundter FLOAT,
        nkillus FLOAT,
        nwoundus FLOAT,
        propvalue FLOAT,
        success INT,
        propvalue_imputed INT,
        us_casualty_imputed INT,
        high_impact INT,
        consistency_flag INT,
        FOREIGN KEY (time_id) REFERENCES dim_time(time_id),
        FOREIGN KEY (location_id) REFERENCES dim_location(location_id),
        FOREIGN KEY (attack_type_id) REFERENCES dim_attacktype(attack_type_id),
        FOREIGN KEY (target_type_id) REFERENCES dim_targettype(target_type_id),
        FOREIGN KEY (group_id) REFERENCES dim_terroristgroup(group_id),
        FOREIGN KEY (property_id) REFERENCES dim_propertydamage(property_id)
    )
"""

# Populate dimension tables
populate_dimension_table(terrorist_dw_engine, 'Terrorist_dw', 'dim_time', 
                        "SELECT year, month, day, extended FROM Terrorist_db.incidents", 
                        pd.DataFrame({'year': [1970], 'month': [1], 'day': [1], 'extended': [0]}), 
                        ['year', 'month', 'day', 'extended'], dim_time_schema)

populate_dimension_table(terrorist_dw_engine, 'Terrorist_dw', 'dim_location', 
                        "SELECT region, country, provstate, city, latitude, longitude, specificity, vicinity, unknown_location AS unknown_location_flag FROM Terrorist_db.locations", 
                        pd.DataFrame({'region': ['Unknown'], 'country': ['Unknown'], 'provstate': ['Unknown'], 
                                      'city': ['Unknown'], 'latitude': [0.0], 'longitude': [0.0], 
                                      'specificity': [1], 'vicinity': [0], 'unknown_location_flag': [1]}), 
                        ['region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location_flag'], dim_location_schema)

populate_dimension_table(terrorist_dw_engine, 'Terrorist_dw', 'dim_attacktype', 
                        "SELECT attacktype1 AS attack_type, weaptype1 AS weapon_type, weapsubtype1 AS weapon_subtype FROM Terrorist_db.attacks", 
                        pd.DataFrame({'attack_type': ['Unknown'], 'weapon_type': ['Unknown'], 'weapon_subtype': ['Unknown']}), 
                        ['attack_type', 'weapon_type', 'weapon_subtype'], dim_attacktype_schema)

populate_dimension_table(terrorist_dw_engine, 'Terrorist_dw', 'dim_targettype', 
                        "SELECT targtype1 AS target_type, targsubtype1 AS target_subtype, nationality1 AS nationality FROM Terrorist_db.targets", 
                        pd.DataFrame({'target_type': ['Unknown'], 'target_subtype': ['Unknown'], 'nationality': ['Unknown']}), 
                        ['target_type', 'target_subtype', 'nationality'], dim_targettype_schema)

populate_dimension_table(terrorist_dw_engine, 'Terrorist_dw', 'dim_terroristgroup', 
                        "SELECT group_name, uncertainty1 AS uncertainty, unknown_group AS unknown_group_flag FROM Terrorist_db.terroristgroups", 
                        pd.DataFrame({'group_name': ['Unknown'], 'uncertainty': [0], 'unknown_group_flag': [1]}), 
                        ['group_name', 'uncertainty', 'unknown_group_flag'], dim_terroristgroup_schema)

populate_dimension_table(terrorist_dw_engine, 'Terrorist_dw', 'dim_propertydamage', 
                        "SELECT extent, property, propvalue_category FROM Terrorist_db.properties", 
                        pd.DataFrame({'extent': ['Unknown'], 'property': [0], 'propvalue_category': ['Unknown']}), 
                        ['extent', 'property', 'propvalue_category'], dim_propertydamage_schema)

# Create fact_incidents table
with terrorist_dw_engine.connect() as conn:
    trans = conn.begin()
    try:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        conn.execute(text(f"CREATE TABLE Terrorist_dw.fact_incidents {fact_incidents_schema}"))
        logging.info("Created table Terrorist_dw.fact_incidents")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        trans.commit()
    except Exception as e:
        trans.rollback()
        logging.error(f"Failed to create table Terrorist_dw.fact_incidents: {str(e)}")
        raise

# Load dimension tables into memory
logging.info("Loading Terrorist_dw dimension tables into memory")
dim_time = pd.read_sql("SELECT * FROM Terrorist_dw.dim_time", terrorist_dw_engine)
dim_location = pd.read_sql("SELECT * FROM Terrorist_dw.dim_location", terrorist_dw_engine)
dim_attacktype = pd.read_sql("SELECT * FROM Terrorist_dw.dim_attacktype", terrorist_dw_engine)
dim_targettype = pd.read_sql("SELECT * FROM Terrorist_dw.dim_targettype", terrorist_dw_engine)
dim_terroristgroup = pd.read_sql("SELECT * FROM Terrorist_dw.dim_terroristgroup", terrorist_dw_engine)
dim_propertydamage = pd.read_sql("SELECT * FROM Terrorist_dw.dim_propertydamage", terrorist_dw_engine)

# Clear Fact_Incidents
logging.info("Clearing Terrorist_dw.fact_incidents")
with terrorist_dw_engine.connect() as conn:
    trans = conn.begin()
    try:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        conn.execute(text("DELETE FROM Terrorist_dw.fact_incidents"))
        count = pd.read_sql("SELECT COUNT(*) as count FROM Terrorist_dw.fact_incidents", conn)['count'].iloc[0]
        logging.info(f"Deleted all rows from Terrorist_dw.fact_incidents, row count: {count}")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        trans.commit()
    except Exception as e:
        trans.rollback()
        logging.error(f"Failed to delete rows in Terrorist_dw.fact_incidents: {str(e)}")
        raise

# Process chunks and insert into Fact_Incidents
logging.info("Processing chunks for Terrorist_dw.fact_incidents")
chunks = pd.read_csv(input_path, encoding='utf-8', chunksize=chunk_size)
for i, chunk in enumerate(chunks):
    logging.info(f"Processing chunk {i+1} with {len(chunk)} rows", extra={'important': True})

    # Rename column to match schema
    chunk.rename(columns={'event_id': 'eventid'}, inplace=True)

    # Map to dimension IDs with defaults
    chunk = chunk.merge(dim_time[['time_id', 'year', 'month', 'day', 'extended']],
                        on=['year', 'month', 'day', 'extended'], how='left')
    chunk['time_id'] = chunk['time_id'].fillna(dim_time[dim_time['year'] == 1970]['time_id'].iloc[0])

    chunk = chunk.merge(dim_location[['location_id', 'region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location_flag']],
                        left_on=['region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location'],
                        right_on=['region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location_flag'],
                        how='left')
    chunk['location_id'] = chunk['location_id'].fillna(dim_location[dim_location['region'] == 'Unknown']['location_id'].iloc[0])

    chunk = chunk.merge(dim_attacktype[['attack_type_id', 'attack_type', 'weapon_type', 'weapon_subtype']],
                        left_on=['attacktype1', 'weaptype1', 'weapsubtype1'],
                        right_on=['attack_type', 'weapon_type', 'weapon_subtype'],
                        how='left')
    chunk['attack_type_id'] = chunk['attack_type_id'].fillna(dim_attacktype[dim_attacktype['attack_type'] == 'Unknown']['attack_type_id'].iloc[0])

    chunk = chunk.merge(dim_targettype[['target_type_id', 'target_type', 'target_subtype', 'nationality']],
                        left_on=['targtype1', 'targsubtype1', 'nationality1'],
                        right_on=['target_type', 'target_subtype', 'nationality'],
                        how='left')
    chunk['target_type_id'] = chunk['target_type_id'].fillna(dim_targettype[dim_targettype['target_type'] == 'Unknown']['target_type_id'].iloc[0])

    chunk = chunk.merge(dim_terroristgroup[['group_id', 'group_name', 'uncertainty', 'unknown_group_flag']],
                        left_on=['group_name', 'uncertainty1', 'unknown_group'],
                        right_on=['group_name', 'uncertainty', 'unknown_group_flag'],
                        how='left')
    chunk['group_id'] = chunk['group_id'].fillna(dim_terroristgroup[dim_terroristgroup['group_name'] == 'Unknown']['group_id'].iloc[0])

    chunk = chunk.merge(dim_propertydamage[['property_id', 'extent', 'property', 'propvalue_category']],
                        on=['extent', 'property', 'propvalue_category'],
                        how='left')
    chunk['property_id'] = chunk['property_id'].fillna(dim_propertydamage[dim_propertydamage['extent'] == 'Unknown']['property_id'].iloc[0])

    # Prepare the fact table DataFrame
    fact_df = pd.DataFrame({
        'eventid': chunk['eventid'],
        'time_id': chunk['time_id'],
        'location_id': chunk['location_id'],
        'attack_type_id': chunk['attack_type_id'],
        'target_type_id': chunk['target_type_id'],
        'group_id': chunk['group_id'],
        'property_id': chunk['property_id'],
        'nkill': chunk['nkill'].fillna(0),
        'nwound': chunk['nwound'].fillna(0),
        'nkillter': chunk['nkillter'].fillna(0),
        'nwoundter': chunk['nwoundter'].fillna(0),
        'nkillus': chunk['nkillus'].fillna(0),
        'nwoundus': chunk['nwoundus'].fillna(0),
        'propvalue': chunk['propvalue'],
        'success': chunk['success'],
        'propvalue_imputed': chunk['propvalue_imputed'],
        'us_casualty_imputed': chunk['us_casualty_imputed'],
        'high_impact': chunk['high_impact'],
        'consistency_flag': 0
    })

    # Filter only for measure and flag constraints
    fact_df = fact_df[
        (fact_df['nkill'] >= 0) &
        (fact_df['nwound'] >= 0) &
        (fact_df['nkillter'] >= 0) &
        (fact_df['nwoundter'] >= 0) &
        (fact_df['nkillus'] >= 0) &
        (fact_df['nwoundus'] >= 0) &
        fact_df['success'].isin([0, 1]) &
        fact_df['propvalue_imputed'].isin([0, 1]) &
        fact_df['us_casualty_imputed'].isin([0, 1]) &
        fact_df['high_impact'].isin([0, 1])
    ]

    # Deduplicate fact_df by eventid, keeping the first occurrence
    fact_df = fact_df.drop_duplicates(subset=['eventid'], keep='first')
    logging.info(f"After deduplication, chunk {i+1} has {len(fact_df)} rows")

    # Insert into Fact_Incidents
    try:
        fact_df.to_sql('fact_incidents', terrorist_dw_engine, schema='Terrorist_dw', if_exists='append', index=False)
        logging.info(f"Loaded {len(fact_df)} rows into Terrorist_dw.fact_incidents for chunk {i+1}")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load Terrorist_dw.fact_incidents for chunk {i+1}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading Terrorist_dw.fact_incidents for chunk {i+1}: {str(e)}")
        raise

print(f"✅ Terrorist_db and Terrorist_dw insertion completed. Check {log_file} for details.")
logging.info("Terrorist_db and Terrorist_dw insertion completed successfully", extra={'important': True})