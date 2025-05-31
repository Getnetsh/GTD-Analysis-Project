import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

# Script version
SCRIPT_VERSION = "2025-05-29-v3"

# Setup logging
log_dir = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'etl_terrorist_db_log.txt')
backup_log_file = os.path.join(log_dir, f'etl_terrorist_db_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

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

logging.info(f"Starting Terrorist_db insertion with script version: {SCRIPT_VERSION}", extra={'important': True})
logging.info(f"Script path: {os.path.abspath(__file__)}")
print(f"Running Terrorist_db insertion version {SCRIPT_VERSION}")

# Database connection
db_user = 'root'
db_password = '1234'
db_host = 'localhost'
terrorist_db_engine = create_engine(f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/Terrorist_db')

# Input path
input_path = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed\gtd_cleaned_enhanced_v5.csv'
if not os.path.exists(input_path):
    logging.error(f"Input file not found: {input_path}")
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Load CSV in chunks
chunk_size = 20000
chunks = pd.read_csv(input_path, encoding='utf-8', chunksize=chunk_size)

# Define table columns based on CSV columns
table_columns = {
    'incidents': ['eventid', 'year', 'month', 'day', 'extended', 'success', 'summary'],
    'locations': ['eventid', 'region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location'],
    'attacks': ['eventid', 'attacktype1', 'weaptype1', 'weapsubtype1'],
    'targets': ['eventid', 'targtype1', 'targsubtype1', 'nationality1'],
    'casualties': ['eventid', 'nkill', 'nwound', 'nkillter', 'nwoundter', 'nkillus', 'nwoundus'],
    'terroristgroups': ['eventid', 'group_name', 'uncertainty1', 'unknown_group'],
    'properties': ['eventid', 'propvalue', 'extent', 'property', 'propvalue_category']
}

# Clear existing data (with foreign key checks disabled)
with terrorist_db_engine.connect() as conn:
    trans = conn.begin()
    try:
        # Disable foreign key checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        logging.info("Disabled foreign key checks")

        # Delete rows from all tables
        for table in table_columns.keys():
            conn.execute(text(f"DELETE FROM {table}"))
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn)['count'].iloc[0]
            logging.info(f"Deleted all rows from Terrorist_db.{table}, row count: {count}")

        # Re-enable foreign key checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        logging.info("Re-enabled foreign key checks")

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

    # Insert into each table
    for table, cols in table_columns.items():
        try:
            df_table = chunk[cols].copy()
            # Apply basic filtering
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

            # Handle numeric columns with NaN
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

print(f"✅ Terrorist_db insertion completed. Check {log_file} for details.")
logging.info("Terrorist_db insertion completed successfully", extra={'important': True})