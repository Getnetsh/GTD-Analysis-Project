import pandas as pd

# Load fixed dataset
df = pd.read_csv(r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed\gtd_cleaned_enhanced_v7_fixed.csv')

# Define location columns
loc_cols = ['region', 'country', 'provstate', 'city', 'latitude', 'longitude']

# Create dim_location
dim_location = df[loc_cols].drop_duplicates().reset_index(drop=True)
dim_location['location_id'] = dim_location.index + 1

# Simulate fact_incident
fact_incident = df[['eventid'] + loc_cols].merge(dim_location, on=loc_cols, how='left')

# Check for missing location_id
missing_loc_id = fact_incident['location_id'].isna().sum()
print(f'Incidents with missing location_id: {missing_loc_id}')
if missing_loc_id > 0:
    print('Sample missing:\n', fact_incident[fact_incident['location_id'].isna()][['eventid'] + loc_cols].head().to_string())

# Stats
print(f'Fact_incident rows: {len(fact_incident)}')
print(f'Unique locations in dim_location: {len(dim_location)}')
print(f'Location_id range: {dim_location["location_id"].min()} to {dim_location["location_id"].max()}')