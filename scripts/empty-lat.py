import pandas as pd

# Correctly load the CSV into a DataFrame
df = pd.read_csv(r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed\gtd_cleaned_enhanced_v6.9.csv')

# Find rows with missing coordinates
empty_coords = df[df['latitude'].isna() | df['longitude'].isna()]
print(f"Number of rows with empty coordinates: {len(empty_coords)}")
print(f"Sample empty coordinate rows:\n{empty_coords[['eventid', 'city', 'provstate', 'country', 'latitude', 'longitude']].head().to_string()}")

# Save to CSV (use raw string for path)
empty_coords.to_csv(r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed\missing_coords.csv', index=False)
