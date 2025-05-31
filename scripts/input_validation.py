import pandas as pd

# Use raw string for the path
file_path = r'D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\processed\gtd_cleaned_enhanced_v4.csv'

# Read CSV
df = pd.read_csv(file_path)

# Validation
print('Rows:', len(df))
print('Rows with year=0:', (df['year'] == 0).sum())
print('Rows with month=0:', (df['month'] == 0).sum())
print('Rows with day=0:', (df['day'] == 0).sum())
print('Rows with year<1970:', (df['year'] < 1970).sum())
print('Rows with month<1:', (df['month'] < 1).sum())
print('Rows with day<1:', (df['day'] < 1).sum())
print('Unique year-month-day:', len(df[['year', 'month', 'day']].drop_duplicates()))
print('Unique year-month-day-extended:', len(df[['year', 'month', 'day', 'extended']].drop_duplicates()))