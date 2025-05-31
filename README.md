GTD Analysis Project
Overview
This project is part of the Data Warehouse and Visualization course at the University of Calabria (First Year, Second Semester). It implements a data warehouse for the Global Terrorism Database (GTD), performing data cleaning, ETL (Extract, Transform, Load), and preparing data for visualization. The project uses Python for data processing, MySQL for the data warehouse, and aims to support analysis in Tableau for insights into terrorist incidents worldwide, with a focus on regions like Yunnan Province, China.
Project Objectives

Data Cleaning: Process the raw GTD dataset (globalterrorismdb_0718dist.csv) to remove duplicates, handle missing values, normalize text, and drop invalid date rows (e.g., year, month, or day = 0).
ETL Process: Load cleaned data into a MySQL data warehouse (Terrorist_dw) with a star schema, including fact (fact_incidents) and dimension tables (dim_location, dim_time, dim_attacktype, etc.).
Analysis and Visualization: Enable multidimensional analysis (e.g., incidents by region, year, attack type) and create visualizations in Tableau, focusing on specific cases like Kunming attacks.

Repository Structure
GTD_Analysis_Project/
├── data/
│   ├── raw/                    # Raw GTD dataset (not tracked)
│   └── processed/              # Cleaned CSVs and logs (not tracked)
├── scripts/
│   ├── clean_gtd_data.py       # Data cleaning script (v6, 2025-05-19)
│   └── etl_gtd_data.py         # ETL script (v22, 2025-05-20)
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation

Prerequisites

Python: 3.8+ with pandas, numpy, mysql-connector-python.
MySQL: 8.0+ with database Terrorist_dw (user: root, password: 1234).
Tableau: Public or Desktop (optional, for visualization).
GTD Dataset: Download globalterrorismdb_0718dist.csv from START GTD (not included in repo due to size and sensitivity).

Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/GTD_Analysis_Project.git
cd GTD_Analysis_Project


Install Dependencies:
pip install pandas numpy mysql-connector-python


Configure MySQL:

Install MySQL and create database:CREATE DATABASE Terrorist_dw;


Update etl_gtd_data.py with your MySQL credentials (default: root, 1234).


Place Raw Data:

Download globalterrorismdb_0718dist.csv from the GTD website.
Place it in data/raw/:D:\Calabria_Universty_Doc\First year\2nd semester\Data Warehouse and Visualization\DWHP\GTD_Analysis_Project\data\raw\globalterrorismdb_0718dist.csv





Usage

Clean the Data:

Run the cleaning script to generate gtd_cleaned_enhanced_v4.csv (~180,800 rows):cd scripts
python clean_gtd_data.py


Output: data/processed/gtd_cleaned_enhanced_v4.csv, dropped_invalid_dates.csv, cleaning_log.txt.


Run ETL Process:

Load cleaned data into Terrorist_dw:python etl_gtd_data.py


Output: Populates fact_incidents (180,800 rows), dim_location, dim_time, etc.


Validate Data:

Connect to MySQL:mysql -u root -p Terrorist_dw


Run validation queries:SELECT COUNT(*) FROM fact_incidents;  -- Expected: 180800
SELECT COUNT(*) FROM dim_location;    -- Expected: ~10000-20000
SELECT COUNT(*) FROM fact_incidents WHERE time_id IS NULL;  -- Expected: 0




Analyze and Visualize:

Query for analysis (e.g., Yunnan incidents):SELECT fi.eventid, dl.city, dl.latitude, dl.longitude, da.attack_type
FROM fact_incidents fi
JOIN dim_location dl ON fi.location_id = dl.location_id
JOIN dim_attacktype da ON fi.attack_type_id = da.attack_type_id
WHERE dl.provstate = 'yunnan province' AND dl.city = 'kunming';


Export to CSV:mysql -u root -p Terrorist_dw -e "SELECT dl.region, dt.year, COUNT(*) AS incident_count FROM fact_incidents fi JOIN dim_location dl ON fi.location_id = dl.location_id JOIN dim_time dt ON fi.time_id = dt.time_id GROUP BY dl.region, dt.year;" > data/processed/region_year_analysis.csv


Import region_year_analysis.csv into Tableau for visualizations (e.g., incidents by region/year).



Results

Cleaning: Produces gtd_cleaned_enhanced_v4.csv with 180,800 rows, dropping ~891 rows with invalid dates (year, month, or day = 0).
ETL: Loads 180,800 rows into fact_incidents, with no missing foreign keys.
Analysis: Supports queries for regional trends, attack types, and specific locations (e.g., Kunming: latitude ~25.0389, longitude ~102.7183, attack_type: armed assault).

Contributing
This is an academic project. For suggestions or improvements, contact [your-email@domain.com] or open an issue/pull request.
License
This project is for educational purposes and uses the GTD dataset under its terms. Code is licensed under MIT License.
Acknowledgments

START GTD: For providing the dataset.
University of Calabria: For the course framework.

