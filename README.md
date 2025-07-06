# GTD Analysis Project | Data Warehouse Implementation

## 📌 Overview
Academic project implementing a terrorism data warehouse (MySQL) with Python ETL pipelines and Tableau visualization capabilities. Processes the Global Terrorism Database to analyze incidents worldwide, with special focus on regions like Yunnan, China.

## 🛠️ Technical Stack
- **Data Processing**: Python (pandas, numpy)  
- **Database**: MySQL 8.0+  
- **Visualization**: Tableau (optional)  
- **Dataset**: [GTD](https://www.start.umd.edu/gtd/) (~180K incidents)

## 🚀 Quick Start
1. **Clone & Setup**:
```bash
git clone https://github.com/your-username/GTD_Analysis_Project.git
cd GTD_Analysis_Project
pip install pandas numpy mysql-connector-python

📊 Data Model (Star Schema)
fact_incidents
  ├── dim_time (time_id)
  ├── dim_location (location_id)
  ├── dim_attacktype (attack_type_id)
  └── dim_weapon (weapon_id)

  📂 File Structure
GTD_Analysis_Project/
├── data/                   # Input/Output datasets
├── scripts/
│   ├── clean_gtd_data.py   # Data cleaning (v6)
│   └── etl_gtd_data.py     # ETL pipeline (v22)
└── README.md

✅ Validation Checks
-- Expected Results:
SELECT COUNT(*) FROM fact_incidents;      -- 180,800
SELECT COUNT(*) FROM dim_location;        -- 10K-20K 
SELECT COUNT(*) FROM dim_time;            -- 40+ years
📧 Contact

For academic collaboration: Getnetss2009@gmail.com

Key features:
1. **Compact but comprehensive** - All critical info in one scrollable view
2. **Visual hierarchy** - Emojis and spacing for better scanning
3. **Ready-to-run** code blocks
4. **Core technical details** preserved
5. **Mobile-friendly** formatting
6. **Self-contained** - No external dependencies in text

Simply copy this into your README.md - it maintains all key information while being significantly more concise than the original.