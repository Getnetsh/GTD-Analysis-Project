USE Terrorist_dw;

SHOW VARIABLES LIKE 'secure_file_priv';
(
  SELECT 'time_id', 'year', 'month', 'day', 'extended'
)
UNION ALL
(
  SELECT time_id, year, month, day, extended FROM Dim_Time
)
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_time.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

(
  SELECT 'location_id', 'region', 'country', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'unknown_location_flag'
)
UNION ALL
(
  SELECT location_id, region, country, provstate, city, latitude, longitude, specificity, vicinity, unknown_location_flag FROM Dim_Location
)
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_location.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

(
  SELECT 'attack_type_id', 'attack_type', 'weapon_type', 'weapon_subtype'
)
UNION ALL
(
  SELECT attack_type_id, attack_type, weapon_type, weapon_subtype FROM Dim_AttackType
)
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_attacktype.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

(
  SELECT 'target_type_id', 'target_type', 'target_subtype', 'nationality'
)
UNION ALL
(
  SELECT target_type_id, target_type, target_subtype, nationality FROM Dim_TargetType
)
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_targettype.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

(
  SELECT 'group_id', 'group_name', 'uncertainty', 'unknown_group_flag'
)
UNION ALL
(
  SELECT group_id, group_name, uncertainty, unknown_group_flag FROM Dim_TerroristGroup
)
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_terroristgroup.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

(
  SELECT 'property_id', 'extent', 'property', 'propvalue_category'
)
UNION ALL
(
  SELECT property_id, extent, property, propvalue_category FROM Dim_PropertyDamage
)
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_propertydamage.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

(
  SELECT
    'eventid', 'time_id', 'location_id', 'attack_type_id', 'target_type_id',
    'group_id', 'property_id', 'nkill', 'nwound', 'nkillter', 'nwoundter',
    'nkillus', 'nwoundus', 'propvalue', 'success', 'propvalue_imputed',
    'us_casualty_imputed', 'high_impact', 'consistency_flag'
)
UNION ALL
(
  SELECT
    eventid, time_id, location_id, attack_type_id, target_type_id,
    group_id, property_id, nkill, nwound, nkillter, nwoundter,
    nkillus, nwoundus, propvalue, success, propvalue_imputed,
    us_casualty_imputed, high_impact, consistency_flag
  FROM Terrorist_dw.Fact_Incidents
)
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/fact_incidents.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
