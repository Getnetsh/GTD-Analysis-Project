CREATE DATABASE IF NOT EXISTS Terrorist_dw;
USE Terrorist_dw;

-- Dimension: Time
CREATE TABLE Dim_Time (
    time_id INT PRIMARY KEY AUTO_INCREMENT,
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
    extended INT NOT NULL,
    CONSTRAINT chk_year CHECK (year BETWEEN 1970 AND 2017),
    CONSTRAINT chk_month CHECK (month BETWEEN 1 AND 12),
    CONSTRAINT chk_day CHECK (day BETWEEN 1 AND 31),
    CONSTRAINT chk_extended CHECK (extended IN (0, 1))
);

-- Dimension: Location
CREATE TABLE Dim_Location (
    location_id INT PRIMARY KEY AUTO_INCREMENT,
    region VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL,
    provstate VARCHAR(100),
    city VARCHAR(100),
    latitude FLOAT,
    longitude FLOAT,
    specificity INT,
    vicinity INT,
    unknown_location_flag INT NOT NULL DEFAULT 0,
    CONSTRAINT chk_latitude CHECK (latitude BETWEEN -90 AND 90),
    CONSTRAINT chk_longitude CHECK (longitude BETWEEN -180 AND 180),
    CONSTRAINT chk_specificity CHECK (specificity BETWEEN 1 AND 5),
    CONSTRAINT chk_vicinity CHECK (vicinity IN (0, 1)),
    CONSTRAINT chk_unknown_location CHECK (unknown_location_flag IN (0, 1))
);

-- Dimension: Attack Type
CREATE TABLE Dim_AttackType (
    attack_type_id INT PRIMARY KEY AUTO_INCREMENT,
    attack_type VARCHAR(100) NOT NULL,
    weapon_type VARCHAR(100),
    weapon_subtype VARCHAR(100)
);

-- Dimension: Target Type
CREATE TABLE Dim_TargetType (
    target_type_id INT PRIMARY KEY AUTO_INCREMENT,
    target_type VARCHAR(100) NOT NULL,
    target_subtype VARCHAR(100),
    nationality VARCHAR(100)
);

-- Dimension: Terrorist Group
CREATE TABLE Dim_TerroristGroup (
    group_id INT PRIMARY KEY AUTO_INCREMENT,
    group_name VARCHAR(200) NOT NULL,
    uncertainty INT,
    unknown_group_flag INT NOT NULL DEFAULT 0,
    CONSTRAINT chk_uncertainty CHECK (uncertainty IN (0, 1)),
    CONSTRAINT chk_unknown_group CHECK (unknown_group_flag IN (0, 1))
);

-- Dimension: Property Damage
CREATE TABLE Dim_PropertyDamage (
    property_id INT PRIMARY KEY AUTO_INCREMENT,
    extent VARCHAR(100),
    property INT NOT NULL,
    propvalue_category VARCHAR(50),
    CONSTRAINT chk_property CHECK (property BETWEEN -1 AND 1)
);

-- Fact: Incidents
CREATE TABLE Fact_Incidents (
    fact_id INT PRIMARY KEY AUTO_INCREMENT,
    eventid INT,
    time_id INT,
    location_id INT,
    attack_type_id INT,
    target_type_id INT,
    group_id INT,
    property_id INT,
    nkill FLOAT NOT NULL DEFAULT 0,
    nwound FLOAT NOT NULL DEFAULT 0,
    nkillter FLOAT NOT NULL DEFAULT 0,
    nwoundter FLOAT NOT NULL DEFAULT 0,
    nkillus FLOAT NOT NULL DEFAULT 0,
    nwoundus FLOAT NOT NULL DEFAULT 0,
    propvalue FLOAT,
    success INT NOT NULL DEFAULT 0,
    propvalue_imputed INT NOT NULL DEFAULT 0,
    us_casualty_imputed INT NOT NULL DEFAULT 0,
    high_impact INT NOT NULL DEFAULT 0,
    consistency_flag INT NOT NULL DEFAULT 0,
    total_casualties FLOAT AS (nkill + nwound) STORED,
    CONSTRAINT fk_time FOREIGN KEY (time_id) REFERENCES Dim_Time(time_id),
    CONSTRAINT fk_location FOREIGN KEY (location_id) REFERENCES Dim_Location(location_id),
    CONSTRAINT fk_attack_type FOREIGN KEY (attack_type_id) REFERENCES Dim_AttackType(attack_type_id),
    CONSTRAINT fk_target_type FOREIGN KEY (target_type_id) REFERENCES Dim_TargetType(target_type_id),
    CONSTRAINT fk_group FOREIGN KEY (group_id) REFERENCES Dim_TerroristGroup(group_id),
    CONSTRAINT fk_property FOREIGN KEY (property_id) REFERENCES Dim_PropertyDamage(property_id),
    CONSTRAINT chk_nkill CHECK (nkill >= 0),
    CONSTRAINT chk_nwound CHECK (nwound >= 0),
    CONSTRAINT chk_nkillter CHECK (nkillter >= 0),
    CONSTRAINT chk_nwoundter CHECK (nwoundter >= 0),
    CONSTRAINT chk_nkillus CHECK (nkillus >= 0),
    CONSTRAINT chk_nwoundus CHECK (nwoundus >= 0),
    CONSTRAINT chk_success CHECK (success IN (0, 1)),
    CONSTRAINT chk_propvalue_imputed CHECK (propvalue_imputed IN (0, 1)),
    CONSTRAINT chk_us_casualty_imputed CHECK (us_casualty_imputed IN (0, 1)),
    CONSTRAINT chk_high_impact CHECK (high_impact IN (0, 1)),
    CONSTRAINT chk_consistency CHECK (consistency_flag IN (0, 1)),
    CONSTRAINT chk_total_casualties CHECK (total_casualties >= 0)
);

SELECT COUNT(*) AS duplicate_eventid FROM Terrorist_dw.fact_incidents GROUP BY eventid HAVING COUNT(*) > 1;
select count(*) from Dim_Time
select * from Dim_AttackType where attack_type like "unknown"
select * from Dim_TargetType where target_type like "unknown"

USE Terrorist_dw;
SHOW TABLES FROM Terrorist_dw;
select count(*) from  Fact_Incidents
select * from  Fact_Incidents
select *from  dim_time where time_id= "61215"
SELECT COUNT(*) FROM Terrorist_dw.dim_time;           -- Expected: 20,404
SELECT COUNT(*) FROM Terrorist_dw.dim_location;       -- Expected: 59,498
SELECT COUNT(*) FROM Terrorist_dw.dim_attacktype;     -- Expected: 238
SELECT COUNT(*) FROM Terrorist_dw.dim_targettype;     -- Expected: 8,072
SELECT COUNT(*) FROM Terrorist_dw.dim_terroristgroup; -- Expected: 4,261
SELECT COUNT(*) FROM Terrorist_dw.dim_propertydamage; -- Expected: 6
SELECT COUNT(*) FROM Terrorist_dw.fact_incidents;     -- Expected: 43,426 (based on log sum)

-- dim_attacktype count
SELECT attack_type, weapon_type, weapon_subtype, COUNT(*) AS duplicate_count
FROM Terrorist_dw.dim_attacktype
GROUP BY attack_type, weapon_type, weapon_subtype
HAVING COUNT(*) > 1;
-- dim_targettype count
SELECT target_type, target_subtype, nationality, COUNT(*) AS duplicate_count
FROM Terrorist_dw.dim_targettype
GROUP BY target_type, target_subtype, nationality
HAVING COUNT(*) > 1;
-- dim_terroristgroup count
SELECT group_name, uncertainty, unknown_group_flag, COUNT(*) AS duplicate_count
FROM Terrorist_dw.dim_terroristgroup
GROUP BY group_name, uncertainty, unknown_group_flag
HAVING COUNT(*) > 1;
 
    
    SET SQL_SAFE_UPDATES = 0;
    SET SQL_SAFE_UPDATES = 1;
    
    
    
    -- dim_attacktype duplcation duplction
DELETE t1 FROM Terrorist_dw.dim_attacktype t1
INNER JOIN Terrorist_dw.dim_attacktype t2
WHERE 
    t1.attack_type_id > t2.attack_type_id AND 
    t1.attack_type = t2.attack_type AND 
    t1.weapon_type = t2.weapon_type AND 
    t1.weapon_subtype = t2.weapon_subtype;

-- dim_targettype duplaction deletion
DELETE t1 FROM Terrorist_dw.dim_targettype t1
INNER JOIN Terrorist_dw.dim_targettype t2
WHERE 
    t1.target_type_id > t2.target_type_id AND 
    t1.target_type = t2.target_type AND 
    t1.target_subtype = t2.target_subtype AND 
    t1.nationality = t2.nationality;

-- dim_terroristgroup duplaction delation
DELETE t1 FROM Terrorist_dw.dim_terroristgroup t1
INNER JOIN Terrorist_dw.dim_terroristgroup t2
WHERE 
    t1.group_id > t2.group_id AND 
    t1.group_name = t2.group_name AND 
    t1.uncertainty = t2.uncertainty AND 
    t1.unknown_group_flag = t2.unknown_group_flag;