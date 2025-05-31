-- Creating star schema for Terrorist_dw
CREATE DATABASE IF NOT EXISTS Terrorist_dw;
USE Terrorist_dw;

-- Dimension: Time
CREATE TABLE Dim_Time (
    time_id INT PRIMARY KEY AUTO_INCREMENT,
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
    approxdate VARCHAR(50),
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
    subgroup_name VARCHAR(200),
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
    CONSTRAINT chk_property CHECK (property BETWEEN -1 AND 1)
);

-- Fact: Incidents
CREATE TABLE Fact_Incidents (
    fact_id INT PRIMARY KEY AUTO_INCREMENT,
    eventid BIGINT NOT NULL,
    time_id INT,
    location_id INT,
    attack_type_id INT,
    target_type_id INT,
    group_id INT,
    property_id INT,
    nkill INT NOT NULL DEFAULT 0,
    nwound INT NOT NULL DEFAULT 0,
    nkillter INT NOT NULL DEFAULT 0,
    nwoundter INT NOT NULL DEFAULT 0,
    nkillus INT NOT NULL DEFAULT 0,
    nwoundus INT NOT NULL DEFAULT 0,
    propvalue FLOAT,
    success INT NOT NULL DEFAULT 0,
    consistency_flag INT NOT NULL DEFAULT 0,
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
    CONSTRAINT chk_consistency CHECK (consistency_flag IN (0, 1))
);

-- Notes:
-- - summary, motive stored in Terrorist_db.Incidents, linked via eventid.
-- - unknown_location_flag, unknown_group_flag included for analysis flexibility.
-- - Indexes can be added post-ETL for performance (e.g., INDEX(eventid)).