-- GTD Attribute Tree for Data Warehouse
-- Represents fact, measures, dimensions, and hierarchies
-- To be visualized in MySQL Workbench EER Diagram

CREATE DATABASE IF NOT EXISTS attribute_tree;
USE attribute_tree;

-- Fact table: Terrorism Incidents
CREATE TABLE Fact_Incidents (
    eventid BIGINT PRIMARY KEY COMMENT 'Fact: Terrorism Incidents',
    incident_count INT COMMENT 'Measure',
    nkill INT COMMENT 'Measure',
    nwound INT COMMENT 'Measure',
    nkillter INT COMMENT 'Measure',
    nwoundter INT COMMENT 'Measure',
    propvalue FLOAT COMMENT 'Measure',
    success INT COMMENT 'Measure',
    time_id INT,
    location_id INT,
    attack_id INT,
    target_id INT,
    group_id INT,
    property_id INT,
    FOREIGN KEY (time_id) REFERENCES Dim_Time(time_id),
    FOREIGN KEY (location_id) REFERENCES Dim_Location(location_id),
    FOREIGN KEY (attack_id) REFERENCES Dim_Attack(attack_id),
    FOREIGN KEY (target_id) REFERENCES Dim_Target(target_id),
    FOREIGN KEY (group_id) REFERENCES Dim_Group(group_id),
    FOREIGN KEY (property_id) REFERENCES Dim_Property(property_id)
) COMMENT 'Central fact linking to dimensions';

-- Dimension: Time
CREATE TABLE Dim_Time (
    time_id INT PRIMARY KEY,
    year INT COMMENT 'Hierarchy: year, month, day',
    month INT,
    day INT,
    approxdate VARCHAR(50) COMMENT 'Non-hierarchy',
    extended INT COMMENT 'Non-hierarchy'
) COMMENT 'Time dimension';

-- Dimension: Location
CREATE TABLE Dim_Location (
    location_id INT PRIMARY KEY,
    region VARCHAR(100) COMMENT 'Hierarchy: region, country, provstate, city',
    country VARCHAR(100),
    provstate VARCHAR(100),
    city VARCHAR(100),
    latitude FLOAT COMMENT 'Non-hierarchy',
    longitude FLOAT COMMENT 'Non-hierarchy',
    specificity INT COMMENT 'Non-hierarchy',
    vicinity INT COMMENT 'Non-hierarchy'
) COMMENT 'Location dimension';

-- Dimension: Attack Type
CREATE TABLE Dim_Attack (
    attack_id INT PRIMARY KEY,
    attack_type VARCHAR(100) COMMENT 'Hierarchy: attack_type, weapon_type, weapon_subtype',
    weapon_type VARCHAR(100),
    weapon_subtype VARCHAR(100)
) COMMENT 'Attack Type dimension';

-- Dimension: Target Type
CREATE TABLE Dim_Target (
    target_id INT PRIMARY KEY,
    target_type VARCHAR(100) COMMENT 'Hierarchy: target_type, target_subtype, nationality',
    target_subtype VARCHAR(100),
    nationality VARCHAR(100)
) COMMENT 'Target Type dimension';

-- Dimension: Terrorist Group
CREATE TABLE Dim_Group (
    group_id INT PRIMARY KEY,
    group_name VARCHAR(255) COMMENT 'Flat',
    subgroup_name VARCHAR(255) COMMENT 'Flat',
    uncertainty INT COMMENT 'Flat'
) COMMENT 'Terrorist Group dimension';

-- Dimension: Property Damage
CREATE TABLE Dim_Property (
    property_id INT PRIMARY KEY,
    extent VARCHAR(100) COMMENT 'Flat',
    property INT COMMENT 'Flat'
) COMMENT 'Property Damage dimension';