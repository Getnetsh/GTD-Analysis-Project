-- Create data warehouse database
CREATE DATABASE IF NOT EXISTS Terrorist_dw;
USE Terrorist_db;

-- Dimension: Time
CREATE TABLE Dim_Time (
    time_id INT AUTO_INCREMENT PRIMARY KEY,
    year INT,
    month INT,
    day INT,
    approxdate VARCHAR(50),
    extended INT
) ENGINE=InnoDB;

-- Dimension: Location
CREATE TABLE Dim_Location (
    location_id INT AUTO_INCREMENT PRIMARY KEY,
    region VARCHAR(100),
    country VARCHAR(100),
    provstate VARCHAR(100),
    city VARCHAR(100),
    latitude FLOAT,
    longitude FLOAT,
    specificity INT,
    vicinity INT
) ENGINE=InnoDB;

-- Dimension: Attack
CREATE TABLE Dim_Attack (
    attack_id INT AUTO_INCREMENT PRIMARY KEY,
    attack_type VARCHAR(100),
    weapon_type VARCHAR(100),
    weapon_subtype VARCHAR(100)
) ENGINE=InnoDB;

-- Dimension: Target
CREATE TABLE Dim_Target (
    target_id INT AUTO_INCREMENT PRIMARY KEY,
    target_type VARCHAR(100),
    target_subtype VARCHAR(100),
    nationality VARCHAR(100)
) ENGINE=InnoDB;

-- Dimension: Group
CREATE TABLE Dim_Group (
    group_id INT AUTO_INCREMENT PRIMARY KEY,
    group_name VARCHAR(255),
    subgroup_name VARCHAR(255),
    uncertainty INT
) ENGINE=InnoDB;

-- Dimension: Property
CREATE TABLE Dim_Property (
    property_id INT AUTO_INCREMENT PRIMARY KEY,
    extent VARCHAR(100),
    property INT
) ENGINE=InnoDB;

-- Fact: Incidents
CREATE TABLE Fact_Incidents (
    eventid BIGINT PRIMARY KEY,
    time_id INT,
    location_id INT,
    attack_id INT,
    target_id INT,
    group_id INT,
    property_id INT,
    incident_count INT DEFAULT 1,
    nkill INT,
    nwound INT,
    nkillter INT,
    nwoundter INT,
    propvalue FLOAT,
    success INT,
    FOREIGN KEY (time_id) REFERENCES Dim_Time(time_id),
    FOREIGN KEY (location_id) REFERENCES Dim_Location(location_id),
    FOREIGN KEY (attack_id) REFERENCES Dim_Attack(attack_id),
    FOREIGN KEY (target_id) REFERENCES Dim_Target(target_id),
    FOREIGN KEY (group_id) REFERENCES Dim_Group(group_id),
    FOREIGN KEY (property_id) REFERENCES Dim_Property(property_id)
) ENGINE=InnoDB;

USE Terrorist_db;
select * from Dim_Attack