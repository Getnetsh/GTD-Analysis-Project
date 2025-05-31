-- Creating updated reconciled database for Terrorist_db
CREATE DATABASE IF NOT EXISTS Terrorist_db;
USE Terrorist_db;

-- incidents table (core table with event metadata)
CREATE TABLE incidents (
    eventid BIGINT PRIMARY KEY,
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
    extended INT NOT NULL DEFAULT 0,
    success INT NOT NULL DEFAULT 0,
    summary TEXT,
    propvalue_imputed INT NOT NULL DEFAULT 0,
    us_casualty_imputed INT NOT NULL DEFAULT 0,
    high_impact INT NOT NULL DEFAULT 0,
    INDEX idx_year (year),
    INDEX idx_date (year, month, day)
);

-- locations table (geographic details)
CREATE TABLE locations (
    location_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    eventid BIGINT NOT NULL,
    region VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL,
    provstate VARCHAR(100),
    city VARCHAR(100),
    latitude DOUBLE,
    longitude DOUBLE,
    specificity INT,
    vicinity INT,
    unknown_location INT NOT NULL DEFAULT 0,
    FOREIGN KEY (eventid) REFERENCES incidents(eventid),
    INDEX idx_region_country (region, country)
);

-- attacks table (attack characteristics)
CREATE TABLE attacks (
    attack_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    eventid BIGINT NOT NULL,
    attacktype1 VARCHAR(100) NOT NULL,
    weaptype1 VARCHAR(100),
    weapsubtype1 VARCHAR(100),
    FOREIGN KEY (eventid) REFERENCES incidents(eventid),
    INDEX idx_attacktype (attacktype1)
);

-- targets table (target details)
CREATE TABLE targets (
    target_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    eventid BIGINT NOT NULL,
    targtype1 VARCHAR(100) NOT NULL,
    targsubtype1 VARCHAR(100),
    nationality1 VARCHAR(100),
    FOREIGN KEY (eventid) REFERENCES incidents(eventid),
    INDEX idx_targtype (targtype1)
);

-- casualties table (casualty metrics)
CREATE TABLE casualties (
    casualty_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    eventid BIGINT NOT NULL,
    nkill INT,
    nwound INT,
    nkillter INT,
    nwoundter INT,
    nkillus FLOAT,
    nwoundus FLOAT,
    FOREIGN KEY (eventid) REFERENCES incidents(eventid)
);

-- terroristgroups table (group details)
CREATE TABLE terroristgroups (
    group_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    eventid BIGINT NOT NULL,
    group_name VARCHAR(255) NOT NULL,
    uncertainty1 INT,
    unknown_group INT NOT NULL DEFAULT 0,
    FOREIGN KEY (eventid) REFERENCES incidents(eventid),
    INDEX idx_group_name (group_name)
);

-- properties table (property damage details)
CREATE TABLE properties (
    property_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    eventid BIGINT NOT NULL,
    propvalue DECIMAL(15,2),
    extent VARCHAR(100),
    property INT,
    propvalue_category VARCHAR(50),
    FOREIGN KEY (eventid) REFERENCES incidents(eventid)
);