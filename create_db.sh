#!/bin/bash

# This script creates a MySQL database and its tables.

# --- Database Credentials ---
# Replace 'your_username' and 'your_password' with your MySQL credentials.
# If you don't use a password for root, you can remove the -p flag.
MYSQL_USER="root"
MYSQL_PASSWORD="czk5185668"
DATABASE_NAME="breast_cosma_db"

# --- SQL Commands ---
# The SQL statements to create the database and tables.
# Note: Foreign key constraints reference tables created earlier in the script.
SQL_SCRIPT="
-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS $DATABASE_NAME;
USE $DATABASE_NAME;

-- 1. Create the 'users' table
CREATE TABLE IF NOT EXISTS users (
    user_id INT PRIMARY KEY,
    wechat_openid VARCHAR(255),
    srrsh_id INT,
    name VARCHAR(100),
    phone_number VARCHAR(20),
    registration_date DATETIME,
    last_login_date DATETIME
);

-- 2. Create the 'recovery_plans' table
CREATE TABLE IF NOT EXISTS recovery_plans (
    plan_id INT PRIMARY KEY,
    plan_name VARCHAR(100),
    description TEXT,
    start_date DATE,
    end_date DATE
);

-- 3. Create the 'exercises' table
CREATE TABLE IF NOT EXISTS exercises (
    exercise_id INT PRIMARY KEY,
    exercise_name VARCHAR(100),
    description TEXT,
    video_url VARCHAR(255),
    image_url VARCHAR(255),
    duration_minutes INT,
    repetitions INT
);

-- 4. Create the 'user_recovery_plans' table
CREATE TABLE IF NOT EXISTS user_recovery_plans (
    user_plan_id INT PRIMARY KEY,
    user_id INT,
    plan_id INT,
    assigned_date DATETIME,
    status VARCHAR(20),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (plan_id) REFERENCES recovery_plans(plan_id)
);

-- 5. Create the 'calendar_schedule' table
CREATE TABLE IF NOT EXISTS calendar_schedule (
    schedule_id INT PRIMARY KEY,
    user_id INT,
    schedule_date DATE,
    schedule_time TIME,
    type VARCHAR(50),
    event_details TEXT,
    is_completed BOOLEAN,
    completion_time DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 6. Create the 'recovery_records' table
CREATE TABLE IF NOT EXISTS recovery_records (
    record_id INT PRIMARY KEY,
    user_id INT,
    record_date DATE,
    exercise_id INT,
    duration_minutes INT,
    repetitions_completed INT,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (exercise_id) REFERENCES exercises(exercise_id)
);

-- 7. Create the 'recovery_assistant' table
CREATE TABLE IF NOT EXISTS recovery_assistant (
    assistant_id INT PRIMARY KEY,
    name VARCHAR(100),
    avatar_url VARCHAR(255),
    dialogue_flow_id VARCHAR(100)
);

-- 8. Create the 'messages_chat' table
CREATE TABLE IF NOT EXISTS messages_chat (
    message_id INT PRIMARY KEY,
    sender_id INT,
    receiver_id INT,
    message_text TEXT,
    message_image VARCHAR(255),
    timestamp DATETIME,
    FOREIGN KEY (sender_id) REFERENCES users(user_id),
    FOREIGN KEY (receiver_id) REFERENCES recovery_assistant(assistant_id)
);

-- 9. Create the 'video_slice_images' table
CREATE TABLE IF NOT EXISTS video_slice_images (
    image_id INT PRIMARY KEY,
    exercise_id INT,
    record_id INT,
    slice_order INT,
    image_path VARCHAR(255),
    timestamp DATETIME,
    FOREIGN KEY (exercise_id) REFERENCES exercises(exercise_id),
    FOREIGN KEY (record_id) REFERENCES recovery_records(record_id)
);

"

# --- Execution ---
# Check if MySQL is running and connect to it.
echo "Attempting to connect to MySQL as user '$MYSQL_USER'..."
if ! mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" -e "SELECT 1;" &> /dev/null; then
    echo "Error: Could not connect to MySQL. Please check your username and password."
    echo "Make sure the MySQL server is running."
    exit 1
fi

# Execute the SQL script
echo "Executing SQL script to create database and tables..."
if echo "$SQL_SCRIPT" | mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD"; then
    echo "Database '$DATABASE_NAME' and all tables created successfully."
else
    echo "An error occurred while executing the SQL script."
    exit 1
fi

echo "Script finished."