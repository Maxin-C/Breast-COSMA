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
    user_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    wechat_openid VARCHAR(255),
    srrsh_id INT,
    name VARCHAR(100),
    phone_number VARCHAR(20),
    registration_date DATETIME,
    last_login_date DATETIME
);

-- 2. Create the 'recovery_plans' table
CREATE TABLE IF NOT EXISTS recovery_plans (
    plan_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    plan_name VARCHAR(100),
    description TEXT,
    start_date DATE,
    end_date DATE
);

-- 3. Create the 'exercises' table
CREATE TABLE IF NOT EXISTS exercises (
    exercise_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    exercise_name VARCHAR(100),
    description TEXT,
    video_url VARCHAR(255),
    image_url VARCHAR(255),
    duration_minutes INT,
    repetitions INT
);

-- 4. Create the 'user_recovery_plans' table
CREATE TABLE IF NOT EXISTS user_recovery_plans (
    user_plan_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    user_id INT,
    plan_id INT,
    assigned_date DATETIME,
    status VARCHAR(20),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (plan_id) REFERENCES recovery_plans(plan_id)
);

-- 5. Create the 'calendar_schedule' table
CREATE TABLE IF NOT EXISTS calendar_schedule (
    schedule_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    user_id INT,
    schedule_date DATE,
    schedule_time TIME,
    type VARCHAR(50),
    event_details TEXT,
    is_completed BOOLEAN,
    completion_time DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS recovery_records (
    record_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    user_id INT,
    record_date DATETIME, -- 使用 DATETIME 更精确地记录发生时间
    notes TEXT, -- 可以添加备注字段，记录本次记录的总体情况
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS recovery_record_details (
    record_detail_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    record_id INT, -- 外键，关联到 recovery_records 表
    exercise_id INT, -- 外键，关联到 exercises 表
    actual_duration_minutes INT, -- 实际锻炼时长
    actual_repetitions_completed INT, -- 实际完成重复次数
    brief_evaluation VARCHAR(50),
    evaluation_details TEXT,
    completion_timestamp DATETIME, -- 该次具体运动完成的时间戳
    FOREIGN KEY (record_id) REFERENCES recovery_records(record_id),
    FOREIGN KEY (exercise_id) REFERENCES exercises(exercise_id)
);

-- 8. Create the 'messages_chat' table
CREATE TABLE IF NOT EXISTS messages_chat (
    message_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    conversation_id VARCHAR(255), -- 新增字段，用于标识一个完整的对话会话
    is_follow_up BOOLEAN,
    sender_id INT, -- 发送者ID (可以是 user_id 或 assistant_id)
    sender_type ENUM('user', 'assistant', 'professional'), -- 新增字段，发送者类型
    receiver_id INT, -- 接收者ID (可以是 user_id 或 assistant_id)
    receiver_type ENUM('user', 'assistant', 'professional'), -- 新增字段，接收者类型
    message_text TEXT,
    timestamp DATETIME,
    INDEX (conversation_id), -- 为对话ID添加索引，便于查询
    INDEX (sender_id, sender_type), -- 为发送者添加索引
    INDEX (receiver_id, receiver_type) -- 为接收者添加索引
);

-- 9. Create the 'video_slice_images' table
CREATE TABLE IF NOT EXISTS video_slice_images (
    image_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    exercise_id INT,
    record_id INT,
    slice_order INT,
    image_path VARCHAR(255),
    timestamp DATETIME,
    FOREIGN KEY (exercise_id) REFERENCES exercises(exercise_id),
    FOREIGN KEY (record_id) REFERENCES recovery_records(record_id)
);

-- 10. Create the 'form' table
CREATE TABLE IF NOT EXISTS form (
    form_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    form_name VARCHAR(50),
    form_content TEXT
);

-- 11. Create the 'quality_of_life' table
CREATE TABLE IF NOT EXISTS quality_of_life (
    qol_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    form_id INT,
    user_id INT,
    score INT,
    level VARCHAR(10),
    timestamp DATETIME,
    FOREIGN KEY (form_id) REFERENCES form(form_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
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