apt upgrade -y
apt install mysql-server -y

service mysql start
ps aux | grep mysql

# CREATE USER 'flask_user'@'localhost' IDENTIFIED BY 'password';
# GRANT ALL PRIVILEGES ON *.* TO 'flask_user'@'localhost' WITH GRANT OPTION;
# FLUSH PRIVILEGES;