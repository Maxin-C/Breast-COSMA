apt upgrade -y
apt install mysql-server -y

service mysql start
ps aux | grep mysql