# install grafana
# https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/
apt-get install -y apt-transport-https software-properties-common wget
mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
# Updates the list of available packages
apt-get update
# Installs the latest OSS release:
apt-get install grafana
systemctl enable grafana-server
systemctl start grafana-server