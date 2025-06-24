#!/bin/bash
# THis script is for setting up the ips on the host machine
# echo colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
GREEN_BG_BLACK_TEXT='\033[42;30m'
NC='\033[0m' # reset

log_step() {
    echo -e "${CYAN}[ $((step++))/$total_steps ] $1${NC}"
}
log_step_ok() {
    echo -e "${GREEN}[ $((step++))/$total_steps ] $1${NC}"
}
log_step_error() {
    echo -e "${RED}[ $((step++))/$total_steps ] $1${NC}"
}
log_info() {
    echo -e "${CYAN}[ i ] $1${NC}"
}
log_error() {
    echo -e "${RED}[ ! ] $1${NC}"
}
log_question() {
    echo -e "${YELLOW}[ ? ] $1${NC}"
}

step=1
total_steps=3

echo -e "${GREEN_BG_BLACK_TEXT} Load Balancer Host setup v0.1 by Adrian Schubek${NC}"

log_question "Is this the HOST machine (outside K8s cluster)? [y/n]"
read node_type
if [ "$node_type" != "y" ]; then
    log_error "Host machine required for Load Balancer config"
    exit 1
fi

log_question "MetalLB already running in the cluster? [y/n]"
read node_type
if [ "$node_type" != "y" ]; then
    log_error "Run MetalLB setup on the master node first"
    exit 1
fi

log_question "Enter master node ip"
read MASTER_NODE_IP
if [ -z "$MASTER_NODE_IP" ]; then
    log_error "Master node IP required"
    exit 1
fi

log_question "Enter network interface [eth0]"
read INTERFACE
if [ -z "$INTERFACE" ]; then
    INTERFACE="eth0"
fi

log_step "Enable IPv4 forwarding"
sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf

log_step "Apply the changes"
sudo sysctl -p

log_step_error "Temporary workaround: Run the following commands on the host machine manually. After reboot you need to run it again. Or set in in Ubuntu > Settings > Network > IPv4 > Routes"
echo -e "${YELLOW}sudo ip route add 192.168.101.0/24 via $MASTER_NODE_IP${NC}"
exit 1
# FIXME

log_step "Setting netplan config /etc/netplan/01-netcfg.yaml"
# network:
#   version: 2
#   ethernets:
#     eth0:
#       routes:
#       - to: 192.168.101.0/24
#         via: MASTER_NODE_IP

# sudo netplan status
# FIXME: this breaks everyhting. sudo ip... funktioniert. maybe falsches interface eth0? workaroudn manuell ip config machen
sudo tee -a /etc/netplan/01-netcfg.yaml <<EOF
network:
  version: 2
  ethernets:
    $INTERFACE:
      routes:
      - to: 192.168.101.0/24
        via: $MASTER_NODE_IP
EOF

log_step "Apply netplan"
sudo netplan apply

log_step_ok "Host configuration completed"
sudo ip route show | grep "via $MASTER_NODE_IP"