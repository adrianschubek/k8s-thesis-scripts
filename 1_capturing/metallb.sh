#!/bin/bash
# loadbalancer on master node only after k8s.sh setup completed
# https://metallb.io/installation/
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

echo -e "${GREEN_BG_BLACK_TEXT} MetalLB (LoadBalancer) setup v0.1 by Adrian Schubek${NC}"

log_question "Kubernetes setup already completed using k8s.sh? [y/n]"
read node_type
if [ "$node_type" != "y" ]; then
    log_error "Install Kubernetes first using k8s.sh"
    exit 1
fi

log_question "Is this the master node? [y/n]"
read node_type
if [ "$node_type" != "y" ]; then
    log_error "MetalLB is only required on the master node"
    exit 1
fi

log_step "Configure kube-proxy strictARP"
kubectl get configmap kube-proxy -n kube-system -o yaml | \
sed -e "s/strictARP: false/strictARP: true/" | \
kubectl apply -f - -n kube-system

log_step "Install MetalLB"
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.14.8/config/manifests/metallb-native.yaml

log_step "Wait for MetalLB controller to be ready"
kubectl wait --namespace metallb-system \
                --for=condition=ready pod \
                --selector=component=controller \
                --timeout=90s

log_step "Apply MetalLB configuration: addresses -> 192.168.101.0/24"
kubectl apply -f https://k8s.adriansoftware.de/metallb/metallb.yaml

log_step_ok "MetalLB installed"
kubectl get pod -n metallb-system

log_info "Host configuration required! Configure IPs on the host using metallb-host.sh"
echo -e "${YELLOW}bash <(curl -s https://k8s.adriansoftware.de/metallb-host.sh)${NC}"

# # log_info "Edit /etc/sysctl.conf and change the following line"
# # echo -e "${YELLOW}net.ipv4.ip_forward = 1${NC}"

# log_info "1. Run the following command on the HOST to enable ip forwarding"
# echo -e "${YELLOW}sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf${NC}"
# log_info "2. Apply the changes"
# echo -e "${YELLOW}sudo sysctl -p${NC}"

# # log_info "Connect routes from host to master node"
# # echo -e "${YELLOW}sudo ip route add 192.168.101.0/24 via MASTER_NODE_IP${NC}"

# log_info "3. Connect routes from host to master node via netplan"
# echo -e "${YELLOW}sudo nano /etc/netplan/01-netcfg.yaml${NC}"

# log_info "4. Add the following lines to the network configuration"
# echo -e "${YELLOW}network:
#   version: 2
#   ethernets:
#     eth0:
#       routes:
#       - to: 192.168.101.0/24
#         via: MASTER_NODE_IP${NC}"

# log_info "5. Apply the changes"
# echo -e "${YELLOW}sudo netplan apply${NC}"