#!/bin/bash
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

echo -e "${GREEN_BG_BLACK_TEXT}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}        Kubernetes Setup v0.3 by Adrian Schubek   ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}         For Ubuntu 24.04 LTS Server              ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}          https://k8s.adriansoftware.de           ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}                                                  ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}            > Kubernetes v1.31                    ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}            > containerd v1.7.22+                 ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}            > Calico CNI v3.29                    ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}══════════════════════════════════════════════════${NC}"

log_question "Select master [m] or worker [w] node setup"
read node_type
if [ "$node_type" != "m" ] && [ "$node_type" != "w" ]; then
    log_error "Invalid node type"
    exit 1
fi
if [ "$node_type" = "m" ]; then
    log_info "Selected master node setup"
else
    log_info "Selected worker node setup"
fi

step=1
total_steps=23
if [ "$node_type" = "m" ]; then
    total_steps=27
fi

log_step "Update packages"
sudo apt update
log_step "Upgrade packages"
sudo apt upgrade -y
log_step "Install curl"
sudo apt install apt-transport-https curl -y

log_step "Disable swap temporarily"
sudo swapoff -a
log_step "Disable swap permanently in config"
sudo sed -i '/swap/ s/^/#/' /etc/fstab
log_step "Disable swap service permanently"
sudo systemctl mask swap.target 

# Exit if swap is not disabled or swap entry is not commented in /etc/fstab
if [ "$(swapon --show)" ] || grep -q -v "^#" /etc/fstab | grep -q "swap"; then
    log_error "Swap is not disabled or not commented in /etc/fstab"
    exit 1
fi

log_step "Setup kernel modules"
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF
log_step "Load kernel module overlay"
sudo modprobe overlay
log_step "Load kernel module br_netfilter"
sudo modprobe br_netfilter
log_step "Setup kernel parameters"
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
log_step "Apply kernel parameters"
sudo sysctl --system

# https://docs.docker.com/engine/install/ubuntu/
log_step "Remove outdated Docker packages"
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
log_step "Setup Docker repository"
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
log_step "Update packages"
sudo apt-get update
log_step "Install containerd"
sudo apt-get install containerd.io
# sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

log_step "Setup containerd config"
sudo rm -f /etc/containerd/config.toml
sudo mkdir -p /etc/containerd
sudo sh -c "containerd config default > /etc/containerd/config.toml"
log_step "Enable SystemdCgroup in /etc/containerd/config.toml"
# sed ok
sudo sed -i 's/SystemdCgroup \= false/SystemdCgroup \= true/g' /etc/containerd/config.toml
log_step "Restart containerd"
sudo systemctl restart containerd.service

# exit if containerd is not running or cat /etc/containerd/config.toml | grep SystemdCgroup is not true
if [ "$(systemctl is-active containerd)" != "active" ] || ! grep -q "SystemdCgroup = true" /etc/containerd/config.toml; then
    log_error "containerd is not running or SystemdCgroup in config is not enabled"
    exit 1
fi

# utils
sudo apt-get install sysdig -y
sudo apt-get install python3 -y
# ssh server if not already installed
sudo apt-get install openssh-server -y

log_step "Setup Kubernetes repository"
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /" | sudo tee /etc/apt/sources.list.d/kubernetes.list
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
log_step "Update packages"
sudo apt-get update
log_step "Install Kubernetes packages"
sudo apt-get install -y kubelet kubeadm kubectl
log_step "Hold Kubernetes packages"
sudo apt-mark hold kubelet kubeadm kubectl

if [ "$node_type" = "w" ]; then
    log_step_ok "Worker node setup complete. Ready to join cluster using 'sudo kubeadm join ...'"
    # echo -e "${CYAN}[ $((step++))/$total_steps ] Join Kubernetes cluster"
    # echo "[ ? ] Enter the IP address of the master node"
    # read master_ip
    # echo "[ ? ] Enter the token"
    # read token
    # sudo kubeadm join $master_ip:6443 --token $token
else
    log_step "Initialize Kubernetes control plane"
    sudo kubeadm init
    log_step "Setup kubeconfig (non-root user)"
    mkdir -p $HOME/.kube
    sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
    sudo chown $(id -u):$(id -g) $HOME/.kube/config
    export KUBECONFIG=
    log_step "Apply Calico CNI"
    kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.0/manifests/calico.yaml

    log_step_ok "Create join token using: kubeadm token create --print-join-command"
    log_step_ok "Master node setup complete"
    kubectl get nodes
fi