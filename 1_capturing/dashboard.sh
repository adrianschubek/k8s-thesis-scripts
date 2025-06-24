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

echo -e "${GREEN_BG_BLACK_TEXT}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}        Kubernetes Dashboard setup v0.1 by Adrian Schubek   ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}                 https://k8s.adriansoftware.de              ${NC}"
echo -e "${GREEN_BG_BLACK_TEXT}════════════════════════════════════════════════════════════${NC}"

# exit if kubectl is not installed
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl could not be found. Install Kubernetes cluster first"
    exit 1
fi

log_question "Run [r] or install [i] Kubernetes dashboard"
read type
if [ "$type" != "r" ] && [ "$type" != "i" ]; then
    log_error "Invalid type"
    exit 1
fi
if [ "$type" = "i" ]; then
    log_info "Installing kubernetes dashboard"
else
    log_info "Starting Kubernetes dashboard"
fi

step=1
total_steps=1
if [ "$type" = "i" ]; then
  total_steps=7
fi

if [ "$type" = "i" ]; then
  log_step "Install helm"
  sudo snap install helm --classic

  log_step "Add kubernetes dashboard helm repo"
  helm repo add kubernetes-dashboard https://kubernetes.github.io/dashboard/

  log_step "Install kubernetes dashboard"
  helm upgrade --install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard --create-namespace --namespace kubernetes-dashboard

  log_step "Create service account"
  kubectl apply -f https://k8s.adriansoftware.de/dashboard/serviceaccount.yaml

  log_step "Create cluster role binding"
  kubectl apply -f https://k8s.adriansoftware.de/dashboard/clusterrolebinding.yaml

  log_step "Create secret"
  kubectl apply -f https://k8s.adriansoftware.de/dashboard/secret.yaml
fi

log_info "Login token"
kubectl get secret admin-user -n kubernetes-dashboard -o jsonpath={".data.token"} | base64 -d
printf "\n"

log_info "Access dashboard at https://CONTROL_PLANE_IP:8443"

# https://stackoverflow.com/a/61836405
log_step_ok "Forwarding port for Kubernetes dashboard"
kubectl port-forward -n kubernetes-dashboard service/kubernetes-dashboard-kong-proxy 8443:443 --address 0.0.0.0


# firefox host url https://stackoverflow.com/a/78678499
# log_info "Open Kubernetes URL: http://CONTROL_PLANE_IP:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard-kong-proxy:443/proxy/#/login"
# kubectl proxy --address='0.0.0.0' --port=8001 --accept-hosts='.*'