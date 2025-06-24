#!/bin/bash
# https://github.com/prometheus-operator/kube-prometheus/tree/release-0.14 <-- used
# echo colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
GREEN_BG_BLACK_TEXT='\033[42;30m'
NC='\033[0m' # reset

KUBEPROMETHEUS_VERSION="release-0.14"

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
total_steps=7

echo -e "${GREEN_BG_BLACK_TEXT} Prometheus (kube-prometheus-stack w/o grafana & alertmanager, helm) setup v0.1 by Adrian Schubek${NC}"

log_question "Open dashboard [r] or install [i]?"
read run_or_install
if [ "$run_or_install" != "i" ]; then
    log_info "Access Prometheus at http://<any_node_ip>:9090"
    kubectl -n prom port-forward prometheus-kube-prometheus-stack-prometheus-0 9090 --address 0.0.0.0
    exit 0
fi

log_question "Is this the master node AND Kubernetes setup already completed using k8s.sh? [y/n]"
read node_type
if [ "$node_type" != "y" ]; then
    log_error "Install Kubernetes first using k8s.sh"
    exit 1
fi

log_step "Preparing config to monitor kube-proxy"
log_info "--------------- IMPORTANT ---------------"
log_info "Manually set metricsBindAddress from ${RED}''${NC} to ${RED}0.0.0.0${NC} using: ${YELLOW}kubectl edit configmap kube-proxy -n kube-system${NC}"
log_info "You can re-run this script after setting it"
log_info "--------------- IMPORTANT ---------------"

log_question "Did you set it? [y/n]"
read question1
if [ "$question1" != "y" ]; then
    log_error "Set it first"
    exit 1
fi

log_question "Enter master node IP address"
read master_ip
if [ -z "$master_ip" ]; then
    log_error "Master node IP address is required"
    exit 1
fi

log_step "Preparing config to monitor kube-controller-manager"
sudo sed -e "s/- --bind-address=127.0.0.1/- --bind-address=0.0.0.0/" -i /etc/kubernetes/manifests/kube-controller-manager.yaml

log_step "Preparing config to monitor kube-scheduler"
sudo sed -e "s/- --bind-address=127.0.0.1/- --bind-address=0.0.0.0/" -i /etc/kubernetes/manifests/kube-scheduler.yaml

# https://fabianlee.org/2022/07/08/prometheus-installing-kube-prometheus-stack-on-a-kubeadm-cluster/
log_step "Preparing config to monitor etcd"
sudo sed -i "s#--listen-metrics-urls=.*#--listen-metrics-urls=http://127.0.0.1:2381,http://$master_ip:2381#" /etc/kubernetes/manifests/etcd.yaml

# nein: kubectl patch configmap kube-proxy -n kube-system --type merge -p '{"data":{"config.conf":"metricsBindAddress: 0.0.0.0:10249"}}'# nein das breaked alles

log_step "Restarting kubelet (this may take a moment). If this timeouts re-run this command again after setup is done: ${YELLOW}kubectl -n kube-system rollout restart daemonset/kube-proxy${NC}"
kubectl -n kube-system rollout restart daemonset/kube-proxy

# sanity check
# kubeControllerManager is on port 10257
# kubeScheduler is on port 10259
# kubeProxy is on port 10249
# etcd is on port 2381

# sanity test that etcd metrics are available. no error response and content not emtpy OK
# if curl -s http://localhost:2381/metrics | grep -q "etcd_server_version"; then
#     log_step_ok "etcd metrics available"
# else
#     log_step_error "etcd metrics not available"
# fi

# # sanity test that kube proxy metrics are available OK
# if curl -s http://localhost:10249/metrics | grep -q "process_cpu_seconds_total"; then
#     log_step_ok "kube-proxy metrics available"
# else
#     log_step_error "kube-proxy metrics not available"
# fi

# # sanity test that kube controller manager metrics are available 
# if curl -s http://localhost:10257/metrics | grep -q "kube_controller_manager_object_counts"; then
#     log_step_ok "kube-controller-manager metrics available"
# else
#     log_step_error "kube-controller-manager metrics not available"
# fi

# # sanity test that kube scheduler metrics are available
# if curl -s http://localhost:10259/metrics | grep -q "kube_scheduler_scheduling_algorithm_predicate_evaluation_seconds"; then
#     log_step_ok "kube-scheduler metrics available"
# else
#     log_step_error "kube-scheduler metrics not available"
# fi


log_step "Install helm"
sudo snap install helm --classic

log_step "Add prometheus-community repo"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update prometheus-community

log_step "Create prometheus namespace"
kubectl create namespace prom

log_step "Install kube-prometheus-stack without grafana and alertmanager"
helm upgrade --install --namespace prom -f https://k8s.adriansoftware.de/prometheus/values.yaml kube-prometheus-stack prometheus-community/kube-prometheus-stack

log_step_ok "Prometheus installed"

kubectl -n prom port-forward prometheus-kube-prometheus-stack-prometheus-0 9090 --address 0.0.0.0
log_info "Access Prometheus at http://<any_node_ip>:9090"

# old. use helm instead
# log_step "Downloading kube-prometheus $KUBEPROMETHEUS_VERSION from GitHub"
# cd $HOME
# git clone -b $KUBEPROMETHEUS_VERSION --single-branch --depth 1 https://github.com/prometheus-operator/kube-prometheus
# cd kube-prometheus/

# log_step "Applying kube-prometheus setup manifests"
# kubectl apply --server-side -f manifests/setup

# log_step "Waiting for CRDs to be ready"
# kubectl wait \
# 	--for condition=Established \
# 	--all CustomResourceDefinition \
# 	--namespace=monitoring

# log_step "Applying kube-prometheus manifests"
# kubectl apply -f manifests/



# uninstall: cd $HOME/kube-prometheus && kubectl delete --ignore-not-found=true -f manifests/ -f manifests/setup

# old
# --https://prometheus-operator.dev/kube-prometheus/kube/kube-prometheus-on-kubeadm/-- OUTDATED
# log_step "Creating monitoring namespace"
# export NAMESPACE='monitoring'
# kubectl create namespace "$NAMESPACE"

# log_step "Applying prometheus-operator manifest"
# kubectl --namespace="$NAMESPACE" apply -f manifests/prometheus-operator

# log_step "Waiting for Prometheus to be ready"
# until kubectl --namespace="$NAMESPACE" get alertmanagers.monitoring.coreos.com > /dev/null 2>&1; do sleep 1; printf "."; done

# log_step "Applying node-exporter manifest"
# kubectl --namespace="$NAMESPACE" apply -f manifests/node-exporter

# log_step "Applying kube-state-metrics manifest"
# kubectl --namespace="$NAMESPACE" apply -f manifests/kube-state-metrics

# # skip grafana. see docs

# log_step "Applying prometheus manifest and roles"
# find manifests/prometheus -type f ! -name prometheus-k8s-roles.yaml ! -name prometheus-k8s-role-bindings.yaml -exec kubectl --namespace "$NAMESPACE" apply -f {} \;
# kubectl apply -f manifests/prometheus/prometheus-k8s-roles.yaml
# kubectl apply -f manifests/prometheus/prometheus-k8s-role-bindings.yaml

# log_step "Creating NodePort service"
# kubectl apply -f https://k8s.adriansoftware.de/prometheus/prometheus.yaml

