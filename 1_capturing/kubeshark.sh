exit
# old 
mkdir -p ~/.kubeshark &&
curl -s https://k8s.adriansoftware.de/kubeshark/config.yaml > ~/.kubeshark/config.yaml &&
sed -i "s/__USERNAME__/$(whoami)/g" ~/.kubeshark/config.yaml &&
mkdir -p ~/ks-scripts &&
curl -s https://k8s.adriansoftware.de/kubeshark/script.js > ~/ks-scripts/script.js &&
curl -Lo kubeshark https://github.com/kubeshark/kubeshark/releases/download/v52.3.90/kubeshark_linux_amd64 && chmod 755 kubeshark &&
sudo apt install sysdig -y
# sh <(curl -Ls https://kubeshark.co/install) &&  # this version has login wall in scripting





export TAG=v52.3.89
kubectl apply -f https://raw.githubusercontent.com/kubeshark/kubeshark/refs/$TAG/manifests/complete.yaml  
kubectl port-forward service/kubeshark-front 8899:80  

# cleanup  
kubectl delete -f https://raw.githubusercontent.com/kubeshark/kubeshark/refs/$TAG/manifests/complete.yaml  




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

step=1
total_steps=7

echo -e "${GREEN_BG_BLACK_TEXT} Kubeshark (helm) setup v0.1 by Adrian Schubek${NC}"



# kubeshark broken by new version. cannot run scripts anymore -> need license.

# TODO: DROp kubeshark completly and use tshark instead. !!

# fixed:
# TODO: host run sudo tshark -i virbr0
# TODO: host run sudo tshark -i virbr0
# TODO: host run sudo tshark -i virbr0
# TODO: host run sudo tshark -i virbr0
# TODO: host run sudo tshark -i virbr0

helm repo add kubeshark https://helm.kubeshark.co 
helm repo update kubeshark 
helm upgrade --install --version v52.3.89 kubeshark kubeshark/kubeshark -f https://k8s.adriansoftware.de/kubeshark/values.yaml
# helm upgrade --install --namespace kubeshark --version v52.3.89 --set tap.packetCapture=ebpf kubeshark kubeshark/kubeshark

# ne fix^^ dann manuell ins scripting GUI pasten -> activaten -> als snapshot speichern   - empty files..

kubectl port-forward service/kubeshark-front 8899:80 --address 0.0.0.0



kubectl -n kubeshark edit configmap kubeshark-config-map
kubectl edit configmap kubeshark-config-map