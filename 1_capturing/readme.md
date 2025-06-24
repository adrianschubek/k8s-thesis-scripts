
---
# Add a new node:
1. Login on the **master** node and run:
```bash
kubeadm token create --print-join-command
```
2. Create new VM in virtual-machine-manager with Ubuntu 24.04 Server LTS image.
3. Open new terminal. Login on the new VM and run: Select worker option when prompted.
```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/k8s.sh)
```
4. Run the command from step 1 on the new VM to join it to the cluster.
5. Modify the capture scripts: Make sure you download all `*.sh` scripts to Desktop from this repo https://github.com/adrianschubek/k8s-thesis-scripts/tree/main/1_capturing/attacks
6. In `attacks.sh` and `run_attacker.sh` add new env variables for the new node:
```bash
WORKER3_IP="192.168.122.228" # set IP of VM
WORKER3_VM="k8s-worker-3" # set VM name
```
7. Edit `prepare_env.sh`, `start_capture.sh` and `end_capture.sh` and add the new env `WORKER3_IP` everywhere where the existing `WORKER1_IP` and `WORKER2_IP` variables are used. Same with `WORKER3_VM`. Example:
```diff
-for NODE in $MASTER_IP $WORKER1_IP $WORKER2_IP; do
+for NODE in $MASTER_IP $WORKER1_IP $WORKER2_IP $WORKER3_IP; do
```
---
# Cluster init
> This is already done.
## Step 0) Prerequisites
- Ubuntu 24.04 Host
- 3x Ubuntu 24.04 Server VMs (1x master, 2x worker)
## Step 1) Kubernetes installation (all nodes)
```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/k8s.sh)
```
<!-- ## ~~Step 2) Install MetalLB for LoadBalancer services (master only)~~
> skip for now. may not be needed.
```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/metallb.sh)
```
**Uninstall MetalLB:**
```bash
kubectl delete -f https://raw.githubusercontent.com/metallb/metallb/v0.14.8/config/manifests/metallb-native.yaml
```
### ~~Step 2a) For host only:~~
> skip for now. may not be needed.
```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/metallb-host.sh)
```
or (change MASTER_NODE_IP to the IP of the master node)
```bash
sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf
sudo sysctl -p
sudo ip route add 192.168.101.0/24 via MASTER_NODE_IP
```
After a reboot you may need to re-run the last command. -->
## Step 2) Install Prometheus (master only)
```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/prometheus.sh)
```

<!-- ## Step 3) Install Kubeshark config (master only)
```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/kubeshark.sh) -->
<!-- ``` -->

<!-- ## Step 4) Install Kubeshark (master only)
```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/kubeshark.sh)
```
The open dashboard, go to `Scripting` and paste the script
```ts
var xxx = [];
function onItemCaptured(metadata) {
  xxx.push(metadata);
}

function writeToFile() {
  file.mkdir("ks");
  var tempFile = file.temp("shark", "ks", "json");
  if (xxx.length > 0) {
    var yyy = JSON.stringify(xxx);
    file.write(tempFile, yyy);
    console.log("Written to " + tempFile + " length: " + xxx.length)
    xxx.length = 0;
  }
}

jobs.schedule("write-to-file", "*/10 * * * * *", writeToFile);
```

After measure move the generated scripts to here:
```bash
sudo find / -type f -name "shark*.json" -exec mv {} . \;
```

Done.

Open dashboard (optional)
```bash
kubectl -n kubeshark port-forward service/kubeshark-front 8899:80 --address 0.0.0.0
``` -->
## Optional) Install k9s CLI dashboard
```bash
curl -fsSL https://github.com/derailed/k9s/releases/download/v0.32.6/k9s_linux_amd64.deb -o k9s.deb && sudo dpkg -i k9s.deb && rm k9s.deb
```
<!-- ## ~~Optional) Run or install Dashboard (master only)~~
> skip
```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/dashboard.sh)
``` -->
<!-- curl -fsSL https://k8s.adriansoftware.de/k8s.sh -o k8s.sh && chmod +x k8s.sh && ./k8s.sh -->