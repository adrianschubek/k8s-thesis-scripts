https://github.com/E1A/CVE-2023-4596

# Setup

uses same wordpress instacne as in [08](../08/readme.md)

```bash
# find wordpress* pod grep
ID=$(kubectl get pods -o wide | grep wordpress | awk '{print $1}')
# exec into container and run
kubectl exec -it $ID -- /bin/bash
### in shell:
apt update && apt install unzip -y
curl https://downloads.wordpress.org/plugin/forminator.1.24.6.zip -o /tmp/forminator.zip \
    && unzip /tmp/forminator.zip -d /var/www/html/wp-content/plugins/ \
    && rm /tmp/forminator.zip
### out

curl https://k8s.adriansoftware.de/attacks/04/exploit.py -o exploit4.py
```

open browser got ot <master_ip>:30080
-> plugins -> Activate Forminator plugin -> follow https://github.com/E1A/CVE-2023-4596?tab=readme-ov-file#installation -> copy post url

http://192.168.122.216:30080/2025/02/18/19/

### Note: Put this URL in your global capture.env file!!

```bash
ATTACK11_URL="http://192.168.122.216:30080/2025/02/18/19/"
```
