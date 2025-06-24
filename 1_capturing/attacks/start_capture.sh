#!/bin/bash

start_monitoring() {
  # save running pods to get their calico IPs
  # either host==node ip 192.. or calico IPs 172...
  echo_b "Dumping pod and cluster information..."
  sshpass -p "$SSH_PASSWD" ssh -q $SSH_USER@$MASTER_IP "kubectl get po -A -o custom-columns=NAME:.metadata.name,NAMESPACE:.metadata.namespace,IP:.status.podIP,NODE:.spec.nodeName | tr -s ' ' ',' >/home/$SSH_USER/pods.csv"
  # dump all nodes and their ips
  sshpass -p "$SSH_PASSWD" ssh -q $SSH_USER@$MASTER_IP \
    "kubectl get nodes -o wide | awk 'NR==1 {print \"NAME,STATUS,ROLES,AGE,VERSION,INTERNAL-IP,EXTERNAL-IP,OS-IMAGE,KERNEL-VERSION\"} NR>1 {print \$1,\",\",\$2,\",\",\$3,\",\",\$4,\",\",\$5,\",\",\$6,\",\",\$7,\",\",\$8,\",\",\$9}' >/home/$SSH_USER/nodes.csv"

  echo_b "Starting tcpdump and sysdig on all nodes..."
  for NODE in $MASTER_IP $WORKER1_IP $WORKER2_IP; do
    sshpass -p "$SSH_PASSWD" ssh -q $SSH_USER@$NODE "echo $SSH_PASSWD | sudo -S nohup tcpdump -n -i any -w /home/$SSH_USER/any.pcap >/dev/null 2>&1 &"
    sshpass -p "$SSH_PASSWD" ssh -q $SSH_USER@$NODE "echo $SSH_PASSWD | sudo -S nohup sysdig > /home/$SSH_USER/sysdig_output.txt 2>&1 &"
    # dump running processes (for PIDs)
    sshpass -p "$SSH_PASSWD" ssh $SSH_USER@$NODE "ps aux > /home/$SSH_USER/ps.txt"
  done
}

run_attack() {
  echo_b "Waiting 12 seconds for monitoring to start..."
  sleep 12 # wait for monitoring to start
  echo_g "  CAPTURING DATA..."
  START_TIME=$(date -u +"%H:%M:%S")
  START_TIME_MS=$(date -u +"%s%3N")
  echo_g "> Attack started at: $START_TIME"
}

start_monitoring
run_attack
