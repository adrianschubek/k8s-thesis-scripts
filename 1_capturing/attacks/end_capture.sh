#!/bin/bash
stop_monitoring() {
  END_TIME=$(date -u +"%H:%M:%S")
  END_TIME_MS=$(date -u +"%s%3N")
  # Calculate duration in seconds
  DURATION_SECONDS=$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s)))
  # Convert duration to minutes (rounding up to ensure at least 1 minute)
  DURATION_MINUTES=$(((DURATION_SECONDS + 59) / 60))
  echo_b "Attack duration: $(date -ud "@$DURATION_SECONDS" +%H:%M:%S)"
  sleep 1
  mkdir -p $HOST_DIR/$FOLDER
  echo "start $START_TIME" >$HOST_DIR/$FOLDER/info.txt
  echo "start_ms $START_TIME_MS" >>$HOST_DIR/$FOLDER/info.txt
  echo "end $END_TIME" >>$HOST_DIR/$FOLDER/info.txt
  echo "end_ms $END_TIME_MS" >>$HOST_DIR/$FOLDER/info.txt
  echo "duration_min $DURATION_MINUTES" >>$HOST_DIR/$FOLDER/info.txt
  echo "duration_sec $DURATION_SECONDS" >>$HOST_DIR/$FOLDER/info.txt
  echo "attack $ATTACK" >>$HOST_DIR/$FOLDER/info.txt

  echo_b "Stopping tcpdump and sysdig on all nodes..."
  for NODE in $MASTER_IP $WORKER1_IP $WORKER2_IP; do
    sshpass -p "$SSH_PASSWD" ssh $SSH_USER@$NODE "echo $SSH_PASSWD | sudo -S pkill -SIGINT sysdig"
    sshpass -p "$SSH_PASSWD" ssh $SSH_USER@$NODE "echo $SSH_PASSWD | sudo -S pkill -SIGINT tcpdump"
  done
}

collect_pod_logs() {
  echo_b "Collecting pod logs..."
  sshpass -p "$SSH_PASSWD" ssh $SSH_USER@$MASTER_IP "mkdir -p /home/$SSH_USER/podlogs && cd /home/$SSH_USER/podlogs && for ns in \$(kubectl get namespaces -o jsonpath="{.items[*].metadata.name}"); do for pod in \$(kubectl get pods -n \$ns -o jsonpath="{.items[*].metadata.name}"); do kubectl logs -n \$ns --timestamps --all-containers=true --since=$(($DURATION_MINUTES + 2))m \$pod > "\${ns}_\${pod}_logs.txt"; done; done"
}

export_prometheus_data() {
  echo_b "Starting Prometheus proxy on the master node..."

  # Start port-forward on the master node in the background
  sshpass -p "$SSH_PASSWD" ssh -f $SSH_USER@$MASTER_IP \
    "nohup kubectl -n prom port-forward prometheus-kube-prometheus-stack-prometheus-0 9090 >/dev/null 2>&1 & echo \$! > /tmp/pf_prometheus.pid"

  # Wait until Prometheus is ready on the master node
  echo_b "Waiting for Prometheus to be ready on the master node..."
  until sshpass -p "$SSH_PASSWD" ssh $SSH_USER@$MASTER_IP "curl -s http://localhost:9090/-/ready > /dev/null"; do
    sleep 1
  done

  echo_b "Prometheus is ready. Exporting data..."
  sshpass -p "$SSH_PASSWD" ssh $SSH_USER@$MASTER_IP \
    "mkdir -p /home/$SSH_USER/prom && cd /home/$SSH_USER/prom && curl -s https://k8s.adriansoftware.de/prometheus/export_csv.py | python3 - --since $(($DURATION_MINUTES + 5)) --url http://localhost:9090"

  echo_b "Stopping Prometheus proxy on the master node..."
  # Stop the port-forward process
  sshpass -p "$SSH_PASSWD" ssh $SSH_USER@$MASTER_IP "echo $SSH_PASSWD | sudo -S kill \$(cat /tmp/pf_prometheus.pid) && rm /tmp/pf_prometheus.pid"
}

copy_data_to_host() {
  mkdir -p $HOST_DIR/$FOLDER
  echo_b "Copying data from $MASTER_VM to host..."
  sshpass -p "$SSH_PASSWD" rsync -avz --exclude=".*" --exclude="node_modules" --exclude="ks*" --exclude="snap" --exclude="10" --exclude="*.py" -e "ssh -o StrictHostKeyChecking=no" $SSH_USER@$MASTER_IP:/home/$SSH_USER/ $HOST_DIR/$FOLDER/$MASTER_VM
  echo_b "Copying data from $WORKER1_VM to host..."
  sshpass -p "$SSH_PASSWD" rsync -avz --exclude=".*" --exclude="node_modules" --exclude="ks*" --exclude="snap" -e "ssh -o StrictHostKeyChecking=no" $SSH_USER@$WORKER1_IP:/home/$SSH_USER/ $HOST_DIR/$FOLDER/$WORKER1_VM
  echo_b "Copying data from $WORKER2_VM to host..."
  sshpass -p "$SSH_PASSWD" rsync -avz --exclude=".*" --exclude="node_modules" --exclude="ks*" --exclude="snap" -e "ssh -o StrictHostKeyChecking=no" $SSH_USER@$WORKER2_IP:/home/$SSH_USER/ $HOST_DIR/$FOLDER/$WORKER2_VM
}

shutdown_vms() {
  echo_b "Forcing shutdown of all VMs..."
  for NODE in "$MASTER_VM" "$WORKER1_VM" "$WORKER2_VM"; do
    virsh destroy $NODE 2>/dev/null || echo "$NODE is not running or already shut down."
  done
  echo_b "All VMs have been shut down."
}

dump_tshark() (
  for NODE in "$MASTER_VM" "$WORKER1_VM" "$WORKER2_VM"; do
    echo_b "Dumping network stats in $NODE..."
    FILE="$HOST_DIR/$FOLDER/$NODE/any.pcap"

    tshark -n -r $FILE -T fields -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e frame.len >$HOST_DIR/$FOLDER/$NODE/tcp.txt
    # tshark -n -r $FILE -T fields -e frame.time_epoch -e ip.src -e ip.dst -e udp.srcport -e udp.dstport -e frame.len >$HOST_DIR/$FOLDER/$NODE/udp.txt

    # tshark -n -r $FILE -q -z conv,ip,tcp >$HOST_DIR/$FOLDER/$NODE/conv_ip_tcp.txt
    # tshark -n -r $FILE -q -z conv,ip,udp >$HOST_DIR/$FOLDER/$NODE/conv_ip_udp.txt
    # tshark -n -r $FILE -q -z conv,ip,http >$HOST_DIR/$FOLDER/$NODE/conv_ip_http.txt
    # tshark -n -r $FILE -q -z conv,ip >$HOST_DIR/$FOLDER/$NODE/conv_ip.txt
    # tshark -n -r $FILE -q -z endpoints,ip >$HOST_DIR/$FOLDER/$NODE/endpoints_ip.txt
    # tshark -n -r $FILE -q -z http_req,tree >$HOST_DIR/$FOLDER/$NODE/httpreq_tree.txt
    # tshark -n -r $FILE -q -z io,phs >$HOST_DIR/$FOLDER/$NODE/io_phs.txt
  done
)

compress_data() {
  echo_b "Compressing data..."
  (cd $HOST_DIR && 7z a $FOLDER.7z $FOLDER)
  echo_g "Capture completed. Data saved in $HOST_DIR/$FOLDER"
}

stop_monitoring
collect_pod_logs
export_prometheus_data
copy_data_to_host
shutdown_vms
dump_tshark
compress_data
