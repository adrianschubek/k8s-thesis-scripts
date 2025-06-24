#!/bin/bash

### Use attack.sh for full automation!!
### This only for testing purposes
### do not run manually

echo_b() {
  # bright blue and bold
  echo -e "\033[1;94m$*\033[0m"
}

echo_g() {
  # bright green and bold
  echo -e "\033[1;92m$*\033[0m"
}

for cmd in sshpass 7z virsh tshark; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo >&2 "$cmd is required but not installed. Aborting."
    exit 1
  }
done

# exit if ATTACK_ID missing
if [ -z "$ATTACK_ID" ]; then
  echo "ATTACK_ID is not set. Exiting."
  exit 1
fi

DATE=$(date -u +"%Y-%m-%d_%H-%M-%S")
FOLDER="$ATTACK_ID-$DATE"

# print config
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
for var in MASTER_IP WORKER1_IP WORKER2_IP HOST_DIR ATTACK_ID SSH_USER MASTER_VM WORKER1_VM WORKER2_VM ATTACK DATE FOLDER; do
  echo "$var=${!var}"
done
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# exit if FOLDER already exists
if [ -d "$HOST_DIR/$FOLDER" ]; then
  echo "Folder $FOLDER already exists in $HOST_DIR. Exiting."
  exit 1
fi

# Function to find the latest snapshot
get_latest_snapshot() {
  SNAPSHOT=$(virsh snapshot-list $1 --name | sort -V | tail -n 1)
  if [ -z "$SNAPSHOT" ]; then
    echo "No snapshot found for $1. Exiting."
    exit 1
  fi
  echo $SNAPSHOT
}

# Function to force shutdown, load snapshot, and start VMs
load_snapshot() {
  echo_b "Forcing shutdown of all VMs and loading the latest snapshots..."
  for NODE in "$MASTER_VM" "$WORKER1_VM" "$WORKER2_VM"; do
    virsh destroy $NODE 2>/dev/null || echo "$NODE is not running or already shut down."
  done

  declare -A SNAPSHOTS
  for NODE in "$MASTER_VM" "$WORKER1_VM" "$WORKER2_VM"; do
    SNAPSHOTS[$NODE]=$(get_latest_snapshot $NODE)
    virsh snapshot-revert $NODE ${SNAPSHOTS[$NODE]}
  done

  echo_b "Starting all VMs..."
  for NODE in "$MASTER_VM" "$WORKER1_VM" "$WORKER2_VM"; do
    virsh start $NODE
  done

  echo_b "Waiting for VMs to become ready..."
  for NODE in $MASTER_IP $WORKER1_IP $WORKER2_IP; do
    until sshpass -p "$SSH_PASSWD" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $SSH_USER@$NODE "echo VM $NODE is ready"; do
      sleep 2
    done
  done

  echo_b "Waiting for Kubernetes cluster to become fully ready..."
  until sshpass -p "$SSH_PASSWD" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $SSH_USER@$MASTER_IP \
    "kubectl get nodes && kubectl wait --for=condition=Ready pods --all --all-namespaces --timeout=300s && kubectl wait --for=condition=available deploy --all --all-namespaces --timeout=300s"; do
    echo_b "Kubernetes cluster is not ready yet. Retrying..."
    sleep 2
  done
  echo_b "Kubernetes cluster is ready."
}

# Main Execution
load_snapshot
