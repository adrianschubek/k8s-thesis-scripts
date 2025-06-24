#!/bin/bash
################################################################################
MASTER_IP="192.168.122.216" # control plane node
WORKER1_IP="192.168.122.233"
WORKER2_IP="192.168.122.228"
HOST_DIR="/home/k8sserver/Desktop" # data output directory
ATTACK_ID="attackX"                # change X to the attack number
SSH_USER="k8s"                     # SSH username
SSH_PASSWD="k8s"                   # SSH password
# VM Names
MASTER_VM="k8s-master-1"
WORKER1_VM="k8s-worker-1"
WORKER2_VM="k8s-worker-2"
ATTACK=1
################################################################################

TEMP_FOLDER_ID=$(
  tr -dc A-Za-z </dev/urandom | head -c 13
  echo
)

cleanup() {
  if [ "$IS_AUTO_DOWNLOAD" = true ]; then
    echo_b "Cleaning up downloaded attacks..."
    rm -rf $ROOT_ATTACKS_FOLDER
  fi
}

trap cleanup EXIT

ATTACK_DL_URL="https://k8s.adriansoftware.de/"
download_attacks_unpack_from_url() {
  echo_b "Downloading latest attacks from $ATTACK_DL_URL..."
  # go to url -> open devtools -> -> click zip download -> copy cURL
  rm -f $TEMP_FOLDER_ID.zip
  curl "$ATTACK_DL_URL" \
    -H 'cache-control: max-age=0' \
    -H 'content-type: application/x-www-form-urlencoded' \
    -o $TEMP_FOLDER_ID.zip \
    --data-raw 'download_batch%5B%5D=%2Fattacks%2F' \ 
  # download_batch[] = /attacks/

  echo_b "Unpacking attacks..."
  # remove temp folder if exists
  rm -rf ./$TEMP_FOLDER_ID
  7z x $TEMP_FOLDER_ID.zip -o$TEMP_FOLDER_ID
  cd ./$TEMP_FOLDER_ID/attacks
  IS_AUTO_DOWNLOAD=true
  ROOT_ATTACKS_FOLDER=$(pwd)
  echo_b "Attacks downloaded and unpacked to $ROOT_ATTACKS_FOLDER."
}

echo_b() {
  # bright blue and bold
  echo -e "\033[1;94m$*\033[0m"
}

echo_g() {
  # bright green and bold
  echo -e "\033[1;92m$*\033[0m"
}

echo_r() {
  # bright red and bold
  echo -e "\033[1;91m$*\033[0m"
}

for cmd in sshpass 7z virsh tshark rsync; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo >&2 "$cmd is required but not installed. Aborting."
    exit 1
  }
done

# --scenarios <folder_path>
# --config <config_file>

# Parse arguments scenario and config file
while [[ $# -gt 0 ]]; do
  case "$1" in
  --scenarios_download)
    download_attacks_unpack_from_url
    shift
    ;;
  --scenarios)
    ROOT_ATTACKS_FOLDER="$2"
    shift 2
    ;;
  --prepare-only)
    PREPARE_ONLY=true
    shift
    ;;
  --only)
    ONLY_THIS_ATTACK="$2" # run only this attack (multiple supported by comma seperated, no space)
    shift 2
    ;;
  --config)
    CONFIG_FILE="$2"
    shift 2
    ;;
  --manual)
    MANUAL=true # wait for confirmation before running each attack
    shift
    ;;
  *)
    shift
    ;;
  esac
done

# if manualmode print
if [ "$MANUAL" = true ]; then
  echo_b "Manual mode enabled. Press enter to start and stop scenario."
fi

# Load environment variables if the config file exists
if [ -f "$CONFIG_FILE" ]; then
  echo_b "Loading environment variables from $CONFIG_FILE..."
  source "$CONFIG_FILE"
else
  echo_r "Config file $CONFIG_FILE not found. Aborting."
  exist 42
fi

# if 192.168.101.0/24 via $MASTER_IP does not exist add
if ! ip route show | grep -q "192.168.101.0/24 via $MASTER_IP"; then
  echo_b "(metalLB) Adding IP route 192.168.101.0/24 via $MASTER_IP..."
  sudo ip route add 192.168.101.0/24 via $MASTER_IP
fi

# check if scenarios folder exists
if [ ! -d "$ROOT_ATTACKS_FOLDER" ]; then
  echo_r "Scenarios folder $ROOT_ATTACKS_FOLDER not found. Aborting."
  exit 43
fi

PREPARE_ENV_SCRIPT="./prepare_env.sh"
START_CAPTURE_SCRIPT="./start_capture.sh"
END_CAPTURE_SCRIPT="./end_capture.sh"

if [ ! -f "$START_CAPTURE_SCRIPT" ]; then
  echo_r "start_capture.sh not found. Aborting."
  exit 44
fi

if [ ! -f "$END_CAPTURE_SCRIPT" ]; then
  echo_r "end_capture.sh not found. Aborting."
  exit 45
fi

# loop through folder and for each folder found -> check if it has a capture.env file and run.sh file
for ATTACK_FOLDER in $ROOT_ATTACKS_FOLDER/*; do
  if [ -d "$ATTACK_FOLDER" ]; then
    ATTACK=$(basename $ATTACK_FOLDER)
    if [ ! -z ${ONLY_THIS_ATTACK+x} ]; then
      SHOULD_SKIP=1
      IFS=',' read -ra ATTACKS <<<"$ONLY_THIS_ATTACK" # Split by comma into an array
      for _attack in "${ATTACKS[@]}"; do
        if [ "$_attack" == "$ATTACK" ]; then
          SHOULD_SKIP=0
          break
        fi
      done
    fi
    if [ "$SHOULD_SKIP" == "1" ]; then
      echo_b "Skipping scenario $ATTACK"
      continue
    fi

    # subshell
    (
      DEBUG_START_TIME=$(date -u +"%H:%M:%S")
      if [ ! -f "$ATTACK_FOLDER/run.sh" ]; then
        # check for run_host.sh
        if [ -f "$ATTACK_FOLDER/run_host.sh" ]; then
          echo_b "[$ATTACK_FOLDER] Found run_host.sh"
          RUN_HOST=1
        else
          echo_r "[$ATTACK_FOLDER] Missing run.sh. Skipping scenario..."
          exit 1
        fi
      else
        echo_r "[$ATTACK_FOLDER] run.sh no longer supported. use run_host"
        exit 66
      fi
      # if run_host.sh exists (run it on host not master)
      if [ -f "$ATTACK_FOLDER/capture.env" ]; then
        echo_b "[$ATTACK_FOLDER] Loading scenario environment from $ATTACK_FOLDER/capture.env..."
        . "$ATTACK_FOLDER/capture.env"
      fi

      echo_b "[$ATTACK_FOLDER] Preparing environment..."
      . $PREPARE_ENV_SCRIPT

      if [ "$PREPARE_ONLY" = true ]; then
        echo_b "[$ATTACK_FOLDER] Skipping attack execution..."
        exit 0
      fi

      # if manual
      if [ "$MANUAL" = true ]; then
        echo_b "[$ATTACK_FOLDER] Press enter to start scenario..."
        read -r
      else
        echo_b "[$ATTACK_FOLDER] Waiting 90 seconds for cluster to stabilize..."
        sleep 30
        echo_b "[$ATTACK_FOLDER] Waiting 60 seconds for cluster to stabilize..."
        sleep 30
        echo_b "[$ATTACK_FOLDER] Waiting 30 seconds for cluster to stabilize..."
        sleep 30
      fi

      echo_b "[$ATTACK_FOLDER] Starting scenario..."
      . $START_CAPTURE_SCRIPT

      if [ "$MANUAL" = true ]; then
        echo_g "Now run run_attacker.sh"
        echo_b "[$ATTACK_FOLDER] Press enter to stop scenario..."
        read -r
      else
        echo_b "[$ATTACK_FOLDER] Running attack on host..."
        (cd $ATTACK_FOLDER && . run_host.sh)
      fi

      echo_b "[$ATTACK_FOLDER] Cleanup scenario..."
      . $END_CAPTURE_SCRIPT

      DEBUG_END_TIME=$(date -u +"%H:%M:%S")
      DEBUG_DURATION_SECONDS=$(($(date -d "$DEBUG_END_TIME" +%s) - $(date -d "$DEBUG_START_TIME" +%s)))
      echo_b "[$ATTACK_FOLDER] Capture completed in $(date -ud "@$DEBUG_DURATION_SECONDS" +%H:%M:%S)"
    )
    # skip scenario if subshell exits with not 0
    if [ $? -ne 0 ]; then
      echo_r "[$ATTACK_FOLDER] Scenario failed."
    fi
  fi
done
