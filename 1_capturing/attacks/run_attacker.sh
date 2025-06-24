# This script is run by the attacker on the host. Run this after cluster is prepared and running using attacks.sh --manual --only 01
# This script is run by the attacker on the host. Run this after cluster is prepared and running using attacks.sh --manual --only 01
# This script is run by the attacker on the host. Run this after cluster is prepared and running using attacks.sh --manual --only 01
# dataset v5



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

# --scenarios <folder_path>
# --config <config_file>

ITERATIONS=1

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
  --timing)
    OUTPUT_FILE="$2" # write output timing to this file
    shift 2
    ;;
  --only)
    ONLY_THIS_ATTACK="$2" # run only this attack (multiple supported by comma seperated, no space)
    shift 2
    ;;
  --config)
    CONFIG_FILE="$2"
    shift 2
    ;;
  --iterations)
    ITERATIONS=$2 # how many times to repeat each attack
    shift
    ;;
  *)
    shift
    ;;
  esac
done

#exit if no OUTPUT_FILE is set
if [ -z ${OUTPUT_FILE+x} ]; then
  echo_r "No OUTPUT_FILE set. Aborting."
  exit 41
fi

touch $OUTPUT_FILE
# get absolute path
OUTPUT_FILE=$(realpath $OUTPUT_FILE)

echo_b "Writing attack timing to $OUTPUT_FILE..."
echo_b "Iterations set to $ITERATIONS..."

# start required containers on the host#
echo_b "Starting required containers on the host..."
# scenario 1
sudo docker rm -f rogue-jndi
sudo docker run -d --name rogue-jndi -p 6066:1389 --rm quay.io/vicenteherrera/rogue-jndi

# sleep for 10 seconds
sleep 10

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


# Pre-collect attack folders based on ONLY_THIS_ATTACK filter
attack_folders=()
for folder in $ROOT_ATTACKS_FOLDER/*; do
  if [ -d "$folder" ]; then
    attack=$(basename "$folder")
    if [ -n "${ONLY_THIS_ATTACK+x}" ]; then
      SHOULD_INCLUDE=0
      IFS=',' read -ra ATTACKS <<<"$ONLY_THIS_ATTACK" # Split by comma into an array
      for _attack in "${ATTACKS[@]}"; do
        if [ "$_attack" == "$attack" ]; then
          SHOULD_INCLUDE=1
          break
        fi
      done
      if [ "$SHOULD_INCLUDE" -eq 1 ]; then
        attack_folders+=("$folder")
      else
        echo_b "Skipping scenario $attack"
      fi
    else
      attack_folders+=("$folder")
    fi
  fi
done

# Repeat attack scenarios x times in randomized order for each iteration
for iteration in $(seq 1 "$ITERATIONS"); do
  echo_b "Starting iteration $iteration..."
  # Randomize order using shuf
  shuffled_folders=$(printf "%s\n" "${attack_folders[@]}" | shuf)
  while IFS= read -r ATTACK_FOLDER; do
    # echo "Attack folder: $ATTACK_FOLDER"
    # continue
    # subshell per scenario
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
      # load scenario environment if capture.env exists
      if [ -f "$ATTACK_FOLDER/capture.env" ]; then
        echo_b "[$ATTACK_FOLDER] Loading scenario environment from $ATTACK_FOLDER/capture.env..."
        . "$ATTACK_FOLDER/capture.env"
      fi

      ATTACK_START_TIME=$(date -u +"%H:%M:%S")
      ATTACK_START_TIME_MS=$(date -u +"%s%3N")

      echo_b "[$ATTACK_FOLDER] Running attack on host..."
      (cd $ATTACK_FOLDER && . run_host.sh)

      ATTACK_END_TIME=$(date -u +"%H:%M:%S")
      ATTACK_END_TIME_MS=$(date -u +"%s%3N")
      ATTACK_DURATION_SECONDS=$(($(date -d "$ATTACK_END_TIME" +%s) - $(date -d "$ATTACK_START_TIME" +%s)))
      ATTACK_DURATION_MS=$(($ATTACK_END_TIME_MS - $ATTACK_START_TIME_MS))
      echo "$ATTACK_FOLDER,$ATTACK_START_TIME,$ATTACK_START_TIME_MS,$ATTACK_END_TIME,$ATTACK_END_TIME_MS,$ATTACK_DURATION_SECONDS,$ATTACK_DURATION_MS" >>$OUTPUT_FILE

      DEBUG_END_TIME=$(date -u +"%H:%M:%S")
      DEBUG_DURATION_SECONDS=$(($(date -d "$DEBUG_END_TIME" +%s) - $(date -d "$DEBUG_START_TIME" +%s)))
      echo_b "[$ATTACK_FOLDER] Capture completed in $(date -ud "@$DEBUG_DURATION_SECONDS" +%H:%M:%S)"
    )
    # check subshell result
    if [ $? -ne 0 ]; then
      echo_r "[$ATTACK_FOLDER] Scenario failed."
    fi

    # sleep for 20s between scenarios
    echo_b "[$ATTACK_FOLDER] Waiting 20 seconds before next scenario..."
    sleep 20
  done <<< "$shuffled_folders"
done

# # loop through folder and for each folder found -> check if it has a capture.env file and run.sh file
# for ATTACK_FOLDER in $ROOT_ATTACKS_FOLDER/*; do
#   if [ -d "$ATTACK_FOLDER" ]; then
#     ATTACK=$(basename $ATTACK_FOLDER)
#     if [ ! -z ${ONLY_THIS_ATTACK+x} ]; then
#       SHOULD_SKIP=1
#       IFS=',' read -ra ATTACKS <<<"$ONLY_THIS_ATTACK" # Split by comma into an array
#       for _attack in "${ATTACKS[@]}"; do
#         if [ "$_attack" == "$ATTACK" ]; then
#           SHOULD_SKIP=0
#           break
#         fi
#       done
#     fi
#     if [ "$SHOULD_SKIP" == "1" ]; then
#       echo_b "Skipping scenario $ATTACK"
#       continue
#     fi

#     # subshell
#     (
#       DEBUG_START_TIME=$(date -u +"%H:%M:%S")
#       if [ ! -f "$ATTACK_FOLDER/run.sh" ]; then
#         # check for run_host.sh
#         if [ -f "$ATTACK_FOLDER/run_host.sh" ]; then
#           echo_b "[$ATTACK_FOLDER] Found run_host.sh"
#           RUN_HOST=1
#         else
#           echo_r "[$ATTACK_FOLDER] Missing run.sh. Skipping scenario..."
#           exit 1
#         fi
#       else
#         echo_r "[$ATTACK_FOLDER] run.sh no longer supported. use run_host"
#         exit 66
#       fi
#       # if run_host.sh exists (run it on host not master)
#       if [ -f "$ATTACK_FOLDER/capture.env" ]; then
#         echo_b "[$ATTACK_FOLDER] Loading scenario environment from $ATTACK_FOLDER/capture.env..."
#         . "$ATTACK_FOLDER/capture.env"
#       fi

#       # if manual
#       if [ "$MANUAL" = true ]; then
#         echo_b "[$ATTACK_FOLDER] Press enter to start scenario..."
#         read -r
#       else
#         echo_b "[$ATTACK_FOLDER] Waiting 90 seconds for cluster to stabilize..."
#         sleep 30
#         echo_b "[$ATTACK_FOLDER] Waiting 60 seconds for cluster to stabilize..."
#         sleep 30
#         echo_b "[$ATTACK_FOLDER] Waiting 30 seconds for cluster to stabilize..."
#         sleep 30
#       fi

#       ATTACK_START_TIME=$(date -u +"%H:%M:%S")
#       ATTACK_START_TIME_MS=$(date -u +"%s%3N")

#       echo_b "[$ATTACK_FOLDER] Running attack on host..."
#       (cd $ATTACK_FOLDER && . run_host.sh)

#       ATTACK_END_TIME=$(date -u +"%H:%M:%S")
#       ATTACK_END_TIME_MS=$(date -u +"%s%3N")
#       ATTACK_DURATION_SECONDS=$(($(date -d "$ATTACK_END_TIME" +%s) - $(date -d "$ATTACK_START_TIME" +%s)))
#       ATTACK_DURATION_MS=$(($ATTACK_END_TIME_MS - $ATTACK_START_TIME_MS))
#       # write to OUTPUT_FILE + newline
#       echo "$ATTACK,$ATTACK_START_TIME,$ATTACK_START_TIME_MS,$ATTACK_END_TIME,$ATTACK_END_TIME_MS,$ATTACK_DURATION_SECONDS,$ATTACK_DURATION_MS" >>$OUTPUT_FILE

#       DEBUG_END_TIME=$(date -u +"%H:%M:%S")
#       DEBUG_DURATION_SECONDS=$(($(date -d "$DEBUG_END_TIME" +%s) - $(date -d "$DEBUG_START_TIME" +%s)))
#       echo_b "[$ATTACK_FOLDER] Capture completed in $(date -ud "@$DEBUG_DURATION_SECONDS" +%H:%M:%S)"
#     )
#     # skip scenario if subshell exits with not 0
#     if [ $? -ne 0 ]; then
#       echo_r "[$ATTACK_FOLDER] Scenario failed."
#     fi
#   fi
# done
