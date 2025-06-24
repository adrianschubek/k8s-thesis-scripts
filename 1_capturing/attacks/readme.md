## Guide

1. Setup the [Kubernetes cluster](https://k8s.adriansoftware.de/)

2. Install the setup of each scenario on the Kubernetes cluster first. See `<scenario>/README.md`.

3. Shutdown all VMs and create a new snapshot in virt-manager. Use version numbers e.g. `v2` for the snapshots.

4. [Capture](#automated-attack-execution-and-capture) the data.

## Scenario folder structure

```bash
<scenario_id>
   |--- run.sh
   |--- run_host.sh (alternative for run.sh. runs on host instead of master)
   |--- capture.env (optional)
```

## Configuration

Global `capture.env` example

```bash
MASTER_IP="192.168.122.216" # control plane node
WORKER1_IP="192.168.122.233"
WORKER2_IP="192.168.122.228"
HOST_DIR="/home/k8sserver/Desktop" # data output directory
SSH_USER="k8s"                     # SSH username
SSH_PASSWD="k8s"                   # SSH password
# VM Names
MASTER_VM="k8s-master-1"
WORKER1_VM="k8s-worker-1"
WORKER2_VM="k8s-worker-2"
```

### Scenario specific configuration inside each scenario folder

**Note: Check each scenario `capture.env` and modify it as requried by the scenario `readme.md`.**

`<scenario_id>/capture.env`

```bash
ATTACK_ID="attackX"                # change X to the attack number
ATTACK=1
# Scenario specific configuration (modify as required)...
```

## Automated attack execution and capture

> Attack 02 and 04 may require manual changes to run_host.sh (02) and capture.env (04) to work properly!!

### Dataset < v4: Option 1: Auto download latest scenarios


```bash
cd ~/Desktop && bash <(curl -fsSL https://k8s.adriansoftware.de/attacks/attacks.sh) --config ~/Desktop/capture.env --scenarios_download
```

### Dataset < v4: Option 2: use local scenarios folder

Download `prepare_env.sh` `start_capture.sh` and `end_capture.sh`.

```bash
cd ~/Desktop && bash <(curl -fsSL https://k8s.adriansoftware.de/attacks/attacks.sh) --config ~/Desktop/capture.env --scenarios /path/to/scenarios --only 01
```

### Dataset v5+
```bash
cd ~/Desktop && bash <(curl -fsSL https://k8s.adriansoftware.de/attacks/attacks.sh) --config ~/Desktop/capture.env --scenarios_download --only 01 --manual
```
wwait for ready then for 1iteration (iteration = how many times each attack is run) other terminal
```bash
cd ~/Desktop && bash <(curl -fsSL https://k8s.adriansoftware.de/attacks/run_attacker.sh) --config ~/Desktop/capture.env --scenarios_download --iterations 1 --timing timing_1it.txt
```

---

## Options

### Run only specific scenario(s)

```bash
--only 01
--only 05,06,07,08
```

## Only prepare the environemnt and don't run attacks.

Useful for modifying the evironment.

```bash
--prepare-only
```

---

## Troubleshooting

If `waiting for Prometheus to be ready` is stuck or `Control socket connect: Connection refused` -> re-run the script. You can use `--only <ids>` to run only the failed scenario.
