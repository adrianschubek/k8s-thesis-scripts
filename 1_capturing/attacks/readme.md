## Guide

1. Setup the [Kubernetes cluster](https://k8s.adriansoftware.de/)

2. Install the setup of each scenario on the Kubernetes cluster first. See `<scenario>/README.md`.

3. Shutdown all VMs and create a new snapshot in virt-manager. Use version numbers e.g. `v2` for the snapshots.

4. [Capture](#automated-attack-execution-and-capture) the data.

## Scenario folder structure

```bash
<scenario_id>
   |--- run_host.sh (runs on the host PC/outside the cluster)
   |--- capture.env (optional)
```

## Configuration

> File already exists on `~/Desktop/capture.env`.

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

<!-- > Attack 02 and 04 may require manual changes to run_host.sh (02) and capture.env (04) to work properly!! -->
1. Start new terminal and run on host pc:
```bash
cd ~/Desktop && bash <(curl -fsSL https://k8s.adriansoftware.de/attacks/attacks.sh) --config ~/Desktop/capture.env --scenarios_download --only 01 --manual
```
2. **wait** for ready then run in new terminal: (iteration = how many times each attack is run)
```bash
cd ~/Desktop && bash <(curl -fsSL https://k8s.adriansoftware.de/attacks/run_attacker.sh) --config ~/Desktop/capture.env --scenarios_download --iterations 1 --timing timing_1it.txt
```
3. Start capturing by pressing any key on the first terminal (quickly after step 2)
4. **wait** for step 2. to be completed

> If you dont want to bash curl the scripts, you can just download them from this repo and replace the <(curl..)> part with `bash run_attacker.sh`...

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
