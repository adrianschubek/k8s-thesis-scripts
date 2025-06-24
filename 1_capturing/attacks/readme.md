<!-- ## Guide

1. Setup the [Kubernetes cluster](https://k8s.adriansoftware.de/)

2. Install the setup of each scenario on the Kubernetes cluster first. See `<scenario>/README.md`.

3. Shutdown all VMs and create a new snapshot in virt-manager. Use version numbers e.g. `v2` for the snapshots.

4. [Capture](#automated-attack-execution-and-capture) the data. -->

# Modify the cluster (create/delete scenarios/pods...)
> Example scenario https://github.com/adrianschubek/k8s-thesis-scripts/tree/main/1_capturing/attacks/01

## Guide

1. Download all `*.sh` scripts to your Desktop from this repo https://github.com/adrianschubek/k8s-thesis-scripts/tree/main/1_capturing/attacks
2. Run in host terminal
```bash
cd ~/Desktop && bash attacks.sh --config ~/Desktop/capture.env --scenarios_download --only 01 --prepare-only
```
3. **Wait** for the script to finish/exit.
4. Connect to the master node and make changes
5. Disconnect from the master, open virtual-machine-manager and **shutdown** all VMs
6. **Wait** for the VMs to be fully shutdown.
7. Open the `Snapshots` tab in each VM and create a new snapshot with a version number (e.g. `v2`). The version number MUST be higher than all previous snapshots.


## Scenario folder structure

```bash
<scenario_id>
   |--- run_host.sh (runs on the host PC/outside the cluster/"attack script")
   |--- capture.env (configuration for the scenario)
```

## Configuration

### Global `capture.env` example
> File already exists on `~/Desktop/capture.env`. Skip this step

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

<!-- **Note: Check each scenario `capture.env` and modify it as requried by the scenario `readme.md`.** -->

`<scenario_id>/capture.env`

```bash
ATTACK_ID="attackX"                # change X to the attack number
ATTACK=1
# Scenario specific configuration (modify as required)...
```

> `ATTACK_ID` is used to identify the attack in the dataset. They should be unique for each scenario. Format is `attackX` where X is the attack number (e.g. `attack1`, `attack2`, etc.).

> `ATTACK` can be either 1 (for malicious) or 0 (for benign). If you want to run a benign scenario, set `ATTACK=0` and `ATTACK_ID` to something like `benignX` where X is the benign scenario number.

# Automated attack execution and capture

> The following commands download the 10 existing scenarios. If you have modified or added new scenarios you must set a local folder path instead 
>```bash 
>--scenarios /path/to/scenarios
>```
> and remove the `--scenarios_download` option.

<!-- > If you dont want to execute the scripts from the server, you can just download them from this repo and replace the <(curl..)> part with `bash run_attacker.sh`... -->

0. Download all `*.sh` scripts to your Desktop from this repo https://github.com/adrianschubek/k8s-thesis-scripts/tree/main/1_capturing/attacks
1. Start new terminal and run on host pc:
```bash
cd ~/Desktop && bash attacks.sh --config ~/Desktop/capture.env --scenarios_download --only 01 --manual
```
2. **wait** for ready status. then run in new terminal: (iterations = how many times each attack is run)
```bash
cd ~/Desktop && bash run_attacker.sh --config ~/Desktop/capture.env --scenarios_download --iterations 1 --timing timing_1it.txt
```
3. Start capturing by pressing any key on the first terminal (*immediately* after step 2)
4. **wait** for step 2. to be completed



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
