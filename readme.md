| Host            | User       | Password   | SSH Command                     |
|-----------------|------------|------------|---------------------------------|
| Host PC         | k8sserver  | kubernetes | -                               |
| k8s-master-1    | k8s        | k8s        | ssh k8s@192.168.122.216         |
| k8s-worker-1    | k8s        | k8s        | ssh k8s@192.168.122.233         |
| k8s-worker-2    | k8s        | k8s        | ssh k8s@192.168.122.228         |

> See README.md in 1_capturing and 2_preprocessing for more info.

### How to init the cluster or add new nodes
https://github.com/adrianschubek/k8s-thesis-scripts/blob/main/1_capturing/readme.md

### How to capture the data
https://github.com/adrianschubek/k8s-thesis-scripts/blob/main/1_capturing/attacks/readme.md#automated-attack-execution-and-capture

### How to preprocess the data into a dataset
https://github.com/adrianschubek/k8s-thesis-scripts/blob/main/2_preprocessing/README.md

### How to modify the cluster (create/modify/delete scenarios/pods...)
https://github.com/adrianschubek/k8s-thesis-scripts/blob/main/1_capturing/attacks/readme.md

