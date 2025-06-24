<!-- | Dataset       | Description / Updates                                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------------------------------- |
| v5it3ms10     | single run/no snapshot restore. Each attack run 3 times. 10ms resolution. Idle time between attacks count as benign traffic |
| v5it3ms10rm99 | single run/no snapshot restore. Each attack run 3 times. 10ms resolution. Idle time between attacks removed                 |
| v4        | uses snapshort load/restore leading to prometheus alignment issues. all attacks now from outside the cluster                          |

The `net2` versions use more advanced network features than the default ones.

The `sampled` versions are sampled from the original dataset where the benign traffic is randomly reduced to a 3:1 benign-attacks ratio. Also two of the three class 8 attack samples are removed as they are very similar and too large (F1 still very good). -->

---

# Generate Dataset, Feature Engineering and Machine Learning

### 1)

Clone https://github.com/adrianschubek/k8s-thesis-scripts

```bash
git clone https://github.com/adrianschubek/k8s-thesis-scripts && cd k8s-thesis/1_preprocessing
```

### 2)

Setup virtual environment using uv https://github.com/astral-sh/uv and install dependencies

```bash
uv venv -p 3.12

# linux
. .venv/bin/activate
# windows
.\venv\Scripts\activate

uv pip install -r requirements.txt
```

Install PyTorch https://pytorch.org/get-started/locally/

Example for Nvidia GPUs:
```bash
uv pip install -U torch --index-url https://download.pytorch.org/whl/cu126
```

(Linux & Nvidia only) install cuDF to speed up processing

```bash
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==24.12.*"
```

### 3)

Create a new `raw_data` folder under `preprocessing` on your workstation.
```bash
mkdir raw_data && cd raw_data 
```

Transfer the created `.7z` files from the host PC to `raw_data` on your work PC. (using e.g. USB stick, RustDesk...)
> Preprocessing needs at least 32GB RAM + SWAP (Max RAM+SWAP usage ~95GB for 15mins of data). Trying to do this on the host PC will likely result in an OOM error.

And extract the `.7z` files in the `raw_data` folder.
```bash
7z x <filename>
```
<!-- for file in *.7z; do 7z x "$file"; done -->

Copy the `timing1it.txt` file from the host PC to the `raw_data` folder. 
> This file contains precise timestamps for when each attack was executed. It is used to align all data sources to the same time axis.

### 4)

Go back to preprocessing folder.

Folder structure should look like this

```
preprocessing/
├── raw_data/
│   ├── attack0/...
│   │   ├── k8s-master-1/...
│   │   ├── ...
│   ├── attack1/...
|   ├── timing1it.txt
│   ...
├── dataset.py
├── ...
```

Run the script to generate the dataset

<!-- > If using a large benign traffic you should skip it and run the dataset generation script twice. 1. for all attacks and 2. for the benign traffic. Use `--skip benign_folder_name` to skip the benign traffic. Only process the benign traffic AFTER you selected the features in step 6 using the attacks dataset! -->

> Set the `--resolution` option to `Xs`/`Xms` e.g `1s` for 1 second resolution or `10ms` for 10 milliseconds resolution. As an example we use 1 second.

```bash
python dataset.py --input raw_data --output datasets --resolution 1s
```

(Linux & Nvidia only) cuDF version (recommended)

```bash
python -m cudf.pandas dataset.py --input raw_data --output datasets --resolution 1s
```

You can also use a full absolute path for both input and output folders.

<!-- #### More options

##### Skips a scenario

```
--skip <scenario_name>
```

##### Process only one scenario

```
--only <scenario_name>
```

##### Output format

```
--format <csv|parquet|xlsx|json>
``` -->

<!-- export CUDF_COPY_ON_WRITE="1" &&  -->

#### Output files

`<scenario>` is the name of the first scenario, e.g. `attack0`

> Note: There will just be a *single* folder with the name of the first scenario for everything. This is expected and the correct labels will be set later. So `attack1...` actually includes all scenarios.

`<node>` is the name of the node, e.g. `k8s-master-1`

| File                                   | Description                                                                                   |
| -------------------------------------- | --------------------------------------------------------------------------------------------- |
| \<scenario>/all_prom.csv               | Preprocessed Prometheus data in correct resolution                                            |
| \<scenario>/dataset.csv                | Final dataset for this scenario (All `<node>.csv` merged)                                     |
| \<scenario>/\<node>.csv                | Final dataset for this node only                                                              |
| \<scenario>/\<node>\_network.csv       | Preprocessed network traffic for this node                                                    |
| \<scenario>/\<node>\_podlogs.csv       | Preprocessed podlogs for this node                                                            |
| \<scenario>/\<node>\_syscalls.csv      | Preprocessed system calls for this node                                                       |
| \<scenario>/\<node>\_temp_podlogs.csv  | Trimmed and filtered version of podlogs for this node                                         |
| \<scenario>/\<node>\_temp_syscalls.csv | Trimmed and filtered version of system calls for this node                                    |
| \<scenario>/\<node>\_temp_tcp.csv      | Trimmed and filtered version of network traffic for this node                                 |
| \<scenario>/prom_metrics.csv           | All available Prometheus metric names for this scenario                                       |
| \<scenario>/temp_s_prom.csv            | Prometheus data in seconds interval                                                           |
| common_prom_metrics.csv                | Prometheus metrics that are common to all scenarios and which will be included in the dataset |
| meta.csv                               | Metadata for each dataset (includes the dataset attack ID column mapping)                     |
<!-- | all_datasets.csv                       | All datasets merged into one file                                                             | -->

Process can be interrupted and resumed at any time. Temp files are cached and reused.

### 5) 
> Skipped
<!-- ### 5) Merge all scenario datasets

In datasets folder run:

> Only needed for multiple scenarios. Skip when processing just benign traffic or any single scenario.

```bash
first_file=$(find . -type f -name "dataset.csv" | sort -V | head -n 1)
echo "Merging dataset files in order..."

# Extract header from the first file
head -n 1 "$first_file" > merged_all_datasets.csv

# Append all files, skipping headers (tail -n +2)
find . -type f -name "dataset.csv" | sort -V | while read file; do
    tail -n +2 "$file" >> merged_all_datasets.csv
    echo "Merged $file"
done

echo "Merged CSV saved as merged_all_datasets.csv"
```

After processing, the dataset files will be saved as `dataset.csv` in each scenario folder.

`attack` column in the final dataset maps to `attack_id` in `meta.csv`. `attack` will always be `0` for normal behavior (when `attack` was set to `1`). -->

### 6) Feature Engineering

<!-- > Only for attack scenario dataset. Skip for benign dataset. After this step you can create the benign dataset in step 4. -->
> Remove `-m cudf.pandas` if you are not using cuDF.

```bash
python -m cudf.pandas featureselection.py --input datasets/attack1-XXXX/dataset.csv --output datasets
```

Final optimized dataset will be saved as `all_datasets_rf.csv` in the `datasets` folder.
A `all_datasets_rf.xlsx` file for debugging purposes in Excel will also be created.

#### Dataset optimization

1. Removes zero-variance columns
2. Removes highly correlated (>95%) columns
3. Random Forest feature selection (0.001 feature importance threshold)

Intermediate files are saved as `all_datasets_*.csv` and `selected_columns_*.csv` in the same folder.

### 6) Apply labels

```bash
python -m cudf.pandas applylabelsv5.py --dataset datasets/attack1-XXXX/dataset.csv --timing raw_data/timing_1it.txt --resolution 1s --between-as-benign true
```

Replace `attack1-XXXX` with the actual folder name. Will create a new file `dataset_labeled.csv` in the same folder

<!-- ```bash
python -m cudf.pandas splitdataset.py --dataset local_datasets_v5it3ms10/attack1-2025-03-27_22-32-47/dataset_labeled.csv --output local_datasets_v5it3ms10 --split-by-benign true
``` -->

### 7) Advanced Network stats

<!-- python -m cudf.pandas network2.py --rawdata raw_data_v2 --input local_datasets_v2/all_datasets_rf.csv --output local_datasets_v2 -->
```bash
python -m cudf.pandas network2.py --rawdata raw_data --input datasets/attack1-XXXX/dataset_labeled.csv --scenario attack1-XXXX --output datasets --workers 1
```

Replace `attack1-XXXX` with the actual folder name

### 8) Machine Learning

> You can skip this step if you want to use your own machine learning models instead.

Note: "Pod_Logs" column contains the raw logs as string. So you need to use something like tf-idf to convert it to a numerical representation before using it in a model.

```bash
python ml.py --input datasets/all_datasets_rf.csv --output models
```

> Do NOT run with `-m cudf.pandas`.

Evaluation metrics are saved as `<model>_<normalization>.txt` in the `models` folder.

---

<!-- Tested and working on Ubuntu 24.04 LTS with 24 GB RAM + swap, RTX 3070 8GB (cuDF version only) and 5600X 6-core CPU. -->
