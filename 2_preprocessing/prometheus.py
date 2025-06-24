from math import floor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import datetime
import os
import sys
import re
import pandas as pd
from datetime import datetime
import time
import globals


def preprocess_prom(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS, START_MS, END_MS):
    filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS, START_MS, END_MS)


def build_prom(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS, START_MS, END_MS):
    build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS, START_MS, END_MS)


def get_prom_folder_path(ROOTPATH, ID):
    for root, dirs, file in os.walk(os.path.join(ROOTPATH, ID)):
        for d in dirs:
            if d == "prom":
                print("Found prom folder in %s" % root)
                return os.path.join(root, d)
    raise FileNotFoundError(f"prom folder not found in {ROOTPATH}/{ID}")


def get_node_ip_add_from_name(ROOTPATH, ID, node_name):
    for root, dirs, file in os.walk(os.path.join(ROOTPATH, ID)):
        for f in file:
            if f == "nodes.csv":
                print("Found nodes.csv file in %s" % root)

                with open(os.path.join(root, f), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.split(",")[0].strip() == node_name.strip():
                            return line.split(",")[5].strip()

    raise FileNotFoundError(f"node {node_name} not found in {ROOTPATH}/{ID}")


def skip_metric(metric):
    return metric.startswith("prometheus") or metric.startswith("go")


#### filter (once per scenario) -> prepare (once per scenario) -> build (once per scenario)


# only runs once for all nodes. don't run for each node
# only used to extract all metric names
def filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS, START_MS, END_MS):
    # filtered_output_csv = os.path.join(ROOTPATH, ID, "filtered_prom.csv")
    output_temp_prom_metrics = os.path.join(DESTROOT, ID, "prom_metrics.csv")

    prom_path = get_prom_folder_path(ROOTPATH, ID)

    csv_files = [file for file in os.listdir(prom_path) if file.endswith(".csv") and not file.startswith("prometheus") and not file.startswith("go")]
    total_original_files = len(csv_files)
    original_file_index = 0

    ### get list of unique columns of all metrics
    metadf = pd.DataFrame()  # store filenames
    if not os.path.exists(output_temp_prom_metrics):
        for file in csv_files:
            if file.endswith(".csv"):
                original_file_index += 1
                metric = file.split(".")[0]

                # skip prometheus internal metrics
                if skip_metric(metric):
                    continue

                df = pd.read_csv(os.path.join(prom_path, metric + ".csv"))
                uniqcols = df["Metric"].unique()

                # add metrics to metadf FILENAME, METRIC, COL for each column
                dfcols = []
                uindex = 0
                for col in uniqcols:
                    uindex += 1
                    dfcols.append({"METRIC": metric, "COL": col})

                metadf = pd.concat([metadf, pd.DataFrame(dfcols)], ignore_index=True)

        metadf.to_csv(output_temp_prom_metrics, index=False)
    else:
        print(f"Loading existing metric names from {output_temp_prom_metrics}")

    metadf = pd.read_csv(output_temp_prom_metrics)

    print(f"Found total {metadf.shape[0]} unique metrics in {total_original_files} files")


# Get common metrics
def prepare_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS, START_MS, END_MS):
    common_metrics_names_path = os.path.join(DESTROOT, "common_prom_metrics.csv")
    if not os.path.exists(common_metrics_names_path):
        # find common metrics from all datasets
        # find all prom_metrics.csv paths
        prom_metrics_files = []                
        for root, dirs, files in os.walk(DESTROOT):
            for f in files:
                if f == "prom_metrics.csv":
                    prom_metrics_files.append(os.path.join(root, f))

        print(f"Found {len(prom_metrics_files)} prom_metrics.csv files")
        df = None
        for f in prom_metrics_files:
            if df is None:
                df = pd.read_csv(f)["COL"]
                continue
            # merge them on Metric
            l = df
            r = pd.read_csv(f)["COL"]
            print(f"Merging {l.shape} with {r.shape} {f}")
            print(l, r)
            df = pd.merge(l, r, on="COL", how="inner")
            print(df)
            print("==========")
            
        df.to_csv(common_metrics_names_path, index=False)
        print(f"Saved {df.shape[0]} common metrics names to {common_metrics_names_path}")
    else:
        print(f"Loading existing common metrics names from {common_metrics_names_path}")

####################################
import os
import time
import psutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import random


def wait_for_memory(min_free_mb=500, sleep_seconds=5):
    """
    Wait until at least `min_free_mb` megabytes of memory is available.
    """
    while True:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        if available_mb >= min_free_mb:
            break
        random_delay = sleep_seconds * (0.5 + 0.5 * random.random())
        print(f"Available memory ({available_mb:.2f} MB) is below threshold ({min_free_mb} MB). Waiting for {random_delay} seconds...")
        time.sleep(random_delay)


def process_file(file, prom_path, used_metrics_set, allowed_file_metrics, DESTROOT, ID, START_MS, END_MS, sanity_check_rows, min_free_mb):
    """
    Process one CSV file in the prom_path folder.
    Returns a dictionary mapping column names to series of values.
    This optimized version:
      - Converts the Timestamp column only once.
      - Uses groupby to process each metric.
    """
    import os
    import pandas as pd
    from datetime import datetime

    results = {}
    metric_from_filename = os.path.splitext(file)[0]
    # Skip if this file's metric (from its filename) is not allowed.
    if metric_from_filename not in allowed_file_metrics:
        return results

    file_path = os.path.join(prom_path, file)

    # Wait for sufficient memory before reading the file.
    wait_for_memory(min_free_mb=min_free_mb)

    try:
        # Read CSV with Timestamp parsing in one go.
        # infer_datetime_format can speed up parsing; adjust parameters as needed.
        df = pd.read_csv(file_path, parse_dates=["Timestamp"])
        # Ensure the Timestamp column is in UTC.
        if df["Timestamp"].dt.tz is None:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
        else:
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("UTC")
    except Exception as e:
        print(f"Error reading or parsing {file_path}: {e}")
        return results

    # Group the rows by the 'Metric' column.
    for col, group in df.groupby("Metric"):
        if col not in used_metrics_set:
            continue

        # Wait for memory before processing the current metric group.
        wait_for_memory(min_free_mb=min_free_mb)

        # Make a copy (groupby object may use shared memory with original df)
        metric_df = group.copy()

        # Set the Timestamp as the index.
        metric_df.set_index("Timestamp", inplace=True)

        # Build a new index with 1-second intervals.
        new_index = pd.date_range(start=pd.to_datetime(START_MS, unit="ms", utc=True), end=pd.to_datetime(END_MS, unit="ms", utc=True), freq="1s")
        try:
            # Reindex using forward-fill (pad) to fill missing timestamps.
            metric_df = metric_df.reindex(new_index, method="pad")
        except Exception as e:
            print(f"Error reindexing for {col} in file {file}: {e}")
            continue

        # Reset the index and rename it to 'Timestamp'.
        metric_df.reset_index(inplace=True)
        metric_df.rename(columns={"index": "Timestamp"}, inplace=True)

        # Check if the DataFrame has the expected number of rows.
        if metric_df.shape[0] != sanity_check_rows:
            print(f"Dropping {col} due to rows mismatch: {metric_df.shape[0]} != {sanity_check_rows}")
            continue

        # Store only the 'Value' column for merging later.
        results[col] = metric_df["Value"]
        results["Timestamp"] = metric_df["Timestamp"]
        
    del df

    import gc
    gc.collect()

    return results


# parrallel processing (new)
def build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS, START_MS, END_MS):
    start_time = time.time()  # for debug/timing
    scenario_local_prom = os.path.join(DESTROOT, ID, "prom_metrics.csv")  # scenario-specific prom metrics
    temp_out_seconds_prom_csv = os.path.join(DESTROOT, ID, "temp_s_prom.csv")
    prom_out_csv = os.path.join(DESTROOT, ID, "all_prom.csv")  # same prom for all nodes
    selected_columns_csv = os.path.join(DESTROOT, "selected_columns_rf.csv") # optional list of selected prom columns. generated after feature selection

    if not os.path.exists(temp_out_seconds_prom_csv):
        print(f"Processing metrics. This may take a while...")

        if os.path.exists(selected_columns_csv):
            selected_columns = pd.read_csv(selected_columns_csv)
            # rename first col to COL
            selected_columns.columns = ["COL"]
            # selected columns - keep_columns
            from featureselection import keep_columns
            selected_columns = selected_columns[~selected_columns["COL"].isin(keep_columns)]
            used_metrics = set(selected_columns["COL"].tolist())
            # HOTFIX remove double """" to " from selected_columns
            used_metrics = [m.replace('""""', '"') for m in used_metrics]
            used_metrics_set = set(used_metrics)
            
            # filter scenario_local_prom based on used_metrics
            scenario_local_prom_df = pd.read_csv(scenario_local_prom)
            # HOTFIX remove double """" to " from selected_columns
            selected_columns["COL"] = selected_columns["COL"].str.replace('""""', '"')
            # inner merge both COL 
            scenario_local_prom_df = pd.merge(scenario_local_prom_df, selected_columns, on="COL", how="inner")
            scenario_local_prom_df.reset_index(drop=True, inplace=True)
            # print(scenario_local_prom_df.head())
            # print(used_metrics_set)
            
            # raise if length of used_metrics is not equal to scenario_local_prom_df
            if len(used_metrics) != scenario_local_prom_df.shape[0]:
                # print missing metrics
                missing_metrics = set(used_metrics) - set(scenario_local_prom_df["COL"].tolist())
                print(f"Missing metrics: {missing_metrics}")
                raise Exception(f"Length of used metrics ({len(used_metrics)}) does not match scenario_local_prom_df ({scenario_local_prom_df.shape[0]}).")
            
            print(f"Using {len(used_metrics)} feature selected columns from {scenario_local_prom_df.shape[0]} files")
        else:
            # Read list of used metrics from the common file and convert to set for fast membership testing
            used_metrics = pd.read_csv(os.path.join(DESTROOT, "common_prom_metrics.csv"))["COL"].tolist()
            used_metrics_set = set(used_metrics)
            
            # Read the scenario-specific prometheus metrics file
            scenario_local_prom_df = pd.read_csv(scenario_local_prom)
            
            print(f"Using {len(used_metrics)} common columns from {scenario_local_prom_df.shape[0]} files")

        # Precompute the set of allowed metrics (based on the file's "METRIC" column)
        allowed_file_metrics = set(scenario_local_prom_df["METRIC"].unique())

        # Compute the expected number of rows based on the millisecond range
        sanity_check_rows = int((END_MS - START_MS) / 1000 + 1)
        prom_path = get_prom_folder_path(ROOTPATH, ID)

        min_free_mb = -1#500  # change as needed
        max_workers = 1#8  # change as needed

        # Dictionary to accumulate the results (each key is a column name and value is a pandas Series)
        processed_results = {}

        # List all CSV files in the prom_path folder
        csv_files = [file for file in os.listdir(prom_path) if file.endswith(".csv") and not file.startswith("prometheus") and not file.startswith("go")]

        # Parallel processing across files using ProcessPoolExecutor.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit a task for each file.
            futures = {
                executor.submit(
                    process_file,
                    file,
                    prom_path,
                    used_metrics_set,
                    allowed_file_metrics,
                    DESTROOT,
                    ID,
                    START_MS,
                    END_MS,
                    sanity_check_rows,
                    min_free_mb=min_free_mb,
                ): file
                for file in csv_files
            }

            total_files = len(futures)
            processed_files = 0
            for future in as_completed(futures):
                file = futures[future]
                try:
                    result = future.result()
                    # Merge the results from this file into the global dictionary.
                    processed_results.update(result)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                processed_files += 1
                # Optional: Print progress information.
                elapsed = time.time() - start_time
                print(f"Processed {processed_files}/{total_files} files in {elapsed:.2f} seconds.")

        # Combine the resulting columns into a single DataFrame.
        if processed_results:
            maindf = pd.DataFrame(processed_results)
            maindf.reset_index(inplace=True)
            # df[df.isna().any(axis=1)].columns # print columns with NaN
            # for missing values, fill forward, backward, and then fill 0
            maindf = maindf.fillna(method="ffill").fillna(method="bfill").fillna(0)

            # abort if NaN still exists
            if maindf.isnull().values.any():
                raise Exception("NaN values detected. Aborting.")

            print(f"Saving temporary prometheus (seconds) data with shape {maindf.shape} to {temp_out_seconds_prom_csv}")
            maindf.to_csv(temp_out_seconds_prom_csv, index=False)
        else:
            raise Exception("No valid metrics were processed.")
    else:
        print(f"Skipping temporary prometheus (seconds) data generation. Using existing {temp_out_seconds_prom_csv}")

    if os.path.exists(prom_out_csv):
        print(f"Skipping final prometheus data generation. Using existing {prom_out_csv}")
        return

    df = pd.read_csv(temp_out_seconds_prom_csv)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.floor(globals.RESOLUTION)
    # Reindex to 1ms from 1s
    new_index = pd.date_range(
        start=pd.to_datetime(START_MS, unit="ms", utc=True).floor(globals.RESOLUTION),
        end=pd.to_datetime(END_MS, unit="ms", utc=True).floor(globals.RESOLUTION),
        freq=globals.RESOLUTION,
    )
    df = df.set_index("Timestamp").reindex(new_index, method="pad").reset_index()

    df = df.copy()  # prevent defrag warning

    if df.isnull().values.any():
        raise Exception("NaN values detected. Aborting.")
    
    # drop level_0 timestamp
    df.drop(columns=["level_0"], inplace=True)

    df.to_csv(prom_out_csv, index=False)
    print(f"Preprocessed {df.shape} data saved to {prom_out_csv}")
