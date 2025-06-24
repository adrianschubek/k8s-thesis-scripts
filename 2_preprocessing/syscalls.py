# format: 1752614 14:41:31.551018650 0 containerd-shim (13642) < close res=0
# id, time, cpu_id, command, thread_id, operation, direction_enter_exit, event_type, event_args
# https://sysdig.com/blog/interpreting-sysdig-output/
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import datetime
import os
import sys
import re
import pandas as pd
from datetime import datetime

import globals

# dataset v3 -> system calls each own column
# whitelist syscalls
syscalls_list = [
    # from kubanomaly:
    ["clone"],
    ["fork"],
    ["execve"],
    ["chdir"],
    ["open"],
    ["creat"],
    ["close"],
    ["connect"],
    ["accept"],
    ["read"],
    ["write"],
    ["unlink"],
    ["rename"],
    ["brk"],
    ["mmap"],
    ["munmap"],
    ["select"],
    ["poll"],
    ["kill"],
    # from maggiDetectingIntrusionsSystem2010:
    ["setuid"],
    ["setgid"],
    ["mount"],
    ["umount"],
    ["chown"],
    ["chmod"]
    ]

def preprocess_syscalls(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS):
    filtered_output_csv = os.path.join(DESTROOT, ID, NODE + "_temp_syscalls.csv")
    if not os.path.exists(filtered_output_csv):
        filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME)
    else:
        print(f"Skipping filtering for {ID} {NODE}")

    build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS)

def filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME):
    filtered_output_csv = os.path.join(DESTROOT, ID, NODE + "_temp_syscalls.csv")
    # read syscalls from file syscalls.txt

    if os.path.exists(filtered_output_csv):
        print(f"Filtered data already exists at {filtered_output_csv}")
        return

    ### Preprocess raw data:get only relevant data within timestamp ###

    # skip filter step if already done
    # find all sysdig_output.txt files in rootpath
    files = []
    for root, dirs, file in os.walk(os.path.join(ROOTPATH, ID, NODE)):
        for f in file:
            if f == "sysdig_output.txt":
                print("Found sysdig_output.txt file in %s" % root)
                files.append(os.path.join(root, f))

    print("Found %d sysdig_output.txt files" % len(files))

    # Initialize a list to store filtered syscalls
    filtered_data = []

    # Convert start and end times to datetime objects for comparison
    # start_time = datetime.fromtimestamp(START_TIME)
    # end_time = datetime.fromtimestamp(END_TIME)
    start_time = pd.to_datetime(START_TIME, unit="ms").replace(year=1900, month=1, day=1)
    end_time = pd.to_datetime(END_TIME, unit="ms").replace(year=1900, month=1, day=1)

    print(f"Filtering syscalls from {start_time} to {end_time}")

    pattern = re.compile(r"^\d+\s+([\d:.\-]+)\s+\d+\s+([\w\-]+)\s+\((\d+)\)\s+([<>])\s+([\w_]+)\s*(.*)$")

    # Read and process each file
    for filepath in files:
        print(f"Processing file: {filepath}")
        try:
            with open(filepath, "r") as file:
                # temp: read entire file
                f = file.readlines()
                # print progress bar
                from tqdm import tqdm
                f = tqdm(f)
                
                print("Found %d lines in file" % len(f))
                # Process each line
                for line in f:
                    match = pattern.match(line.strip())
                    if match:
                        timestamp_str, process, pid, direction, syscall, args = match.groups()
                        # print (timestamp_str, process, pid, direction, syscall, args)

                        # Convert the timestamp to a datetime object
                        timestamp = datetime.strptime(timestamp_str[:15], "%H:%M:%S.%f")
                        # print (timestamp)
                        # print (start_time, end_time)
                        # exit(1)

                        # Filter by the specified time range
                        if start_time <= timestamp <= end_time:
                            # filter syscall whitelist here already
                            if not any(syscall in syscall_group for syscall_group in syscalls_list):
                                continue
                            
                            filtered_data.append(
                                {
                                    "Timestamp": timestamp_str,
                                    # "Process": process,
                                    # "PID": pid,
                                    # "Direction": direction,
                                    "Syscall": syscall,
                                    # "Arguments": args,
                                }
                            )
            print(f"Found {len(filtered_data)} syscalls in the specified time range.")

        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            exit(1)

    # Save filtered data to a CSV file
    if filtered_data:
        print(f"Saving filtered data to {filtered_output_csv}")
        df = pd.DataFrame(filtered_data)

        # Convert the timestamp column to datetime for proper sorting
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M:%S.%f", errors="coerce")
        df = df.sort_values(by="Timestamp")  # Sort by timestamp

        df.to_csv(filtered_output_csv, index=False)
        print(f"Filtered data saved to {filtered_output_csv}")
    else:
        raise Exception("No data found in the specified time range.")


def build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_MS):
    filtered_output_csv = os.path.join(DESTROOT, ID, NODE + "_temp_syscalls.csv")
    output_csv = os.path.join(DESTROOT, ID, NODE + "_syscalls.csv")

    if os.path.exists(output_csv):
        print(f"Syscalls already preprocessed for {ID} {NODE}. Skipping...")
        return

    # Load the dataset
    print("Loading filtered data from %s" % filtered_output_csv)
    df = pd.read_csv(filtered_output_csv)

    # Ensure Timestamp is a datetime object
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    # Flatten the whitelist for easy comparison
    whitelist_syscalls = [syscall for group in syscalls_list for syscall in group]
    syscall_mapping = {syscall: group[0] for group in syscalls_list for syscall in group}
    grouped_syscall_columns = list({group[0] for group in syscalls_list})

    # Keep only whitelisted syscalls
    df = df[df["Syscall"].isin(whitelist_syscalls)]

    # Map syscalls to their groups
    df["Grouped_Syscall"] = df["Syscall"].map(syscall_mapping).fillna(df["Syscall"])

    # Aggregate occurrences into 1-second intervals
    df["Timestamp_Bucket"] = df["Timestamp"].dt.floor(globals.RESOLUTION)
    aggregated = df.groupby(["Timestamp_Bucket", "Grouped_Syscall"]).size().reset_index(name="Count")

    # Pivot to make each syscall group a column
    pivot_table = aggregated.pivot_table(
        index="Timestamp_Bucket",
        columns="Grouped_Syscall",
        values="Count",
        fill_value=0,
    )

    # Ensure all grouped syscall columns are present, with default values as 0
    for col in grouped_syscall_columns:
        if col not in pivot_table.columns:
            pivot_table[col] = 0

    # Reorder columns to match the predefined group order
    pivot_table = pivot_table[grouped_syscall_columns]

    # Generate a complete range of 1ms intervals
    # utc=True important!!! else bug
    start_time = pd.to_datetime(START_TIME, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    end_time = pd.to_datetime(END_TIME, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)

    full_index = pd.date_range(start=start_time, end=end_time, freq=globals.RESOLUTION)

    pivot_table.index = pd.to_datetime(pivot_table.index, utc=True)
    pivot_table.index = pivot_table.index.floor(globals.RESOLUTION)

    # Reindex the pivot table to include all intervals, filling missing rows with 0
    pivot_table.index.name = "Timestamp_Bucket"
    pivot_table = pivot_table.reindex(full_index, fill_value=0)

    # Save the aggregated data
    pivot_table.to_csv(output_csv, index=False)
    print(f"Preprocessed {pivot_table.shape} data saved to {output_csv}")
