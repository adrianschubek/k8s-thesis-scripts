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

def preprocess_podlogs(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME):
    filtered_output_csv = os.path.join(DESTROOT, ID, NODE + "_temp_podlogs.csv")
    if not os.path.exists(filtered_output_csv):
        filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME)
    else:
        print(f"Skipping filtering for {ID} {NODE}")

    build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME)


def get_running_pods(ROOTPATH, ID, NODE):
    # get only pods running on this node
    pods = []
    # find pods.csv
    for root, dirs, file in os.walk(os.path.join(ROOTPATH, ID)):
        for f in file:
            if f == "pods.csv":
                print("Found pods.csv file in %s" % root)

                with open(os.path.join(root, f), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.split(",")[3].strip() == NODE.strip():
                            pods.append(
                                line.split(",")[1].strip()  # namespace
                                + "_"
                                + line.split(",")[0].strip()  # podname
                            )

                return pods
    return pods


def filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME):
    filtered_output_csv = os.path.join(DESTROOT, ID, NODE + "_temp_podlogs.csv")

    # get only pods running on this node
    pods = get_running_pods(ROOTPATH, ID, NODE)
    print(f"Found {len(pods)} pods running on {NODE}")

    files = []
    for root, dirs, file in os.walk(os.path.join(ROOTPATH, ID)):
        for f in file:
            # file ends with _logs.txt
            if f.replace("_logs.txt", "") in pods:
                files.append(os.path.join(root, f))

    print("Found %d log files" % len(files))

    # Initialize a list to store filtered syscalls
    filtered_data = []

    # Convert start and end times to datetime objects for comparison
    start_time = pd.to_datetime(START_TIME, unit="ms").replace(year=1900, month=1, day=1)
    end_time = pd.to_datetime(END_TIME, unit="ms").replace(year=1900, month=1, day=1)

    # Read and process each file
    for filepath in files:
        print(f"Processing file: {filepath}")
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            # Process each line
            for line in lines:
                # Example: Parse syscall information using regex
                match = re.match(
                    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(.*)$",
                    line.strip(),
                )
                if match:
                    timestamp_str, msg = match.groups()

                    # Convert the timestamp to a datetime object
                    # 2024-12-09T14:40:17.153214917Z <message>
                    timestamp = datetime.strptime(
                        (timestamp_str.split("T", 1)[1].split("Z", 1)[0])[:15],
                        "%H:%M:%S.%f",
                    ).replace(year=1900, month=1, day=1)
                    # print (timestamp)
                    # print (start_time, end_time)
                    # exit(1)

                    # Filter by the specified time range
                    if start_time <= timestamp <= end_time:
                        filtered_data.append(
                            {
                                "Timestamp": timestamp,
                                "Message": msg,
                            }
                        )
            print(f"Found {len(filtered_data)} log entries in the specified time range.")

        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            exit(1)

    # create empty file if no logs found
    if len(filtered_data) == 0:
        print("No logs found in the specified time range. Creating empty file.")
        with open(filtered_output_csv, "w") as f:
            f.write("Timestamp,Message\n")
        return

    # Save filtered data to a CSV file
    print(f"Saving filtered data to {filtered_output_csv}")
    df = pd.DataFrame(filtered_data)

    # Convert the timestamp column to datetime for proper sorting
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M:%S.%f", errors="raise")
    df = df.sort_values(by="Timestamp")  # Sort by timestamp

    df.to_csv(filtered_output_csv, index=False)
    print(f"Filtered data saved to {filtered_output_csv}")

# just join all string for this node and time. raw. later use tf-idf
def build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME):
    filtered_output_csv = os.path.join(DESTROOT, ID, NODE + "_temp_podlogs.csv")
    output_csv = os.path.join(DESTROOT, ID, NODE + "_podlogs.csv")
    
    # Load the dataset
    print("Loading filtered data from %s" % filtered_output_csv)
    df = pd.read_csv(filtered_output_csv)

    # Ensure Timestamp is a datetime object
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    df["Second"] = df["Timestamp"].dt.floor(globals.RESOLUTION)

    # Initialize columns 
    df["Pod_Logs"] = ""
    
    # Append all logs for the same globals.RESOLUTION
    df = df.groupby("Second").agg(Pod_Logs=("Message", lambda x: " ".join(x))).reset_index()
    
    # fill all missing with 0. Create a continuous time range with no gaps
    start_time = pd.to_datetime(START_TIME, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    end_time = pd.to_datetime(END_TIME, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    full_time_range = pd.date_range(start=start_time, end=end_time, freq=globals.RESOLUTION)
    
    # Create a DataFrame for the full time range with zero counts
    full_df = pd.DataFrame(full_time_range, columns=["Second"])
    
    # Merge the aggregated data with the full range, filling missing values with zero
    result_df = pd.merge(full_df, df, on="Second", how="left").fillna("")
    
    # drop Second
    result_df.drop(columns=["Second"], inplace=True)
    
    # Save the result to the output CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Preprocessed {result_df.shape} data saved to {output_csv}")

keywords_list = [
    [
        "error",
        "err",
        "erro",
        "fail",
        "fatal",
        "panic",
        "failure",
        "failed",
        "fail",
        "unauthorized",
        "denied",
        "denial",
        "critical",
        "crit",
        "exception",
        "segfault",
        "oom",
        "timeout",
    ],
]

def count_build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME):
    filtered_output_csv = os.path.join(DESTROOT, ID, NODE + "_temp_podlogs.csv")
    output_csv = os.path.join(DESTROOT, ID, NODE + "_podlogs.csv")
    
    if os.path.exists(output_csv):
        print(f"Podlogs already preprocessed for {ID} {NODE}. Skipping...")
        return

    # Load the dataset
    print("Loading filtered data from %s" % filtered_output_csv)
    df = pd.read_csv(filtered_output_csv)

    # Ensure Timestamp is a datetime object
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    df["Second"] = df["Timestamp"].dt.floor(globals.RESOLUTION)

    # Initialize columns for error and warning counts
    df["Error_Logs"] = 0

    def count_keywords(message, keywords):
        count = 0
        if isinstance(message, str):  # Ensure message is a string
            for keyword in keywords:
                count += message.lower().count(keyword)
        return count

    # Count occurrences of error and warning tokens
    df["Error_Logs"] = df["Message"].apply(lambda msg: count_keywords(msg, keywords_list[0]))

    # Group by the normalized timestamp (1-second intervals) and aggregate counts
    aggregated_df = df.groupby("Second").agg(Error_Logs=("Error_Logs", "sum")).reset_index()

    # fill all missing with 0. Create a continuous time range with no gaps
    start_time = pd.to_datetime(START_TIME, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    end_time = pd.to_datetime(END_TIME, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    full_time_range = pd.date_range(start=start_time, end=end_time, freq=globals.RESOLUTION)

    # print("full range time: ", full_time_range.shape)

    # Create a DataFrame for the full time range with zero counts
    full_df = pd.DataFrame(full_time_range, columns=["Second"])

    # Merge the aggregated data with the full range, filling missing values with zero
    result_df = pd.merge(full_df, aggregated_df, on="Second", how="left").fillna(0)

    # drop Second
    result_df.drop(columns=["Second"], inplace=True)

    # Save the result to the output CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Preprocessed {result_df.shape} data saved to {output_csv}")
