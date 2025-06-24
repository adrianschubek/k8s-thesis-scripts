import requests
import argparse
import csv
from datetime import datetime


# Set up the argument parser
parser = argparse.ArgumentParser(description="Process start and end time.")
# parser.add_argument('--start', type=str, required=True, help="Start time in the format 'YYYY-MM-DD HH:MM:SS'")
# parser.add_argument('--end', type=str, required=True, help="End time in the format 'YYYY-MM-DD HH:MM:SS'")
parser.add_argument("--since", type=str, help="in minutes")
parser.add_argument("--url", type=str, help="prometheus url")

# Parse the arguments
args = parser.parse_args()
PROMETHEUS_URL = args.url

# Convert the input strings to UNIX timestamps
# START = datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S').timestamp()
# END = datetime.strptime(args.end, '%Y-%m-%d %H:%M:%S').timestamp()

# calculate the start time using since x minutes
START = datetime.now().timestamp() - int(args.since) * 60
END = datetime.now().timestamp()

STEP = '1'           # Step in seconds (e.g., 60s for 1-minute resolution)

# Function to fetch all metric names
def get_all_metrics(prometheus_url):
    response = requests.get(f"{prometheus_url}/api/v1/label/__name__/values")
    response.raise_for_status()
    return response.json()['data']

# Function to fetch metric data
def get_metric_data(prometheus_url, metric_name, start, end, step):
    response = requests.get(f"{prometheus_url}/api/v1/query_range", params={
        'query': metric_name,
        'start': start,
        'end': end,
        'step': step,
    })
    response.raise_for_status()
    return response.json()

# Fetch all metrics
metrics = get_all_metrics(PROMETHEUS_URL)
print(f"Found {len(metrics)} metrics. Processing... (this may take a while)")

# Export data for each metric
for metric in metrics:
    # print(f"Processing metric: {metric}")
    data = get_metric_data(PROMETHEUS_URL, metric, START, END, STEP)
    results = data['data']['result']

    # Write each metric to its own CSV file
    with open(f"{metric}.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Metric', 'Timestamp', 'Value'])
        # Write rows
        for result in results:
            metric_labels = result['metric']
            metric_name = metric_labels.pop('__name__', metric)
            labels = ",".join([f'{key}="{value}"' for key, value in metric_labels.items()])
            for value in result['values']:
                timestamp = datetime.utcfromtimestamp(float(value[0])).strftime('%Y-%m-%d %H:%M:%S')
                csvwriter.writerow([f"{metric_name}{{{labels}}}", timestamp, value[1]])

print("All metrics exported.")
