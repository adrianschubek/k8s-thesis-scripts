# parse in python: https://github.com/kbandla/dpkt
import os
from prometheus import get_node_ip_add_from_name
import pandas as pd
from datetime import datetime
import math

import globals

# tpc in out, udp in out, size in out


def preprocess_network(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_SEC, START_MS, END_MS, DURATION_MS):
    filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, START_MS, END_MS, DURATION_MS)
    build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_SEC, START_MS, END_MS, DURATION_MS)


import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import re


def filter_protocol_data(raw_csv, temp_csv, protocol, start_time, end_time):
    """Helper function to filter TCP/UDP data."""
    if os.path.exists(temp_csv):
        print(f"Skipping {protocol} filtering as {temp_csv} already exists.")
        return

    data = []
    with open(raw_csv, "r") as f:
        for line in f:
            ts, src_ip, dst_ip, src_port, dst_port, size = line.strip().split("\t")
            ts = pd.to_datetime(ts, unit="s").replace(year=1900, month=1, day=1)
            if start_time <= ts <= end_time:
                data.append(
                    {
                        "Timestamp": ts,
                        "SrcIp": src_ip,
                        "SrcPort": src_port,
                        "DstIp": dst_ip,
                        "DstPort": dst_port,
                        "Size": size,
                        "Protocol": protocol,
                    }
                )

    print(f"Found {len(data)} {protocol} packets.")
    df = pd.DataFrame(data)
    df.to_csv(temp_csv, index=False)
    print(f"Filtered {protocol} data {df.shape} saved to {temp_csv}")


def filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, START_MS, END_MS, DURATION_MS):
    jobs = []

    start_time = pd.to_datetime(START_MS, unit="ms").replace(year=1900, month=1, day=1)
    end_time = pd.to_datetime(END_MS, unit="ms").replace(year=1900, month=1, day=1)
    print(f"Filtering network traffic from {start_time} to {end_time}")

    for root, dirs, files in os.walk(os.path.join(ROOTPATH, ID)):
        for f in files:
            if f == "tcp.txt":
                print(f"Found tcp.txt file in {root}")
                tcp_raw_csv = os.path.join(root, f)
                temp_tcp_csv = os.path.join(DESTROOT, ID, f"{NODE}_temp_tcp.csv")
                if os.path.exists(temp_tcp_csv):
                    print(f"Skipping tcp filtering as {tcp_raw_csv} already exists.")
                else:
                    jobs.append((filter_protocol_data, tcp_raw_csv, temp_tcp_csv, "tcp", start_time, end_time))

            # if f == "udp.txt": # already included in tcp.txt
            #     print(f"Found udp.txt file in {root}")
            #     udp_raw_csv = os.path.join(root, f)

    # Run filtering in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        for job in jobs:
            executor.submit(*job)


def build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_SEC, START_MS, END_MS, DURATION_MS):
    tcp_temp_csv = os.path.join(DESTROOT, ID, f"{NODE}_temp_tcp.csv")
    output_csv = os.path.join(DESTROOT, ID, f"{NODE}_network.csv")
    pods_csv = None
    for root, dirs, files in os.walk(os.path.join(ROOTPATH, ID)):
        for f in files:
            if f == "pods.csv":
                pods_csv = os.path.join(root, f)
                break

    if not os.path.exists(pods_csv):
        raise FileNotFoundError(f"pods.csv not found in {root}")

    if os.path.exists(output_csv):
        print(f"Network already preprocessed for {ID} {NODE}. Skipping...")
        return

    if not os.path.exists(tcp_temp_csv):
        raise FileNotFoundError(f"TCP temp file not found in {tcp_temp_csv}")

    local_ips = pd.read_csv(pods_csv)
    # local pod IPs for this node
    local_ips = local_ips[local_ips["NODE"] == NODE]["IP"].unique().tolist()
    print(f"Found {len(local_ips)} pod IPs for node {NODE}")

    temp_df = pd.read_csv(tcp_temp_csv)
    temp_df["Timestamp"] = pd.to_datetime(temp_df["Timestamp"], utc=True)
    temp_df["Timestamp"] = temp_df["Timestamp"].dt.floor(globals.RESOLUTION)

    start_time = pd.to_datetime(START_MS, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    end_time = pd.to_datetime(END_MS, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    full_index = pd.date_range(start=start_time, end=end_time, freq=globals.RESOLUTION)

    # output structure: Inbound Count, Outbound Count, Size In, Size Out
    df = pd.DataFrame(full_index, columns=["Timestamp"])

    ip_pattern = r"(?:^|,)\s*(" + "|".join([re.escape(ip) for ip in local_ips]) + r")\s*(?:,|$)"

    # Fill NaN with empty string before string operations
    temp_df["SrcIp"] = temp_df["SrcIp"].fillna("")
    temp_df["DstIp"] = temp_df["DstIp"].fillna("")

    # Create boolean masks (explicitly handle NaN)
    src_contains = temp_df["SrcIp"].str.contains(ip_pattern, regex=True)
    dst_contains = temp_df["DstIp"].str.contains(ip_pattern, regex=True)
    filtered = temp_df[src_contains | dst_contains]

    # Classify direction (outbound if SrcIp contains IP) (inbound if DstIp contains IP)
    outbound_mask = filtered["SrcIp"].str.contains(ip_pattern, regex=True)
    inbound_mask = filtered["DstIp"].str.contains(ip_pattern, regex=True)
    # inbound_mask = (~outbound_mask) & filtered["DstIp"].str.contains(ip_pattern, regex=True)

    # Split into outbound/inbound
    outbound = filtered[outbound_mask.fillna(False)]
    inbound = filtered[inbound_mask.fillna(False)]

    # Rest of aggregation remains the same...
    outbound_group = outbound.groupby("Timestamp", as_index=False).agg(NetOutCount=("Size", "count"), NetOutSize=("Size", "sum"))
    inbound_group = inbound.groupby("Timestamp", as_index=False).agg(NetInCount=("Size", "count"), NetInSize=("Size", "sum"))

    # Merge with full timestamp index
    df = df.merge(outbound_group, on="Timestamp", how="left")
    df = df.merge(inbound_group, on="Timestamp", how="left")
    df = df.fillna(0).astype({"NetOutCount": int, "NetOutSize": int, "NetInCount": int, "NetInSize": int})[
        ["Timestamp", "NetInCount", "NetOutCount", "NetInSize", "NetOutSize"]
    ]

    # # drop Timestamp (keep Timestamp! v5+)
    # df.drop(columns=["Timestamp"], inplace=True)

    # Save the aggregated data
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed {df.shape} data saved to {output_csv}")


def old_filter_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME):
    pcapfile = os.path.join(ROOTPATH, ID, NODE, "any.pcap")
    if not os.path.exists(pcapfile):
        raise FileNotFoundError(f"pcap file not found in {ROOTPATH}/{ID}/{NODE}")

    packets = PcapReader(pcapfile)

    frames = []

    for packet in packets:
        # utc time HH:mm:ss in utc seconds epoch
        # print(datetime.strptime(START_TIME, "%H:%M:%S"))
        ts = datetime.utcfromtimestamp(math.ceil(packet.time)).replace(year=1900, month=1, day=1)
        # print(ts)

        if ts >= datetime.strptime(START_TIME, "%H:%M:%S") and ts <= datetime.strptime(END_TIME, "%H:%M:%S"):
            if TCP in packet and IP in packet:
                protocol = "tcp"
                ip_layer = packet[IP]
                tcp_layer = packet[TCP]
                src_ip = ip_layer.src
                src_port = tcp_layer.sport
                dst_ip = ip_layer.dst
                dst_port = tcp_layer.dport
            elif UDP in packet and IP in packet:
                protocol = "udp"
                ip_layer = packet[IP]
                udp_layer = packet[UDP]
                src_ip = ip_layer.src
                src_port = udp_layer.sport
                dst_ip = ip_layer.dst
                dst_port = udp_layer.dport
            else:
                continue

            # size
            size = len(packet)
            # src and/or dst is node ip or both are local loopback on this node
            frames.append(
                {
                    "Time": ts,
                    "SrcIp": src_ip,
                    "SrcPort": src_port,
                    "Dst": dst_ip,
                    "DstPort": dst_port,
                    "Size": size,
                    "Protocol": protocol,
                }
            )

    pd.DataFrame(frames).to_csv(os.path.join(DESTROOT, ID, NODE + "_temp_network.csv"), index=False)
    print(f"Filtered data saved to {os.path.join(DESTROOT, ID, NODE + '_temp_network.csv')}")


def old_build_data(DESTROOT, ROOTPATH, ID, NODE, START_TIME, END_TIME, DURATION_SEC):
    filtered_csv = os.path.join(DESTROOT, ID, NODE + "_temp_network.csv")
    output_csv = os.path.join(DESTROOT, ID, NODE + "_network.csv")

    df = pd.read_csv(filtered_csv)

    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"].dt.floor("s")

    aggregated_df = (
        df.groupby("Time")
        .apply(
            lambda x: pd.Series(
                {
                    "Tcp": (x[x["Protocol"] == "tcp"].shape[0]),
                    "Udp": (x[x["Protocol"] == "udp"].shape[0]),
                    "Size": x["Size"].sum(),
                }
            )
        )
        .reset_index()
    )

    start_time = datetime.strptime(START_TIME, "%H:%M:%S").replace(year=1900, month=1, day=1)
    end_time = datetime.strptime(END_TIME, "%H:%M:%S").replace(year=1900, month=1, day=1)

    full_time_range = pd.date_range(start=start_time, end=end_time, freq="s")
    full_df = pd.DataFrame(full_time_range, columns=["Time"])
    result_df = pd.merge(full_df, aggregated_df, on="Time", how="right").fillna(0)

    result_df.drop(columns=["Time"])

    result_df.to_csv(output_csv, index=False)
    print(f"Network {result_df.shape} data saved to {output_csv}")
