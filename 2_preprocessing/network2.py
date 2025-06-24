from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import os
from dataset import get_scenarios
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from sklearn.preprocessing import OneHotEncoder

import globals


def parse_pcap(file_path):
    packets = rdpcap(file_path)
    flows = defaultdict(list)

    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            proto = "TCP" if TCP in pkt else "UDP"
            src, dst = pkt[IP].src, pkt[IP].dst
            sport, dport = (pkt[TCP].sport, pkt[TCP].dport) if TCP in pkt else (pkt[UDP].sport, pkt[UDP].dport)
            flow_key = (src, dst, sport, dport, proto)
            flows[flow_key].append(pkt)

    return flows


def extract_features(flows):
    feature_list = []

    for flow_key, packets in flows.items():
        timestamps = [pkt.time for pkt in packets]
        sizes = [len(pkt) for pkt in packets]
        fw_packets = [pkt for pkt in packets if (pkt[IP].src, pkt[IP].dst) == (flow_key[0], flow_key[1])]
        bw_packets = [pkt for pkt in packets if (pkt[IP].src, pkt[IP].dst) == (flow_key[1], flow_key[0])]

        # Compute basic statistics
        fl_dur = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        tot_fw_pk = len(fw_packets)
        tot_bw_pk = len(bw_packets)
        tot_l_fw_pkt = sum(len(pkt) for pkt in fw_packets)
        fw_pkt_l_max = max([len(pkt) for pkt in fw_packets]) if fw_packets else 0
        fw_pkt_l_min = min([len(pkt) for pkt in fw_packets]) if fw_packets else 0
        fw_pkt_l_avg = np.mean([len(pkt) for pkt in fw_packets]) if fw_packets else 0
        fw_pkt_l_std = np.std([len(pkt) for pkt in fw_packets]) if fw_packets else 0

        bw_pkt_l_max = max([len(pkt) for pkt in bw_packets]) if bw_packets else 0
        bw_pkt_l_min = min([len(pkt) for pkt in bw_packets]) if bw_packets else 0
        bw_pkt_l_avg = np.mean([len(pkt) for pkt in bw_packets]) if bw_packets else 0
        bw_pkt_l_std = np.std([len(pkt) for pkt in bw_packets]) if bw_packets else 0

        fl_byt_s = sum(sizes) / fl_dur if fl_dur > 0 else 0
        fl_pkt_s = len(packets) / fl_dur if fl_dur > 0 else 0

        # Inter-arrival times
        if len(timestamps) > 1:
            iat = np.diff(sorted(timestamps))
            # to float
            iat = [float(i) for i in iat]
            fl_iat_avg = np.mean(iat)
            fl_iat_std = np.std(iat)
            fl_iat_max = np.max(iat)
            fl_iat_min = np.min(iat)
        else:
            fl_iat_avg = fl_iat_std = fl_iat_max = fl_iat_min = 0

        # TCP Flag Counts
        fin_cnt = sum(1 for pkt in packets if TCP in pkt and pkt[TCP].flags & 0x01)
        syn_cnt = sum(1 for pkt in packets if TCP in pkt and pkt[TCP].flags & 0x02)
        rst_cnt = sum(1 for pkt in packets if TCP in pkt and pkt[TCP].flags & 0x04)
        pst_cnt = sum(1 for pkt in packets if TCP in pkt and pkt[TCP].flags & 0x08)
        ack_cnt = sum(1 for pkt in packets if TCP in pkt and pkt[TCP].flags & 0x10)
        urg_cnt = sum(1 for pkt in packets if TCP in pkt and pkt[TCP].flags & 0x20)

        down_up_ratio = (tot_fw_pk / tot_bw_pk) if tot_bw_pk > 0 else 0
        pkt_size_avg = np.mean(sizes) if sizes else 0
        fw_seg_avg = fw_pkt_l_avg
        bw_seg_avg = bw_pkt_l_avg

        subfl_fw_pk = tot_fw_pk / 2 if tot_fw_pk > 1 else tot_fw_pk
        subfl_fw_byt = tot_l_fw_pkt / 2 if tot_l_fw_pkt > 1 else tot_l_fw_pkt
        subfl_bw_pkt = tot_bw_pk / 2 if tot_bw_pk > 1 else tot_bw_pk
        subfl_bw_byt = sum(len(pkt) for pkt in bw_packets) / 2 if bw_packets else 0
        fw_win_byt = sum(pkt[TCP].window for pkt in fw_packets if TCP in pkt) if fw_packets else 0
        bw_win_byt = sum(pkt[TCP].window for pkt in bw_packets if TCP in pkt) if bw_packets else 0
        fw_act_pkt = sum(1 for pkt in fw_packets if len(pkt) > 0)
        fw_seg_min = min([len(pkt) for pkt in fw_packets]) if fw_packets else 0

        for timestamp in timestamps:  # add for every packet timestamp not just first
            timestamp = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
            feature_list.append(
                {
                    # "flow_key": flow_key,  # object
                    "timestamp": timestamp,
                    "fl_dur": fl_dur,
                    "tot_fw_pk": tot_fw_pk,
                    "tot_bw_pk": tot_bw_pk,
                    "tot_l_fw_pkt": tot_l_fw_pkt,
                    "fw_pkt_l_max": fw_pkt_l_max,
                    "fw_pkt_l_min": fw_pkt_l_min,
                    "fw_pkt_l_avg": fw_pkt_l_avg,
                    "fw_pkt_l_std": fw_pkt_l_std,
                    "bw_pkt_l_max": bw_pkt_l_max,
                    "bw_pkt_l_min": bw_pkt_l_min,
                    "bw_pkt_l_avg": bw_pkt_l_avg,
                    "bw_pkt_l_std": bw_pkt_l_std,
                    "fl_byt_s": fl_byt_s,
                    "fl_pkt_s": fl_pkt_s,
                    "fl_iat_avg": fl_iat_avg,
                    "fl_iat_std": fl_iat_std,
                    "fl_iat_max": fl_iat_max,
                    "fl_iat_min": fl_iat_min,
                    "fin_cnt": fin_cnt,
                    "syn_cnt": syn_cnt,
                    "rst_cnt": rst_cnt,
                    "pst_cnt": pst_cnt,
                    "ack_cnt": ack_cnt,
                    "urg_cnt": urg_cnt,
                    "down_up_ratio": down_up_ratio,
                    "pkt_size_avg": pkt_size_avg,
                    "fw_seg_avg": fw_seg_avg,
                    "bw_seg_avg": bw_seg_avg,
                    "subfl_fw_pk": subfl_fw_pk,
                    "subfl_fw_byt": subfl_fw_byt,
                    "subfl_bw_pkt": subfl_bw_pkt,
                    "subfl_bw_byt": subfl_bw_byt,
                    "fw_win_byt": fw_win_byt,
                    "bw_win_byt": bw_win_byt,
                    "fw_act_pkt": fw_act_pkt,
                    "fw_seg_min": fw_seg_min,
                }
            )

    # soll 77 columns!

    return pd.DataFrame(feature_list)


def process_nodepcap(pcap, idx, is_normal, nodes_labels, this_node_label, start_ms, end_ms):
    start_time = pd.to_datetime(start_ms, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    end_time = pd.to_datetime(end_ms, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(globals.RESOLUTION)
    print(f"Processing {pcap}...{start_time} -> {end_time} ({(end_ms - start_ms)} ms)")
    flows = parse_pcap(pcap)
    df = extract_features(flows)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp"] = df["timestamp"].apply(lambda x: x.replace(year=1900, month=1, day=1))

    # filter start_time <= timestamp <= end_time
    df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]

    df["timestamp"] = df["timestamp"].dt.floor(globals.RESOLUTION)

    # remove NaN rows
    df = df.dropna()

    # cast object to float
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(float)

    # group by tiemstamp mean
    df = df.groupby("timestamp").mean().reset_index()

    full_index = pd.date_range(start=start_time, end=end_time, freq=globals.RESOLUTION)

    result_df = pd.DataFrame(full_index, columns=["timestamp"])

    result_df = pd.merge(result_df, df, on="timestamp", how="left")
    # result_df = result_df.fillna(method="ffill").fillna(method="bfill").fillna(0) # Dont do this here keep it 0!!! filling manipulates the data
    result_df = result_df.fillna(0)

    result_df["attack"] = 0 if is_normal else idx
    # set column node = 1 if this_node_label == node else 0
    # for node in nodes_labels:
    #     result_df[node] = 1 if this_node_label == node else 0
    result_df["node"] = this_node_label

    print(result_df.shape)
    print(result_df)
    return (idx, result_df)


def mergewithattacksandbenigndatasets(temp_path, attacks_path, benign_path):
    raise ValueError("This function is not used anymore v5+. Use mergewithdataset() instead.")
    attacks_df = pd.read_csv(attacks_path)
    benign_df = pd.read_csv(benign_path)
    temp_df = pd.read_csv(temp_path)

    # drop attack column
    temp_df.drop(columns=["attack", "timestamp"], inplace=True)

    print(attacks_df.shape)
    print(attacks_df.head(5))
    print(benign_df.shape)
    print(benign_df.head(5))
    print(temp_df.shape)
    print(temp_df.head(5))

    # attack row + benign rows == temp_df rows
    if attacks_df.shape[0] + benign_df.shape[0] != temp_df.shape[0]:
        raise ValueError("attack row + benign rows != temp_df rows")

    attacksout = os.path.abspath(attacks_path).replace(".csv", "") + "_net2.csv"
    benignout = os.path.abspath(benign_path).replace(".csv", "") + "_net2.csv"
    print(f"Output attacks: {attacksout}")
    print(f"Output benign: {benignout}")

    # print(attacks_df["attack"].unique())
    # print(benign_df["attack"].unique())
    # print(temp_df.columns)

    # select rows atack0..attack10..benign
    # all_datasets_rf -> [attack0..attack10] idx
    # benign -> [benign] idx
    # from idx 0 select all rows until "attack" col == 0
    # temp_df = temp_df[temp_df["attack"] != 0]

    temp_attack_df = temp_df.copy()[0 : attacks_df.shape[0]]
    temp_benign_df = temp_df.copy()[attacks_df.shape[0] :]

    # print shapes
    print(temp_attack_df.shape)
    print(temp_benign_df.shape)
    # print heads
    print(temp_attack_df.head(5))
    print(temp_benign_df.head(5))
    # make sure they add up
    if temp_attack_df.shape[0] + temp_benign_df.shape[0] != temp_df.shape[0]:
        raise ValueError("temp_attack_df + temp_benign_df != temp_df")

    # check for Nan intemp_df
    if temp_df.isnull().values.any():
        raise ValueError("temp_df has NaN values.")

    # Reset index to avoid potential issues during concatenation
    attacks_df = attacks_df.reset_index(drop=True)
    benign_df = benign_df.reset_index(drop=True)
    temp_attack_df = temp_attack_df.reset_index(drop=True)
    temp_benign_df = temp_benign_df.reset_index(drop=True)

    # concat columns
    print("Concatenating columns...")
    attack_df = pd.concat([attacks_df, temp_attack_df], axis=1)
    benign_df = pd.concat([benign_df, temp_benign_df], axis=1)

    # print heads
    print(attack_df.head(5))
    print(benign_df.head(5))

    # save to csv
    attack_df.to_csv(attacksout, index=False)
    print(f"Saved to {attacksout}")
    benign_df.to_csv(benignout, index=False)
    print(f"Saved to {benignout}")

    pass


def mergewithdataset(temp_path, dataset_path, output_path):
    ds_df = pd.read_csv(dataset_path)
    temp_df = pd.read_csv(temp_path)
    print(ds_df.shape)
    print(ds_df.head(5))
    print(temp_df.shape)
    print(temp_df.head(5))

    # rop attack column
    temp_df.drop(columns=["attack", "timestamp"], inplace=True)

    # check for Nan intemp_df
    if temp_df.isnull().values.any():
        raise ValueError("temp_df has NaN values.")

    # raise if rows are not equal
    if ds_df.shape[0] != temp_df.shape[0]:
        raise ValueError("Rows are not equal. Should throw if seperate benign+attacks datasets. use mergewithattacksandbenigndatasets()")

    # concat columns. old
    # df = pd.concat([ds_df, temp_df], axis=1)

    # Find the index of NetOutSize column
    net_out_idx = ds_df.columns.get_loc("NetOutSize")

    # Split the original dataframe into two parts
    left_columns = ds_df.columns[: net_out_idx + 1]  # Columns up to and including NetOutSize
    right_columns = ds_df.columns[net_out_idx + 1 :]  # Columns after NetOutSize

    # Create the new dataframe with desired column order
    df = pd.concat(
        [
            ds_df[left_columns],  # First part of ds_df
            temp_df,  # All temp_df columns
            ds_df[right_columns],  # Rest of ds_df
        ],
        axis=1,
    )

    print(df.columns)
    print(df.shape)
    print(df.head(5))

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # # save xlsx
    # output_xlsx = output_path.replace(".csv", ".xlsx")
    # df.to_excel(output_xlsx, index=False)
    # print(f"Saved to {output_xlsx}")


# Run this with the FULL non-splited labelled dataset !!!
# Run this with the FULL non-splited labelled dataset !!!
# Run this with the FULL non-splited labelled dataset !!!


def main():
    parser = argparse.ArgumentParser(description="Process captured data into datasets.")
    parser.add_argument(
        "--rawdata",
        type=str,
        help="Folder path to raw_data containing pcap files",
        required=True,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name to process (single scenario folder name)",
        required=True,
    )
    parser.add_argument(
        "--input",
        type=str,
        help="File path to full labelled dataset .csv",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Destination folder path for datasets",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers",
        default=8,
    )
    parser.add_argument("--resolution", type=str, help="Resolution of the dataset (1s, 10ms,...)", required=True)
    args = parser.parse_args()
    globals.RESOLUTION = args.resolution
    MAX_WORKERS = int(args.workers)
    print(f"Using {MAX_WORKERS} workers.")
    # input
    ROOTPATH = os.path.abspath(args.rawdata)
    print(f"Input file: {args.input}")
    DESTROOT = os.path.abspath(args.output)
    print(f"Destination folder: {DESTROOT}")

    final_dsoutput = os.path.abspath(args.input).replace(".csv", "") + "_net2.csv"
    print(f"Final output: {final_dsoutput}")

    pd.set_option("mode.copy_on_write", True)

    output_csv = os.path.join(DESTROOT, args.scenario + "_network2_stats_temp.csv")
    print(f"Temp data will be saved to {output_csv}")
    if os.path.exists(output_csv):
        print(f"Skipping as {output_csv} already exists.")
        mergewithdataset(output_csv, args.input, final_dsoutput)
        return

    scenarios = [s for s in get_scenarios(ROOTPATH) if s["id"] == args.scenario]
    if len(scenarios) == 0:
        raise ValueError(f"No scenarios found in {ROOTPATH} for {args.scenario}.")
    if len(scenarios) > 1:
        raise ValueError(f"More than one scenario found in {ROOTPATH} for {args.scenario}.")

    print(f"Found {len(scenarios)} scenarios.")
    print(scenarios)
    # find pcap files for each node
    scenario_dfs = []
    for scenario in scenarios:  # bakcward compatibility. len==1 always now
        start_ms = scenario["start_ms"]
        end_ms = scenario["end_ms"]
        pcaps = []
        for node in scenario["nodes"]:
            pcap = os.path.join(ROOTPATH, scenario["id"], node, "any.pcap")
            if not os.path.exists(pcap):
                raise FileNotFoundError(f"PCAP file not found in {pcap}")
            pcaps.append((pcap, scenario["id"], scenario["nodes"], node, start_ms, end_ms))
        print(f"Found {len(scenarios[0]['nodes'])} x {len(scenarios)} = {len(pcaps)} pcap files.")

        temp_nodes_df = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for idx, (pcap, is_normal, nodes_labels, this_node_label, start_ms, end_ms) in enumerate(pcaps):
                futures.append(executor.submit(process_nodepcap, pcap, idx, is_normal, nodes_labels, this_node_label, start_ms, end_ms))

            # add to temp_nodes_df in idx order
            for future in as_completed(futures):
                idx, df = future.result()
                temp_nodes_df.append((idx, df))

        # sort by idx
        temp_nodes_df = sorted(temp_nodes_df, key=lambda x: x[0])
        temp_nodes_df = [x[1] for x in temp_nodes_df]

        # concat all nodes
        # df = pd.concat(temp_nodes_df, axis=0, ignore_index=True)
        df = pd.concat(temp_nodes_df).sort_index(kind="merge").reset_index(drop=True)
        # one hot encode node (from dataset.py
        #### uncomment to check that stucutre is equal to dataset.py
        # encoder = OneHotEncoder(sparse_output=False, drop=None)
        # node_encoded = encoder.fit_transform(df[["node"]])
        # encoded_columns = encoder.get_feature_names_out(["node"])
        # encoded_df = pd.DataFrame(node_encoded, columns=encoded_columns)
        # df = pd.concat([df.drop(columns=["node"]), encoded_df], axis=1)
        # print(df.shape)
        # print(df)
        ####
        df.drop(columns=["node"], inplace=True)
        scenario_dfs.append(df)

    # concat all scenarios
    df = pd.concat(scenario_dfs)

    # save to csv
    df.to_csv(output_csv, index=False)
    print(f"Temp data saved to {output_csv}")

    mergewithdataset(output_csv, args.input, final_dsoutput)


if __name__ == "__main__":
    main()
