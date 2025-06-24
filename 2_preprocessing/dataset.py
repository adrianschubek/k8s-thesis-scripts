from syscalls import preprocess_syscalls
from podlogs import preprocess_podlogs
from network import preprocess_network
from prometheus import filter_data as filter_prom, prepare_data as prepare_prom, build_data as build_prom
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import OneHotEncoder
import os
import pandas as pd
import argparse
from natsort import natsorted
import globals

def read_nodes_datasets(args):
    DESTROOT, scenarioid, node = args
    path = os.path.join(DESTROOT, scenarioid, f"{node}.csv")
    return pd.read_csv(path).reindex()


def parrallel_syscalls(DESTROOT, ROOTPATH, scenario, node_id):
    print(f"# {scenario['id']}/{node_id} => syscalls")
    preprocess_syscalls(
        DESTROOT,
        ROOTPATH,
        scenario["id"],
        node_id,
        scenario["start_ms"],
        scenario["end_ms"],
        scenario["duration_ms"],
    )


def parrallel_podlogs(DESTROOT, ROOTPATH, scenario, node_id):
    print(f"# {scenario['id']}/{node_id} => podlogs")
    preprocess_podlogs(
        DESTROOT,
        ROOTPATH,
        scenario["id"],
        node_id,
        scenario["start_ms"],
        scenario["end_ms"],
    )


def parrallel_network(DESTROOT, ROOTPATH, scenario, node_id):
    print(f"# {scenario['id']}/{node_id} => network")
    preprocess_network(
        DESTROOT,
        ROOTPATH,
        scenario["id"],
        node_id,
        scenario["start"],
        scenario["end"],
        scenario["duration_sec"],
        scenario["start_ms"],
        scenario["end_ms"],
        scenario["duration_ms"],
    )


def get_scenarios(ROOTPATH, only=None, skip=None) -> list:
    scenarios = []

    # <scenario>/<node>
    # <scenario>/info.txt
    for folder in os.listdir(ROOTPATH):
        folder_path = os.path.join(ROOTPATH, folder)

        if os.path.isdir(folder_path):  # Ensure it's a folder
            s = {}
            files = os.listdir(folder_path)
            s["id"] = folder
            if "info.txt" in files:
                with open(os.path.join(folder_path, "info.txt"), "r", encoding="utf-8") as f:
                    for line in f:
                        key, value = line.strip().split(" ", 1)
                        s[key] = value

            s["duration_ms"] = int(s["end_ms"]) - int(s["start_ms"])
            s["start_ms"] = int(s["start_ms"])
            s["end_ms"] = int(s["end_ms"])

            nodes = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            s["nodes"] = nodes
            if only is None or only == s["id"]:
                if skip is None or skip != s["id"]:
                    scenarios.append(s)

    # sort
    scenarios = natsorted(scenarios, key=lambda x: x["id"])
    return scenarios

def main():
    parser = argparse.ArgumentParser(description="Process captured data into datasets.")
    parser.add_argument(
        "--input",
        type=str,
        help="Root folder path for data",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Destination folder path for datasets",
        required=True,
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Only preprocess the given scenario",
        default=None,
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Skips the given scenario",
        default=None,
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Output format",
        default="csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers",
        default=8,
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Resolution of the dataset (1s, 10ms,...)",
        required=True
    )

    args = parser.parse_args()
    MAX_WORKERS = int(args.workers)
    
    globals.RESOLUTION = args.resolution

    pd.set_option("mode.copy_on_write", True)

    ROOTPATH = os.path.abspath(args.input)
    DESTROOT = os.path.abspath(args.output)

    # csv = os.path.join(DESTROOT, "attack3-2025-01-22_15-18-43", "all_prom.csv")
    # df = pd.read_csv(csv, nrows=100)
    # string_df = df.select_dtypes(include=["object"])
    # print(string_df.columns)
    # exit(0)

    print(f"Input captured data: {ROOTPATH}")
    print(f"Output datasets: {DESTROOT}")
    print(f"Output formats: {args.format}")
    print(f"Output resolution: {globals.RESOLUTION}")

    # for debugging #
    # df = pd.read_csv(os.path.join(DESTROOT, "attack7-2025-01-22_16-14-05", "dataset.csv"))
    # print(df.dtypes)
    # exit(0)

    # scenarios[] -> nodes[]
    scenarios = get_scenarios(ROOTPATH, args.only, args.skip)
    print([s["id"] for s in scenarios])

    # <rootfolder>/<scenario>/<node>/...
    for scenario in scenarios:
        os.makedirs(os.path.join(DESTROOT, scenario["id"]), exist_ok=True)

    # parrallel processing for syscalls and podlogs
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for scenario in scenarios:
            for node_id in scenario["nodes"]:
                executor.submit(parrallel_syscalls, DESTROOT, ROOTPATH, scenario, node_id)
                executor.submit(parrallel_podlogs, DESTROOT, ROOTPATH, scenario, node_id)
                executor.submit(parrallel_network, DESTROOT, ROOTPATH, scenario, node_id)

    for scenario in scenarios:
        # same for all nodes (cached).
        node_id = scenario["nodes"][0]
        print(f"# {scenario['id']}/{node_id} => prometheus (filter)")
        filter_prom(
            DESTROOT,
            ROOTPATH,
            scenario["id"],
            node_id,
            scenario["start"],
            scenario["end"],
            scenario["duration_ms"],
            scenario["start_ms"],
            scenario["end_ms"],
        )

    for scenario in scenarios:
        # same for all nodes (cached).
        node_id = scenario["nodes"][0]
        print(f"# {scenario['id']}/{node_id} => prometheus (prepare)")
        prepare_prom(
            DESTROOT,
            ROOTPATH,
            scenario["id"],
            node_id,
            scenario["start"],
            scenario["end"],
            scenario["duration_ms"],
            scenario["start_ms"],
            scenario["end_ms"],
        )

    for scenario in scenarios:
        # same for all nodes (cached).
        node_id = scenario["nodes"][0]
        print(f"# {scenario['id']}/{node_id} => prometheus (build)")
        build_prom(
            DESTROOT,
            ROOTPATH,
            scenario["id"],
            node_id,
            scenario["start"],
            scenario["end"],
            scenario["duration_ms"],
            scenario["start_ms"],
            scenario["end_ms"],
        )

    meta_df = pd.DataFrame()
    meta_df_outpath = os.path.join(DESTROOT, "meta.csv")
    if os.path.exists(meta_df_outpath):
        meta_df = pd.read_csv(meta_df_outpath)
    attackid = 1  # must start at 1 (0 is or normal behavior)
    for scenario in scenarios:
        # skip if dataset already exists
        if os.path.exists(os.path.join(DESTROOT, scenario["id"], "dataset." + args.format)):
            print(f"# {scenario['id']} => Dataset already exists. Skipping...")
            attackid += 1
            continue
        for node_id in scenario["nodes"]:
            savepath = os.path.join(DESTROOT, scenario["id"], node_id + ".csv")
            if os.path.exists(savepath):
                print(f"# {scenario['id']}/{node_id} => {savepath} already exists")
                continue
            # combine all nodes-local datasets adn add node ID column
            prefix_path = os.path.join(DESTROOT, scenario["id"], node_id) + "_"
            paths = [
                prefix_path + "syscalls.csv",
                prefix_path + "podlogs.csv",
                prefix_path + "network.csv",
                os.path.join(DESTROOT, scenario["id"], "all_prom.csv"),
            ]

            # Read all CSVs in parallel using threads
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                dataframes = list(executor.map(pd.read_csv, paths))

            # Unpack the results into individual variables
            syscalls, podlogs, network, prom = dataframes

            # print shapes
            print(f"# {scenario['id']}/{node_id} => Validating shapes")
            print(f"syscalls: {syscalls.shape}")
            print(f"podlogs: {podlogs.shape}")
            print(f"network: {network.shape}")
            print(f"prom: {prom.shape}")

            if not (syscalls.shape[0] == podlogs.shape[0] == network.shape[0] == prom.shape[0]):
                raise Exception("Rows mismatch")

            node_merged = pd.concat([syscalls, podlogs, network, prom], axis=1)
            node_merged["node"] = node_id

            node_merged.to_csv(savepath, index=False)
            # node_merged.to_csv(savepath, index=False)
            print(f"# {scenario['id']}/{node_id} => {savepath} saved")

        print(f"# {scenario['id']} => Building full dataset")

        dfnodes = []
        _args = [(DESTROOT, scenario["id"], node) for node in scenario["nodes"]]
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            dfnodes = list(executor.map(read_nodes_datasets, _args))

        df = pd.concat(dfnodes).sort_index(kind="merge").reset_index(drop=True)

        # One-hot encode the 'node' column
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        node_encoded = encoder.fit_transform(df[["node"]])
        encoded_columns = encoder.get_feature_names_out(["node"])
        encoded_df = pd.DataFrame(node_encoded, columns=encoded_columns)
        df = pd.concat([df.drop(columns=["node"]), encoded_df], axis=1)

        # add attack=1 column if ATTACK=1
        if scenario["attack"] == "1":
            df["attack"] = attackid
        else:
            df["attack"] = 0

        # if a row with same scenario["id"] already exists, remove it
        if "id" in meta_df.columns:
            meta_df = meta_df[meta_df["id"] != scenario["id"]]
        meta_df = pd.concat(
            [
                meta_df,
                pd.DataFrame(
                    [
                        {
                            "dataset_id": attackid if scenario["attack"] == "1" else 0,
                            "id": scenario["id"],
                            "attack": scenario["attack"],
                            "start": scenario["start"],
                            "start_ms": scenario["start_ms"],
                            "end": scenario["end"],
                            "end_ms": scenario["end_ms"],
                            "duration_ms": scenario["duration_ms"],
                            "nodes": scenario["nodes"],
                            "nodes_len": len(scenario["nodes"]),
                            "dataset_shape": df.shape,
                            "dataset_generated_utc_timestamp": pd.Timestamp.now(tz="UTC"),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        print(f"Saving metadata to {meta_df_outpath}")
        meta_df.to_csv(meta_df_outpath, index=False)
        # FIXME use feather wayy faster write+read!!! https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#performance-considerations

        attackid += 1

        print(f"final dataset size: {df.shape}")
        
        # if "index" exists drop it (was gen by prometheus)
        if "index" in df.columns:
            print(f"# {scenario['id']} => Dropping temp 'index' col used by prometheus")
            df.drop(columns=["index"], inplace=True)
        
        print(f"# {scenario['id']} => Saving csv")
        savepath = os.path.join(DESTROOT, scenario["id"], "dataset.csv")
        df.to_csv(savepath, index=False)
        print(f"# {scenario['id']} => {savepath} saved")
        del df
        
        # # save feather
        # print(f"# {scenario['id']} => Saving feather")
        # savepath = os.path.join(DESTROOT, scenario["id"], "dataset.feather")
        # df.to_feather(savepath)
        # print(f"# {scenario['id']} => {savepath} saved")
        # del df

        # save the merged file into ROOT ID dataset.csv
        # # parquet
        # if "parquet" in args.format:
        #     print(f"# {scenario['id']} => Saving parquet")
        #     savepath = os.path.join(DESTROOT, scenario["id"], "dataset.parquet")
        #     df.to_parquet(savepath, index=False)
        #     print(f"# {scenario['id']} => {savepath} saved")
        # # csv
        # if "csv" in args.format:
        #     print(f"# {scenario['id']} => Saving csv")
        #     savepath = os.path.join(DESTROOT, scenario["id"], "dataset.csv")
        #     df.to_csv(savepath, index=False)
        #     print(f"# {scenario['id']} => {savepath} saved")
        # # xlsx
        # if "xlsx" in args.format:
        #     print(f"# {scenario['id']} => Saving xlsx")
        #     savepath = os.path.join(DESTROOT, scenario["id"], "dataset.xlsx")
        #     df.to_excel(savepath, index=False)
        #     print(f"# {scenario['id']} => {savepath} saved")
        # # json
        # if "json" in args.format:
        #     print(f"# {scenario['id']} => Saving json")
        #     savepath = os.path.join(DESTROOT, scenario["id"], "dataset.json")
        #     df.to_json(savepath, orient="records")
        #     print(f"# {scenario['id']} => {savepath} saved")

    # concat all datasets into a single one
    print("(v5+ only) Now run applylabelsv5.py to set the correct attack/benbign labels!")
    print("(<v4 only) # Concatenating all datasets")
    print("(<v4 only) Now use the command from readme step 5) to merge all scenario datsets into one dataset")
    # all_datasets_paths = []
    # for scenario in scenarios:
    #     savepath = os.path.join(DESTROOT, scenario["id"], "dataset.csv")
    #     if os.path.exists(savepath):
    #         all_datasets_paths.append(savepath)

    # all_output_path = os.path.join(DESTROOT, "all_datasets.csv")
    # if os.path.exists(all_output_path):
    #     print(f"# All datasets already saved to {all_output_path}. Please remove it to regenerate.")
    #     return
    # with open(all_output_path, "w", newline="") as outfile:
    #     writer = None
    #     for i, file in enumerate(all_datasets_paths):
    #         # Read CSV in chunks
    #         for chunk in pd.read_csv(file, chunksize=10000):
    #             print(f"Appending chunk of {file} to {all_output_path}")
    #             if i == 0 and writer is None:
    #                 # Write header only for the first file
    #                 chunk.to_csv(outfile, index=False, mode="w", header=True)
    #                 writer = True
    #             else:
    #                 # Append without header
    #                 chunk.to_csv(outfile, index=False, mode="a", header=False)

    # print(f"# All datasets saved to {all_output_path}")


if __name__ == "__main__":
    main()
