import pandas as pd
import numpy as np


# Run this AFTER everything else. optional

import argparse
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler


def main():
    parser = argparse.ArgumentParser(description="split dataset into attack/benign or individual attacks. v5+")
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to the dataset",
        required=True,
    )
    parser.add_argument(
        "--split-nodes",
        type=lambda x: str(x).lower() == "true",
        help="true = split by nodes, false = split by attacks",
        default=True,
    )
    parser.add_argument(
        "--split-by-benign",
        type=lambda x: str(x).lower() == "true",
        help="true = benign (2 sets) or false = attacks (x sets)",
        default=True,
    )
    parser.add_argument(
        "--rolling-windows",
        type=lambda x: str(x).lower() == "true",
        help="",
        default=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="out folder path",
        required=True,
    )
    args = parser.parse_args()
    out_folder = os.path.abspath(args.output)
    in_csv = os.path.abspath(args.dataset)
    print(f"Reading {in_csv}")
    print(f"Writing to folder {out_folder}")
    if args.split_nodes:
        print("Split by nodes")
    else:
        print(f"Split by benign: {'attacks/benign' if args.split_by_benign else 'individual attacks'}")

    df = pd.read_csv(in_csv)

    if args.split_nodes:
        # Split by nodes
        # Get columns that start with "node-"
        node_columns = [col for col in df.columns if col.startswith("node_k8s-")]
        node_names = [col.replace("node_", "") for col in node_columns]
        print(f"Detected nodes: {node_names}")

        # Create a dataset for each node
        for i, node in enumerate(node_names):
            node_csv = in_csv.replace(".csv", f"_node_{node}.csv")
            node_col = node_columns[i]
            # Filter rows where this node column has data
            node_df = df[df[node_col].notna() & (df[node_col] != 0)]
            node_df.reset_index(drop=True, inplace=True)
            node_df.to_csv(node_csv, index=False)
            print(node_df.shape)
            print(f"Node dataset {node} saved to {node_csv}")
        return

    if args.split_by_benign:
        # Split by benign
        benign_df = df[df["attack"] == 0]
        attacks_df = df[df["attack"] != 0]
        benign_df.reset_index(drop=True, inplace=True)
        attacks_df.reset_index(drop=True, inplace=True)

        # force same order as beign df
        attacks_df = attacks_df[benign_df.columns]

        # Save benign and attack datasets
        benign_df.to_csv(os.path.join(out_folder, "final_benign_dataset.csv"), index=False)
        print(benign_df.shape)
        print(f"Benign dataset saved to {os.path.join(out_folder, 'final_benign_dataset.csv')}")
        attacks_df.to_csv(os.path.join(out_folder, "final_attacks_dataset.csv"), index=False)
        print(attacks_df.shape)
        print(f"Attacks dataset saved to {os.path.join(out_folder, 'final_attacks_dataset.csv')}")
    else:
        # split by attack label in order! save each section as a separate csv + zscore version
        # 0001110033 -> 0_0.csv, 1_0.csv, 0_1.csv, 3_0.csv  <class>_<idx>.csv
        print("Splitting dataset by consecutive attack labels...")
        os.makedirs(out_folder, exist_ok=True)  # Ensure output directory exists
        os.makedirs(os.path.join(out_folder, "l2"), exist_ok=True)  # Ensure output directory exists
        os.makedirs(os.path.join(out_folder, "zscore"), exist_ok=True)  # Ensure output directory exists
        os.makedirs(os.path.join(out_folder, "minmax"), exist_ok=True)  # Ensure output directory exists

        if args.rolling_windows:
            print("Using rolling windows")
            from ml import make_rolling_window

            window_sizes = [2, 5, 10, 50, 100]
            all_window_dfs = []
            for window_size in window_sizes:
                print(f"Creating rolling window of size {window_size}")
                rolling_df = make_rolling_window(df, window_size)
                rolling_df.reset_index(drop=True, inplace=True)
                all_window_dfs.append(rolling_df)

            nodeassign_idx = df.columns.get_loc("node_k8s-master-1")
            left_cols = df.columns[:nodeassign_idx]
            right_cols = df.columns[nodeassign_idx:]
            # Concatenate all window dataframes with the original dataframe
            df = pd.concat([df[left_cols]] + all_window_dfs + [df[right_cols]], axis=1)

        attack_labels = df["attack"].tolist()
        if not attack_labels:
            print("Dataset is empty. No blocks to save.")
            return  # Exit if the dataframe is empty

        start_index = 0
        current_label = attack_labels[0]
        label_indices = {}  # Dictionary to store the next index for each label {label: next_idx}

        for i in range(1, len(attack_labels)):
            if attack_labels[i] != current_label:
                # --- End of a block ---
                block_df = df.iloc[start_index:i].copy()
                block_df.reset_index(drop=True, inplace=True)

                # Get index for this label
                idx = label_indices.get(current_label, 0)
                label_indices[current_label] = idx + 1

                # --- Save original block ---
                filename = os.path.join(out_folder, f"{current_label}_{idx}.csv")
                block_df.to_csv(filename, index=False)
                print(f"Saved block {current_label}_{idx} (rows: {len(block_df)}) to {filename}")

                # # --- Save L2 normalized block (normalized individually) ---
                l2_df = block_df.copy()
                numeric_cols = l2_df.select_dtypes(include=np.number).columns.tolist()

                # Exclude the 'attack' label column itself from normalization
                if "attack" in numeric_cols:
                    numeric_cols.remove("attack")

                if numeric_cols and not l2_df[numeric_cols].empty:  # Only normalize if there are numeric columns and data
                    # Initialize Normalizer for this specific block
                    l2_normalizer = Normalizer(norm="l2")
                    # Apply L2 normalization row-wise to numeric columns of this block
                    l2_df[numeric_cols] = l2_normalizer.fit_transform(l2_df[numeric_cols])

                l2_filename = os.path.join(out_folder, "l2", f"{current_label}_{idx}_l2norm.csv")
                l2_df.to_csv(l2_filename, index=False)
                print(f"Saved L2 norm block {current_label}_{idx} (rows: {len(l2_df)}) to {l2_filename}")

                # --- zscore (standardscaler)
                z_df = block_df.copy()
                numeric_cols = z_df.select_dtypes(include=np.number).columns.tolist()
                # Exclude the 'attack' label column itself from normalization
                if "attack" in numeric_cols:
                    numeric_cols.remove("attack")
                if numeric_cols and not z_df[numeric_cols].empty:
                    # Initialize StandardScaler for this specific block
                    z_scaler = StandardScaler()
                    # Apply z-score normalization to numeric columns of this block
                    z_df[numeric_cols] = z_scaler.fit_transform(z_df[numeric_cols])
                z_filename = os.path.join(out_folder, "zscore", f"{current_label}_{idx}_zscore.csv")
                z_df.to_csv(z_filename, index=False)
                print(f"Saved zscore block {current_label}_{idx} (rows: {len(z_df)}) to {z_filename}")

                # --- minmax (minmaxscaler)
                minmax_df = block_df.copy()
                numeric_cols = minmax_df.select_dtypes(include=np.number).columns.tolist()
                # Exclude the 'attack' label column itself from normalization
                if "attack" in numeric_cols:
                    numeric_cols.remove("attack")
                if numeric_cols and not minmax_df[numeric_cols].empty:
                    # Initialize MinMaxScaler for this specific block
                    minmax_scaler = MinMaxScaler()
                    # Apply minmax normalization to numeric columns of this block
                    minmax_df[numeric_cols] = minmax_scaler.fit_transform(minmax_df[numeric_cols])
                minmax_filename = os.path.join(out_folder, "minmax", f"{current_label}_{idx}_minmax.csv")
                minmax_df.to_csv(minmax_filename, index=False)
                print(f"Saved minmax block {current_label}_{idx} (rows: {len(minmax_df)}) to {minmax_filename}")

                # --- Start next block ---
                start_index = i
                current_label = attack_labels[i]

        # --- Handle the very last block ---
        block_df = df.iloc[start_index:].copy()
        block_df.reset_index(drop=True, inplace=True)

        idx = label_indices.get(current_label, 0)
        label_indices[current_label] = idx + 1

        # --- Save original last block ---
        filename = os.path.join(out_folder, f"{current_label}_{idx}.csv")
        block_df.to_csv(filename, index=False)
        print(f"Saved block {current_label}_{idx} (rows: {len(block_df)}) to {filename}")

        # # --- Save L2 normalized last block (normalized individually) ---
        l2_df = block_df.copy()
        numeric_cols = l2_df.select_dtypes(include=np.number).columns.tolist()
        if "attack" in numeric_cols:
            numeric_cols.remove("attack")

        if numeric_cols and not l2_df[numeric_cols].empty:
            # Initialize Normalizer for this specific block
            l2_normalizer = Normalizer(norm="l2")
            # Apply L2 normalization row-wise to numeric columns of this block
            l2_df[numeric_cols] = l2_normalizer.fit_transform(l2_df[numeric_cols])

        l2_filename = os.path.join(out_folder, "l2", f"{current_label}_{idx}_l2norm.csv")
        l2_df.to_csv(l2_filename, index=False)
        print(f"Saved L2 norm block {current_label}_{idx} (rows: {len(l2_df)}) to {l2_filename}")

        # --- zscore (standardscaler)
        z_df = block_df.copy()
        numeric_cols = z_df.select_dtypes(include=np.number).columns.tolist()
        # Exclude the 'attack' label column itself from normalization
        if "attack" in numeric_cols:
            numeric_cols.remove("attack")
        if numeric_cols and not z_df[numeric_cols].empty:
            # Initialize StandardScaler for this specific block
            z_scaler = StandardScaler()
            # Apply z-score normalization to numeric columns of this block
            z_df[numeric_cols] = z_scaler.fit_transform(z_df[numeric_cols])
        z_filename = os.path.join(out_folder, "zscore", f"{current_label}_{idx}_zscore.csv")
        z_df.to_csv(z_filename, index=False)
        print(f"Saved zscore block {current_label}_{idx} (rows: {len(z_df)}) to {z_filename}")

        # --- minmax (minmaxscaler)
        minmax_df = block_df.copy()
        numeric_cols = minmax_df.select_dtypes(include=np.number).columns.tolist()
        # Exclude the 'attack' label column itself from normalization
        if "attack" in numeric_cols:
            numeric_cols.remove("attack")
        if numeric_cols and not minmax_df[numeric_cols].empty:
            # Initialize MinMaxScaler for this specific block
            minmax_scaler = MinMaxScaler()
            # Apply minmax normalization to numeric columns of this block
            minmax_df[numeric_cols] = minmax_scaler.fit_transform(minmax_df[numeric_cols])
        minmax_filename = os.path.join(out_folder, "minmax", f"{current_label}_{idx}_minmax.csv")
        minmax_df.to_csv(minmax_filename, index=False)
        print(f"Saved minmax block {current_label}_{idx} (rows: {len(minmax_df)}) to {minmax_filename}")

    # <v4 only:
    # attacks_df = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v4/all_datasets_rf.csv")
    # benign_df = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v4/benign_dataset.csv")
    # print(attacks_df.shape)
    # print(benign_df.shape)

    # # remove all " from column names to prevent mismatches due to too many escpaed " created in csv by pandas
    # attacks_df.columns = [col.replace('"', '') for col in attacks_df.columns]
    # benign_df.columns = [col.replace('"', '') for col in benign_df.columns]

    # # force same order as attacks df
    # benign_df = benign_df[attacks_df.columns]
    # print(benign_df)

    # # sanity check: make sure all columns are same
    # assert attacks_df.columns.all() == benign_df.columns.all()

    # # Sanity check: no NaN values except Pod_Logs
    # for col in attacks_df.columns:
    #     if col != 'Pod_Logs':
    #         assert not attacks_df[col].isnull().values.any(), f"NaN values found in attacks_df column: {col}"

    # for col in benign_df.columns:
    #     if col != 'Pod_Logs':
    #         assert not benign_df[col].isnull().values.any(), f"NaN values found in benign_df column: {col}"

    # attacks_df.to_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v4/final_attacks_dataset.csv", index=False)
    # benign_df.to_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v4/final_benign_dataset.csv", index=False)


if __name__ == "__main__":
    main()
