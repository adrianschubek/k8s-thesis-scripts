# feature selection for Prometheus metrics
# using random forest and statistical tests (e.g. always 0...)
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse
import os
import multiprocessing as mp
from multiprocessing import sharedctypes
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

keep_columns = [
    "attack",
    "node_k8s-master-1",
    "node_k8s-worker-1",
    "node_k8s-worker-2",
    "Pod_Logs",  # string
    # from kubanomaly:
    "clone",
    "fork",
    "execve",
    "chdir",
    "open",
    "creat",
    "close",
    "connect",
    "accept",
    "read",
    "write",
    "unlink",
    "rename",
    "brk",
    "mmap",
    "munmap",
    "select",
    "poll",
    "kill",
    # from maggiDetectingIntrusionsSystem2010:
    "setuid",
    "setgid",
    "mount",
    "umount",
    "chown",
    "chmod",
    # base network features
    "NetInCount",
    "NetOutCount",
    "NetInSize",
    "NetOutSize",
]

def main():
    parser = argparse.ArgumentParser(description="Process captured data into datasets.")
    parser.add_argument(
        "--input",
        type=str,
        help="File path to all_datasets.csv",
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

    args = parser.parse_args()
    MAX_WORKERS = int(args.workers)
    # input
    print(f"Input file: {args.input}")
    DESTROOT = os.path.abspath(args.output)
    print(f"Destination folder: {DESTROOT}")

    pd.set_option("mode.copy_on_write", True)

    # always keep these columns. dont optimize them away
    
    # # load *_network, _podlogs, syscalls.csv to get columns names to keep
    ##### Uncomment this to find columns to keep!
    # df = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets/attack3-2025-01-22_15-18-43/k8s-master-1_network.csv")
    # keep_columns.extend(df.columns.tolist())
    # df = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets/attack3-2025-01-22_15-18-43/k8s-master-1_podlogs.csv")
    # keep_columns.extend(df.columns.tolist())
    # df = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets/attack3-2025-01-22_15-18-43/k8s-master-1_syscalls.csv")
    # keep_columns.extend(df.columns.tolist())
    # print(keep_columns)
    # exit(1)
    #####

    # print(f"Loading data from {args.input}")
    # ddf = dd.read_csv(args.input, sample=100000000, blocksize="1GB")

    # df1 = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets/selected_columns_hicorr.csv")
    # df2 = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets/selected_columns_rf.csv")
    # # print same columns
    # print(set(df1.columns).intersection(df2.columns))
    # exit(1)

    # df = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v2/all_datasets_hicorr.csv")
    # # df = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets/all_datasets_rf.csv")
    # # print first 50 column node_k8s-master-1
    # print(df["Pod_Logs"].unique())
    # exit()

    # df = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v2/all_datasets_hicorr.csv")
    # # print coluns containing NaN
    # print(df.columns[df.isna().any()].tolist()) # should just be pod_logs
    # exit()

    in_csv = args.input  # ~37k rows
    selcols_file = os.path.join(DESTROOT, "selected_columns_novar.csv")
    if not os.path.exists(selcols_file):
        print("Removing zero variance features... finding columns")
        remove_zero_variance_features_find(in_csv, selcols_file, keep_columns)
    else:
        print(f"Using selected columns from {selcols_file}")

    novarout_csv = os.path.join(DESTROOT, "all_datasets_novar.csv")
    if not os.path.exists(novarout_csv):
        print("Removing zero variance features... applying filter")
        apply_cols(in_csv, novarout_csv, selcols_file)  # ~30k rows
    else:
        print(f"Using filtered data from {novarout_csv}")

    hicorr_selcols_file = os.path.join(DESTROOT, "selected_columns_hicorr.csv")
    if not os.path.exists(hicorr_selcols_file):
        print("Removing highly correlated features...finding columns")
        remove_highly_correlated_features_numpy_find(novarout_csv, hicorr_selcols_file, keep_columns)
    else:
        print(f"Using filtered columns from {hicorr_selcols_file}")

    hicorr_csv = os.path.join(DESTROOT, "all_datasets_hicorr.csv")
    if not os.path.exists(hicorr_csv):
        print("Removing highly correlated features...applying filter")
        apply_cols(novarout_csv, hicorr_csv, hicorr_selcols_file)  # ~4k rows
    else:
        print(f"Using filtered data from {hicorr_csv}")

    rf_selcols_file = os.path.join(DESTROOT, "selected_columns_rf.csv")
    if not os.path.exists(rf_selcols_file):
        print("Random forest feature selection...finding columns")
        random_forest_find(hicorr_csv, rf_selcols_file, keep_columns)
    else:
        print(f"Using filtered columns from {rf_selcols_file}")

    rf_csv = os.path.join(DESTROOT, "all_datasets_rf.csv")
    if not os.path.exists(rf_csv):
        print("Random forest feature selection...applying filter")
        apply_cols(hicorr_csv, rf_csv, rf_selcols_file)
    else:
        print(f"Using filtered data from {rf_csv}")

    # drop xlsx for debugging
    rf_xlsx = os.path.join(DESTROOT, "all_datasets_rf.xlsx")
    if not os.path.exists(rf_xlsx):
        df = pd.read_csv(rf_csv)
        df.to_excel(rf_xlsx, index=False)


def remove_zero_variance_features_find(input_file, selcols_file, keep_columns, chunk_size=15000):
    global_uniquevals: pd.DataFrame = None

    # First pass: Determine unique values per column
    iterations = 0  # each chunk i same  adds + 1 so compare with iterations!
    for idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {idx} to determine unique values")
        iterations += 1

        # Compute number of unique values per column
        uniquevals = chunk.nunique()

        # Aggregate unique value counts across chunks
        global_uniquevals = uniquevals if global_uniquevals is None else global_uniquevals.add(uniquevals, fill_value=0)

    # Ensure we have collected unique value counts
    if global_uniquevals is None:
        print("No data processed.")
        return []

    # Determine columns to keep (i.e., those with more than one unique value)
    print("Determining columns to keep...")
    selected_columns = global_uniquevals[global_uniquevals > iterations].index.tolist()

    for col in keep_columns:
        if col not in selected_columns:
            selected_columns.append(col)

    # Print results
    print(f"Total columns before filtering: {len(global_uniquevals)}")
    print(f"Total columns after filtering: {len(selected_columns)}")

    # writing selected columns to file
    selcols_df = pd.DataFrame(selected_columns)
    print(f"Writing selected columns to file {selcols_file}")
    selcols_df.to_csv(selcols_file, index=False)


def apply_cols(input_file, output_file, selcols_file, chunk_size=15000):
    print(f"Applying selected columns from {selcols_file} to file {input_file} -> {output_file}")
    selected_columns = pd.read_csv(selcols_file).values.flatten().tolist()
    print(f"Length of selected columns: {len(selected_columns)}")
    
    with open(output_file, mode='w', newline='') as f_out:
        for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, usecols=selected_columns)):
            print(f"Processing chunk {i}")
            chunk.to_csv(f_out, index=False, header=(i == 0))
            print(f"Chunk {i} written to {output_file}")


def remove_highly_correlated_features_numpy_find(input_file, selcols_file, keep_columns, chunk_size=10000, corr_threshold=0.95):
    selected_columns = None
    first_chunk = True

    # Process the file in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        print("Processing chunk...")
        chunk = chunk.select_dtypes(include=[np.number])  # Select only numerical columns
        print(f"Chunk shape: {chunk.shape}")

        if first_chunk:
            print("np.corrcoef np.triu")
            upper_tri = np.triu(np.corrcoef(chunk, rowvar=False), k=1)  # Compute correlation matrix then Upper triangle without diagonal
            print("np.triu done")
            to_remove = set()

            print("np.where")
            rows, cols = np.where(np.abs(upper_tri) > corr_threshold)
            print("np.where done")
            # to_remove.update(chunk.columns[j] for i, j in zip(rows, cols) if i < j)
            mask = rows < cols
            print("mask done")
            to_remove.update(chunk.columns[cols[mask]])
            print("to_remove done")
            # temp fix: add level_0 (timestamp) to remove. may not be neded bc we filter select_dtype already
            to_remove.add("level_0")

            selected_columns = [col for col in chunk.columns if col not in to_remove]
            for col in keep_columns:
                if col not in selected_columns:
                    selected_columns.append(col)

            print(f"Total columns before filtering: {len(chunk.columns)}")
            print(f"Total columns after filtering: {len(selected_columns)}")
            # Save selected columns
            print(f"Writing selected columns to file {selcols_file}")
            col_df = pd.DataFrame(selected_columns, columns=["Kept_Features"])
            col_df.to_csv(selcols_file, index=False)

            first_chunk = False
            return


def random_forest_find(input_csv, selcols_file, keep_columns):
    # Step 1: Load Your Data (Assume 'df' is your dataset)
    # 'df' should contain Prometheus metrics as features and a target column (e.g., "label")
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)

    # Split data into features (X) and labels (y)
    X = df.drop(columns=["attack", "Pod_Logs"])  # target column name. dont include raw pod logs
    y = df["attack"]
    
    # experiment. drop columns contain "prometheus" in name
    # X = X.loc[:, ~X.columns.str.contains("prometheus", case=False)]
    
    # drop columns containing inf values
    print("Dropping columns with inf values")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="any")
    
    # Step 2: Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Step 3: Train a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    
    # print classifcation report
    y_pred = rf.predict(X_test)
    print(f"Classification report before topfeatures: {classification_report(y_test, y_pred)}")

    # Step 4: Get Feature Importances
    feature_importances = rf.feature_importances_

    # Step 5: Create a DataFrame for feature importance ranking
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    
    # Step 6: Select Top N Features (e.g., Top 100 most important features)
    # print how many features have > 0.001 importance
    # print(f"Features with importance > 0.001: {len(importance_df[importance_df['Importance'] > 0.001])}")
    # print(importance_df[importance_df["Importance"] > 0.001])
    # exit(1)
    # N = 100  # how many features to keep
    # top_features = importance_df.head(N)["Feature"].tolist()

    # keep all features with importance > 0.001
    top_features = importance_df[importance_df["Importance"] > 0.001]["Feature"].tolist()

    # # Step 7: Reduce Dataset to Selected Features
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    # # Step 8: Train a New Model with Reduced Features
    rf_reduced = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_reduced.fit(X_train_selected, y_train)

    # # Evaluate Model Performance with Reduced Features
    y_pred = rf_reduced.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)  # -> 339 features: 1.0000 accuracy
    print(f"Model Accuracy with {len(top_features)} Features: {accuracy:.4f}")

    for col in keep_columns:
        if col not in top_features:
            top_features.append(col)

    # fix pandas double escaped already escaped string uding replace('"""', '"')
    top_features = [col.replace('"""', '"') for col in top_features]

    # Save top features to a file
    print(f"Writing selected columns to file {selcols_file}")
    df = pd.DataFrame(top_features)

    df.to_csv(selcols_file, index=False)


if __name__ == "__main__":
    main()
