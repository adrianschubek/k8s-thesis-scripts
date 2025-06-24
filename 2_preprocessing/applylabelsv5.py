# ds v5
# apply correct labels
# RUn this AFTER dataset.py (1s) generated and BEFORE featureselecection.py
# If already feature selceted then just run AFTER dataset.py (10ms...)
import argparse
import os

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="set correct labels. v5+")
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to the dataset. must include Timestamp column",
        required=True,
    )
    parser.add_argument(
        "--timing",
        type=str,
        help="path to the timing file in raw data",
        required=True,
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="resolution of the dataset (1s, 10ms,...)",
        required=True,
    )
    parser.add_argument(
        "--between-as-benign",
        type=lambda x: str(x).lower() == "true",
        required=True,
        help="treat time between attacks as benign true/false",
    )
    args = parser.parse_args()

    # absolute path to the dataset
    in_csv = os.path.abspath(args.dataset)
    out_csv = in_csv.replace(".csv", "_labeled.csv")
    RESOLUTION = args.resolution
    timing_csv = os.path.abspath(args.timing)
    print(f"Reading {in_csv}")
    print(f"Writing to {out_csv}")
    print(f"Resolution: {RESOLUTION}")
    print(f"Reading timing from {timing_csv}")
    print("Treating time between attacks as benign" if args.between_as_benign else "Removing time between attacks")

    timing_df = pd.read_csv(timing_csv, header=None)

    timing_df[0] = timing_df[0].apply(lambda name: name.split("/")[-1])
    timing_df[0] = timing_df[0].apply(lambda name: 0 if name.startswith("benign") else name)
    timing_df[0] = timing_df[0].apply(lambda name: int(name))

    # from ms col 2 to dt col 1
    timing_df[1] = timing_df[1 + 1].apply(lambda x: pd.to_datetime(x, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(RESOLUTION))
    timing_df[1] = timing_df[1].dt.floor(RESOLUTION)

    timing_df[3] = timing_df[3 + 1].apply(lambda x: pd.to_datetime(x, unit="ms", utc=True).replace(year=1900, month=1, day=1).floor(RESOLUTION))
    timing_df[3] = timing_df[3].dt.floor(RESOLUTION)

    # EXPERIEMNT: just keep one attack 8 sample because 3 too large (~30s each). keep first row where timing_df[0] == 8. remove all other rows with 8
    found_8 = False
    for index, row in timing_df.iterrows():
        if row[0] == 8:
            if found_8:
                timing_df.drop(index, inplace=True)
            else:
                found_8 = True
                # col 3 subtract 15 seconds
                timing_df.at[index, 3] = row[3] - pd.Timedelta(seconds=15)
                # col 6 subtract 15000 ms
                timing_df.at[index, 6] = row[6] - 15000

    print(timing_df)
    # group by col 0 sumarize col 6! 
    xxdf = timing_df.groupby(0).agg({6: "sum"}).reset_index()
    # int divide by 10
    xxdf[6] = xxdf[6].apply(lambda x: int(x / 10) * 3)
    print(xxdf)
    exit(111)

    df = pd.read_csv(in_csv)

    # Timestamp to ms
    df["Timestamp"] = df["Timestamp"].apply(lambda x: pd.to_datetime(x, utc=True).replace(year=1900, month=1, day=1).floor(RESOLUTION))
    df["Timestamp"] = df["Timestamp"].dt.floor(RESOLUTION)  # unnecessary ds is already floored

    # set all attack labels to 99 as unprocessed
    df["attack"] = 99

    # code OK
    # for each row in timing_df -> apply label timing[0] if df timestamp is between start timting[1] and end timing[3]
    for index, row in timing_df.iterrows():
        print(f"Applying label {row[0]} from {row[1]}/{row[2]} to {row[3]}/{row[4]}")
        mask = (df["Timestamp"] >= row[1]) & (df["Timestamp"] <= row[3])
        df.loc[mask, "attack"] = row[0]

    if args.between_as_benign:
        # experiment: set all 99 to 0 too. now special benign sceanrios and time between attacks are also labeled as benign
        df.loc[df["attack"] == 99, "attack"] = 0
    else:
        # remove all rows between attacks
        # df = df[df["attack"] != 99]
        print(" Setting between rows to 99 ! Just remove it in the end")

    df.reset_index(drop=True, inplace=True)

    # groupby attack label and count
    # print(df[["Timestamp", "attack", "node_k8s-master-1", "node_k8s-worker-1", "node_k8s-worker-2"]].groupby("attack").count())
    # print(df[["Timestamp", "attack", "node_k8s-master-1", "node_k8s-worker-1", "node_k8s-worker-2"]])

    # drop timestamp
    df.drop(columns=["Timestamp"], inplace=True)

    # sanity check: no NaN values except Pod_Logs
    for col in df.columns:
        if col == "Pod_Logs":
            continue
        assert df[col].notna().all(), f"NaN values found in {col}"

    # save to csv
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")


if __name__ == "__main__":
    main()
