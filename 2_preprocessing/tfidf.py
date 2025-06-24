import pandas as pd
import numpy as np
import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    parser = argparse.ArgumentParser(description="Process captured data into datasets.")
    parser.add_argument(
        "--input",
        type=str,
        help="File path to all_datasets_rf.csv",
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

    df = pd.read_csv(args.input)
    
    # cast Pod-Logs to string
    df["Pod_Logs"] = df["Pod_Logs"].astype(str)
    # replace NaN with empty string
    df["Pod_Logs"] = df["Pod_Logs"].replace(np.nan, "", regex=True)
    
    # print(df["Pod_Logs"].head())

    # Pod-Logs tf-idf
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["Pod_Logs"])
    X = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())
    # drop Pod_Logs column
    df.drop(columns=["Pod_Logs"], inplace=True)
    # merge tfidf columns
    df = pd.concat([df, X], axis=1)
    df.to_csv(os.path.join(DESTROOT, "all_datasets_tfidf.csv"), index=False)
    print("tfidf_all_datasets_rf.csv saved")

    # # also save xlxs
    # df.to_excel(os.path.join(DESTROOT, "all_datasets_tfidf.xlsx"), index=False)
    # print("tfidf_all_datasets_rf.xlsx saved")

if __name__ == "__main__":
    main()
