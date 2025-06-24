import pandas as pd
import numpy as np

df1 = "/home/adrian/k8s-thesis/preprocessing/local_datasets_v6/attack1-2025-04-07_11-18-27/dataset_labeled_net2.csv"
# print column name 79 & 83
df = pd.read_csv(df1)
print(df.columns[79])
print(df.columns[83])
exit(9)

df1 = "/home/adrian/k8s-thesis/preprocessing/local_datasets_v6/attack1-2025-04-07_11-18-27/dataset_labeled_net2_rollwindows.csv"
# df1 = "/home/adrian/k8s-thesis/preprocessing/local_datasets_v6/attack1-2025-04-07_11-18-27/dataset_labeled_net2_rollwindows.csv"
# df2 = f"/home/adrian/k8s-thesis/preprocessing/local_datasets_v6gapnotbenign/a_split/{m}/2_0_{mm}.csv"
df1 = pd.read_csv(df1)
# df2 = pd.read_csv(df2)
print(df1)
# print(df2)

print(df1["attack"].value_counts())

# for i, col in enumerate(df1.columns):
#     print(f"{i}: {col}")

exit(0)

# # DFA = "/home/adrian/k8s-thesis/scripts/datasets/v6hicorr/dataset_a.csv.gz"
# # DFB = "/home/adrian/k8s-thesis/scripts/datasets/v6more/dataset_b.csv.gz"
# DFA = "/home/adrian/k8s-thesis/preprocessing/local_datasets_v6/attack1-2025-04-07_11-18-27/dataset_labeled_net2_rollwindows.csv"
# DFB = "/home/adrian/k8s-thesis/preprocessing/local_datasets_v6/normal1-2025-04-07_12-31-15/dataset_labeled_net2_rollwindows.csv"
# dfa = pd.read_csv(DFA)
# dfb = pd.read_csv(DFB)
# print(dfa.head())
# print(dfb.head())
# print(dfa["attack"].value_counts())
# print(dfb["attack"].value_counts())
# exit(99)

# # List column indices ranges by data type
# def list_dtype_ranges(df):
#     # Group columns by dtype
#     dtype_groups = {}

#     for i, (column, dtype) in enumerate(df.dtypes.items()):
#         dtype_str = str(dtype)
#         if dtype_str not in dtype_groups:
#             dtype_groups[dtype_str] = []
#         dtype_groups[dtype_str].append(i)

#     # Print ranges of indices with the same dtype
#     for dtype, indices in dtype_groups.items():
#         ranges = []
#         start = indices[0]
#         prev = indices[0]

#         for i in indices[1:] + [None]:
#             if i != prev + 1:
#                 # End of a consecutive range
#                 if prev == start:
#                     ranges.append(str(start))
#                 else:
#                     ranges.append(f"{start}-{prev}")

#                 if i is not None:
#                     start = i

#             if i is not None:
#                 prev = i

#         print(f"dtype {dtype}: columns {', '.join(ranges)}")

# # Add this after your existing line
# print("\nColumn index ranges by dtype:")
# list_dtype_ranges(dfa)

# # list first 30 columns wiht their idnex from 0
# for i, col in enumerate(dfa.columns[250:]):
#     i = i + 250
#     print(f"{i}: {col}")

# exit(5)
# dfb = pd.read_csv(DFB)
# print(dfa.head())
# print(dfb.head())
# print(dfa["attack"].value_counts())
# print(dfb["attack"].value_counts())

# exit(0)


### Run after final datasets generated to simplify column names. run BEFORE network2
def simplycolumnames(df):
    """
    # remove all " from column names to prevent mismatches due to too many escaped " created in csv by pandas
    # replace "," in column names so csv reader does not split them
    """
    df.columns = [col.replace(",", "|").replace('"', "") for col in df.columns]
    return df


DFA = "/home/adrian/k8s-thesis/preprocessing/local_datasets_v6gapnotbenign/attack1-2025-04-07_11-18-27/dataset_labeled.csv"
DFB = "/home/adrian/k8s-thesis/preprocessing/local_datasets_v6gapnotbenign/normal1-2025-04-07_12-31-15/dataset_labeled.csv"

dfa = pd.read_csv(DFA)
dfa = simplycolumnames(dfa)
print(dfa.head())
dfb = pd.read_csv(DFB)
dfb = simplycolumnames(dfb)
print(dfb.head())


# make sure column order is the same
dfa = dfa[dfb.columns]

# find missing columns from dfa in dfb or vice versa
missing_cols = set(dfa.columns) - set(dfb.columns)
if missing_cols:
    print(f"Missing columns in dfb: {missing_cols}")
    print(f"Missing columns in dfb: {len(missing_cols)}")

missing_cols = set(dfb.columns) - set(dfa.columns)
if missing_cols:
    print(f"Missing columns in dfa: {missing_cols}")
    print(f"Missing columns in dfa: {len(missing_cols)}")


print(dfa["attack"].value_counts())
print(dfb["attack"].value_counts())

# backup existing datasert_labeled.csv to dataset_labeled.csv.bkup
import shutil

shutil.move(DFA, DFA + ".bkup")
shutil.move(DFB, DFB + ".bkup")

# save back to csv dataset_labeled.csv
dfa.to_csv(DFA, index=False)
dfb.to_csv(DFB, index=False)

exit(0)

##############################################
# Balancing
# adf = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v5it3ms10/net2/final_attacks_dataset.csv")
# bdf = pd.read_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v5it3ms10/net2/final_benign_dataset.csv")

### only for sampled
# limit benign 1. shuffle benign then limit

# bdf = bdf.sample(frac=1, random_state=42).reset_index(drop=True)
# bdf = bdf.iloc[:bdf.shape[0] // 6].reset_index(drop=True)

### Do for both net2 and sampled!
# same for attack=8. leave rest untouched

# tmp8_df = adf[adf["attack"] == 8]
# tmp8_df = tmp8_df.sample(frac=1, random_state=42).reset_index(drop=True)
# rows8 = tmp8_df.shape[0]
# tmp8_df = tmp8_df.iloc[:rows8 // 6].reset_index(drop=True)
# adf = adf[adf["attack"] != 8]
# adf = pd.concat([adf, tmp8_df])

# group by attack -> count
# print(adf["attack"].value_counts())
# # print total rows
# print(adf.shape[0])
# print(bdf["attack"].value_counts())

# adf.to_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v5it3ms10/net2_sampled/final_attacks_dataset.csv", index=False)
# bdf.to_csv("/home/adrian/k8s-thesis/preprocessing/local_datasets_v5it3ms10/net2_sampled/final_benign_dataset.csv", index=False)
