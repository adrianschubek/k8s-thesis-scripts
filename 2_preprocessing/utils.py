import pandas as pd
import os


def validate_shapes_and_get_dataframes(prefix_path) -> tuple:
    syscalls = pd.read_csv(prefix_path + "syscalls.csv")
    podlogs = pd.read_csv(prefix_path + "podlogs.csv")
    network = pd.read_csv(prefix_path + "network.csv")
    prom = pd.read_csv("all_prom.csv")

    # print shapes
    print(f"syscalls: {syscalls.shape}")
    print(f"podlogs: {podlogs.shape}")
    print(f"network: {network.shape}")
    print(f"prom: {prom.shape}")

    if not (syscalls.shape[0] == podlogs.shape[0] == network.shape[0] == prom.shape[0]):
        raise Exception("Rows mismatch")

    return syscalls, podlogs, network, prom
