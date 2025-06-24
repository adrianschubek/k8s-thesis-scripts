# graphics for ml algos comparison

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

in_folder = "/home/adrian/k8s-thesis/preprocessing/models_v6sampled"
dataset_csv = "/home/adrian/k8s-thesis/preprocessing/local_datasets_v6/attack1-2025-04-07_11-18-27/dataset_labeled_net2_sampled.csv"
out_folder = "/home/adrian/k8s-thesis/preprocessing/models_v6sampled/_graphs"

results = {}
kfold_results = {}


def make_el(name):
    if name not in results:
        results[name] = {"final": {}, "classes": {}}


color_map = {
    "CNN": "red",
    "MLP": "darkred",
    "RF": "blue",
    "DT": "green",
    "KNN": "purple",
    "SVM": "darkorange",
    "NB": "black",
}


def parse_dataset_distribution():
    df = pd.read_csv(dataset_csv)

    # PIE
    plt.figure(figsize=(6, 6))
    df["attack"].value_counts().plot(kind="pie", startangle=90, labels=None, colormap="tab20")  # Move percentage labels closer to center
    plt.ylabel("")
    plt.title("Distribution of Classes in Dataset", fontsize=14)
    plt.axis("equal")
    labels = df["attack"].value_counts().index
    percentages = (df["attack"].value_counts() / len(df) * 100).round(1)
    legend_labels = [f"{label}: {pct}%" for label, pct in zip(labels, percentages)]
    plt.legend(labels=legend_labels, loc="center left", fontsize=12, title="Classes")
    plt.savefig(os.path.join(out_folder, "dataset_distribution.png"), bbox_inches="tight")
    # plt.show()
    plt.close()

    # BAR CHART SAMPLES + show value above each bar
    plt.figure(figsize=(10, 6))
    # get color fromtab20 by index
    colors = plt.get_cmap("tab20", len(df["attack"].value_counts()))(np.arange(len(df["attack"].value_counts())))
    df["attack"].value_counts().plot(kind="bar", color=colors, alpha=0.7)
    plt.title("Distribution of Classes in Dataset", fontsize=14)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, df["attack"].value_counts().max() * 1.1)  # Add some space above the highest bar
    for i, v in enumerate(df["attack"].value_counts()):
        plt.text(i, v + 0.02, str(v), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "dataset_distribution_bar.png"), bbox_inches="tight")
    # plt.show()
    plt.close()


def parse_conf_mat():
    for root, dirs, files in os.walk(in_folder):
        for file in files:
            # skip if last root after / not in file
            # if root.split("/")[-1] not in file:
            #     print(f"Skipping {file} in {root} (root not in file)")
            #     continue
            if file.endswith(".txt") and "kfold" not in file:
                # if "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no" in file:
                model, config = file.replace("skip99_", "").split("-", 1)
                config = config.replace(".txt", "")

                # temp hack for svm
                # if "svm" in model:
                #     config = config.replace("minmax", "zscore")

                # parse scikit confusion matrix:
                #  precision    recall  f1-score   support

                #            0     0.9987    0.9949    0.9968     10325
                #            1     0.9780    1.0000    0.9889       267
                #            2     0.9485    0.9946    0.9710       185
                #            3     0.9840    0.9893    0.9867       187
                #            4     0.9804    1.0000    0.9901       150
                #            5     0.9960    0.9801    0.9880       251
                #            6     0.9941    1.0000    0.9971      1018
                #            7     0.9886    0.9560    0.9721        91
                #            8     0.9990    0.9990    0.9990      1000
                #            9     0.9786    1.0000    0.9892      1004
                #           10     1.0000    1.0000    1.0000       143

                #     accuracy                         0.9955     14621
                #    macro avg     0.9860    0.9922    0.9890     14621
                # weighted avg     0.9956    0.9955    0.9955     14621

                with open(os.path.join(root, file), "r") as f:
                    temp_classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    fit_time = -1
                    eval_time = -1
                    lines = f.readlines()
                    for line in lines:
                        # check if part of class
                        try:
                            class_id = line.split()[0]
                            if class_id.isdigit():
                                class_id = int(class_id)
                                # check if class id is in range
                                temp_classes[class_id] = {
                                    "precision": float(line.split()[1]),
                                    "recall": float(line.split()[2]),
                                    "f1-score": float(line.split()[3]),
                                    "support": int(line.split()[4]),
                                }
                        except IndexError:
                            pass

                        if "Balanced accuracy" in line:
                            balanced_accuracy = float(line.split()[3])
                            print(f"Balanced accuracy: {balanced_accuracy}")
                            continue
                        if "accuracy" in line and "Balanced" not in line:
                            accuracy = float(line.split()[1])
                            support = int(line.split()[2])
                            print(f"Accuracy f1: {accuracy}")
                            continue
                        if "macro avg" in line:
                            macro_avg_prec = float(line.split()[2])
                            macro_avg_recall = float(line.split()[3])
                            macro_avg_f1 = float(line.split()[4])
                            # ValueError: not enough values to unpack (expected 3, got 2)
                            print(f"Macro avg f1: {macro_avg_f1}")
                            continue
                        if "weighted avg" in line:
                            weighted_avg = float(line.split()[4])
                            print(f"Weighted avg f1: {weighted_avg}")
                            continue
                        if "fit_time" in line:
                            print(model, line)
                            fit_time = int(line.split()[1])
                            continue
                        if "eval_time" in line:
                            eval_time = int(line.split()[1])
                            continue

                # KNN by defaulot same params EXCEPT no windows!!
                # svm default wihtout windows so replace window_0 with window_2+5+10+50+100
                if model == "knn" and "window_0" in config:
                    print(f"Replacing {config} with window_2+5+10+50+100")
                    config = config.replace("window_0", "window_2+5+10+50+100")
                elif model == "knn" and "window_2+5+10+50+100" in config:
                    print(f"Replacing {config} with window_0")
                    config = config.replace("window_2+5+10+50+100", "window_0")

                make_el(config)
                results[config]["final"][model.upper()] = {
                    "accuracy": accuracy,
                    "macro_avg_prec": macro_avg_prec,
                    "macro_avg_recall": macro_avg_recall,
                    "macro_f1_avg": macro_avg_f1,
                    "weighted_f1_avg": weighted_avg,
                    "balanced_accuracy": balanced_accuracy,
                    "fit_time": fit_time,
                    "eval_time": eval_time,
                    "support": support,
                }
                results[config]["classes"][model.upper()] = temp_classes

    print(results.keys())

    backup = results.copy()
    configs = [k for k in results.keys()]
    print("############# FINAL RESULTS #############")
    # # FINAL RESULTS
    if True:
        eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg"]
        labels = ["Accuracy", "Average Precision (macro)", "Average Recall (macro)", "Average F1 Score (macro)", "Average Weighted F1 Score"]
        for config in configs:
            if "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no" in config and "exp" not in config:
                for metric in eval_metrics:
                    label = labels[eval_metrics.index(metric)]
                    df = pd.DataFrame.from_dict(results[config]["final"], orient="index")
                    df = df.sort_values(by=metric, ascending=False)
                    print(df)

                    # Plot vertical bar chart
                    plt.figure(figsize=(6, 6))  # Adjust height to reduce vertical space
                    for i, v in enumerate(df[metric]):
                        plt.text(i, v - 0.04, str(round(v, 4)), ha="center", va="bottom", color="white")

                    # Use color map to color bars
                    colors = [color_map.get(model, "gray") for model in df.index]
                    plt.bar(df.index, df[metric], color=colors, alpha=0.7)
                    plt.xticks(rotation=0, ha="center")
                    # Adjust y-axis limits to fit data more closely
                    min_val = df[metric].min()
                    plt.ylim(max(0, min_val - 0.05), 1)  # Start y-axis closer to the minimum value
                    plt.title(label)
                    plt.xlabel("Model")
                    plt.ylabel(label)
                    plt.tight_layout(pad=0.5)  # Reduce padding
                    plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to minimize space
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "_final.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                break

    print("############# BINARY RESULTS #############")
    # BINARY RESULTS
    if True:
        eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg", "balanced_accuracy"]
        labels = [
            "Accuracy",
            "Average Precision (macro)",
            "Average Recall (macro)",
            "Average F1 Score (macro)",
            "Average Weighted F1 Score",
            "Balanced Accuracy",
        ]
        for config in configs:
            if "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no-exp_binary" in config:
                for metric in eval_metrics:
                    label = labels[eval_metrics.index(metric)]
                    df = pd.DataFrame.from_dict(results[config]["final"], orient="index")
                    df = df.sort_values(by=metric, ascending=False)
                    print(df)

                    # Plot vertical bar chart
                    plt.figure(figsize=(6, 6))  # Adjust height to reduce vertical space
                    for i, v in enumerate(df[metric]):
                        plt.text(i, v - 0.04, str(round(v, 4)), ha="center", va="bottom", color="white")

                    # Use color map to color bars
                    colors = [color_map.get(model, "gray") for model in df.index]
                    plt.bar(df.index, df[metric], color=colors, alpha=0.7)
                    plt.xticks(rotation=0, ha="center")
                    # Adjust y-axis limits to fit data more closely
                    min_val = df[metric].min()
                    plt.ylim(max(0, min_val - 0.05), 1)  # Start y-axis closer to the minimum value
                    plt.title(label)
                    plt.xlabel("Model")
                    plt.ylabel(label)
                    plt.tight_layout(pad=0.5)  # Reduce padding
                    plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to minimize space
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "_binaryfinal.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                break

    print("############# PER CLASS #############")
    # PER CLASS: PRECISION, RECALL, F1-SCORE different models
    if True:
        eval_metrics = ["precision", "recall", "f1-score"]
        for config in configs:
            if "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no" in config and "exp" not in config:
                for metric in eval_metrics:
                    label = metric.capitalize()
                    models = [model for model in results[config]["classes"].keys()]

                    data = {}
                    for model in models:
                        data[model] = []
                        for i in range(11):
                            data[model].append(float(results[config]["classes"][model][i][metric]))

                    # Create DataFrame with classes as index
                    classes = list(range(11))
                    df = pd.DataFrame(data, index=classes)
                    print(df)

                    # Create a figure with 11 subplots in a 4x3 grid
                    fig, axes = plt.subplots(4, 3, figsize=(10, 15), sharey=True)
                    axes = axes.flatten()  # Flatten to easily index the subplots

                    # Plot a bar chart for each class
                    for i in range(11):
                        ax = axes[i]
                        class_data = df.loc[i]  # Get data for class i
                        # sort bars by value
                        class_data = class_data.sort_values(ascending=False)
                        class_data.plot(kind="bar", ax=ax, color=[color_map.get(model, "gray") for model in class_data.index], alpha=0.7)
                        ax.set_title(f"Class {i}")
                        ax.set_xlabel("Models")
                        ax.set_ylabel(label)
                        ax.legend().remove()  # Remove individual legends

                    # Add a common legend
                    # handles, labels = ax.get_legend_handles_labels()
                    # fig.legend(handles, labels, title="Models", loc="upper right")

                    # Turn off the unused subplot (12th position)
                    axes[-1].axis("off")

                    # Adjust layout and display
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "_perclass.png"))
                    # plt.show()
                    plt.close()
                break

    print("############# ROLLING WINDOWS vs NON-ROLLING WINDOWS #############")
    # MODELS: Rolling Windows vs Non-Rolling Windows: 7 bars with 2 subbars Roll/NonRoll
    if True:
        eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg"]
        labels = ["Accuracy", "Average Precision (macro)", "Average Recall (macro)", "Average F1 Score (macro)", "Average Weighted F1 Score"]
        logs_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no"
        nologs_config = "rollwindow_0-norm_zscore-logs_tfidf-oversmpl_no"
        if logs_config in results.keys() and nologs_config in results.keys():
            for metric in eval_metrics:
                label = labels[eval_metrics.index(metric)]
                onlynet_df = pd.DataFrame.from_dict(results[logs_config]["final"], orient="index")
                onlynet_df = onlynet_df.sort_values(by=metric, ascending=False)
                print(onlynet_df)

                eval_df = pd.DataFrame.from_dict(results[nologs_config]["final"], orient="index")
                # sort same order as window_df
                eval_df = eval_df.reindex(onlynet_df.index)
                print(eval_df)

                # switch KNN 0 <-> 2 5 10 50 100 rows! ## HACK
                # switch rows onlynet <> eval
                temp = onlynet_df.loc["KNN"].copy()
                onlynet_df.loc["KNN"] = eval_df.loc["KNN"]
                eval_df.loc["KNN"] = temp

                # plot Rolling Windows vs Non-Rolling Windows: 7 bars with 2 subbars Roll/NonRoll

                # Evaluation metrics and labels
                eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg"]
                labels = ["Accuracy", "Average Precision (macro)", "Average Recall (macro)", "Average F1 Score (macro)", "Average Weighted F1 Score"]

                models = [str(model) for model in onlynet_df.index]
                n_models = len(models)
                bar_width = 0.35  # Width of each bar
                print(models)

                for metric, label in zip(eval_metrics, labels):
                    # Get data for the metric
                    roll_values = onlynet_df[metric]
                    nonroll_values = eval_df[metric]

                    # Create a new figure
                    plt.figure(figsize=(10, 6))

                    # Set up bar positions
                    x = np.arange(n_models)

                    # Plot bars
                    plt.bar(x - bar_width / 2, roll_values, bar_width, label="Rolling Windows", color="blue", alpha=0.7)
                    plt.bar(x + bar_width / 2, nonroll_values, bar_width, label="No Rolling Windows", color="black", alpha=0.7)

                    # Customize plot
                    plt.title(f"Rolling Windows vs No Rolling Windows: {label}")
                    plt.xlabel("Models")
                    plt.ylabel(label)
                    plt.xticks(x, models)
                    plt.ylim(0, 1)  # Set y-axis limit for better comparison
                    plt.legend()
                    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

                    # Save the plot
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "_rollVSnonroll.png"))
                    # plt.show()
                    plt.close()

                # 2. Plot Percentage Change
                for metric, label in zip(eval_metrics, labels):
                    # Calculate percentage change
                    roll_values = onlynet_df[metric]
                    nonroll_values = eval_df[metric]
                    # Avoid division by zero
                    percentage_change = ((roll_values - nonroll_values) / nonroll_values.replace(0, np.nan)) * 100

                    # Create a new figure for percentage change
                    plt.figure(figsize=(10, 6))

                    # Plot bars
                    x = np.arange(n_models)
                    # Assign colors based on percentage change (green for positive, red for negative)
                    colors = ["green" if change > 0 else "red" for change in percentage_change]
                    plt.bar(x, percentage_change, color=colors, alpha=0.7)

                    # Customize plot
                    plt.title(f"Percentage Change (Rolling Windows vs No Rolling Windows): {label}")
                    plt.xlabel("Models")
                    plt.ylabel("Percentage Change (%)")
                    plt.xticks(x, models)
                    # Set y-axis limits dynamically based on data, with some padding
                    max_abs_change = max(abs(percentage_change.max()), abs(percentage_change.min()))
                    plt.ylim(-max_abs_change * 1.1, max_abs_change * 1.1)
                    plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Add zero line
                    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "_rollVSnonrollchanges.png"))
                    # plt.show()
                    plt.close()

                break  # break outer for. only one it needed

    print("############# TF-IDF vs No TF-IDF #############")
    # MODELS: TFDF vs No TFIDF
    if True:
        eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg"]
        labels = ["Accuracy", "Average Precision (macro)", "Average Recall (macro)", "Average F1 Score (macro)", "Average Weighted F1 Score"]
        window_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no"
        nonwindow_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_no-oversmpl_no"
        if window_config in results.keys() and nonwindow_config in results.keys():
            for metric in eval_metrics:
                label = labels[eval_metrics.index(metric)]
                onlynet_df = pd.DataFrame.from_dict(results[window_config]["final"], orient="index")
                onlynet_df = onlynet_df.sort_values(by=metric, ascending=False)
                print(onlynet_df)

                eval_df = pd.DataFrame.from_dict(results[nonwindow_config]["final"], orient="index")
                # force same order as window_df
                eval_df = eval_df.reindex(onlynet_df.index)
                print(eval_df)

                # plot Rolling Windows vs Non-Rolling Windows: 7 bars with 2 subbars Roll/NonRoll

                # Evaluation metrics and labels
                eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg"]
                labels = ["Accuracy", "Average Precision (macro)", "Average Recall (macro)", "Average F1 Score (macro)", "Average Weighted F1 Score"]

                # force model order
                models = [str(model) for model in onlynet_df.index]
                n_models = len(models)
                bar_width = 0.35  # Width of each bar
                print(models)

                for metric, label in zip(eval_metrics, labels):
                    # Get data for the metric
                    roll_values = onlynet_df[metric]
                    nonroll_values = eval_df[metric]

                    # Create a new figure
                    plt.figure(figsize=(10, 6))

                    # Set up bar positions
                    x = np.arange(n_models)

                    # Plot bars
                    plt.bar(x - bar_width / 2, roll_values, bar_width, label="TF-IDF Pod Logs", color="blue", alpha=0.7)
                    plt.bar(x + bar_width / 2, nonroll_values, bar_width, label="No Pod Logs", color="black", alpha=0.7)

                    # Customize plot
                    plt.title(f"TF-IDF Pod Logs vs No Pod Logs: {label}")
                    plt.xlabel("Models")
                    plt.ylabel(label)
                    plt.xticks(x, models)
                    plt.ylim(0, 1)  # Set y-axis limit for better comparison
                    plt.legend()
                    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

                    # Save the plot
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "podVSnopod.png"))
                    # plt.show()
                    plt.close()

                # 2. Plot Percentage Change
                for metric, label in zip(eval_metrics, labels):
                    # Calculate percentage change
                    roll_values = onlynet_df[metric]
                    nonroll_values = eval_df[metric]
                    # Avoid division by zero
                    percentage_change = ((roll_values - nonroll_values) / nonroll_values.replace(0, np.nan)) * 100

                    # print(percentage_change)
                    # exit(1)

                    # Create a new figure for percentage change
                    plt.figure(figsize=(10, 6))

                    # Plot bars
                    x = np.arange(n_models)
                    # Assign colors based on percentage change (green for positive, red for negative)
                    colors = ["green" if change > 0 else "red" for change in percentage_change]
                    plt.bar(x, percentage_change, color=colors, alpha=0.7)

                    # Customize plot
                    plt.title(f"Percentage Change (Pod Logs vs No Pod Logs): {label}")
                    plt.xlabel("Models")
                    plt.ylabel("Percentage Change (%)")
                    plt.xticks(x, models)
                    # Set y-axis limits dynamically based on data, with some padding
                    max_abs_change = max(abs(percentage_change.max()), abs(percentage_change.min()))
                    plt.ylim(-max_abs_change * 1.1, max_abs_change * 1.1)
                    plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Add zero line
                    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "podVSnopodchanges.png"))
                    # plt.show()
                    plt.close()

                break  # break outer for. only one it needed

    print("############# CNN PER CLASS #############")
    # CNN per-class -> 4 bar full,m1,w1,w2
    if True:
        full_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no"
        m1_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no-exp_master1"
        w1_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no-exp_worker1"
        w2_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no-exp_worker2"
        if full_config in results.keys() and m1_config in results.keys() and w1_config in results.keys() and w2_config in results.keys():
            eval_metrics = ["precision", "recall", "f1-score"]
            for metric in eval_metrics:
                label = metric.capitalize()
                model = "CNN"

                data = {
                    "full": [],
                    "m1": [],
                    "w1": [],
                    "w2": [],
                }
                for i in range(11):
                    data["full"].append(float(results[full_config]["classes"][model][i][metric]))
                    data["m1"].append(float(results[m1_config]["classes"][model][i][metric]))
                    data["w1"].append(float(results[w1_config]["classes"][model][i][metric]))
                    data["w2"].append(float(results[w2_config]["classes"][model][i][metric]))

                # Create DataFrame with classes as index
                classes = list(range(11))
                df = pd.DataFrame(data, index=classes)
                print(df)

                # Create a figure with 11 subplots in a 4x3 grid
                fig, axes = plt.subplots(4, 3, figsize=(6, 10), sharey=True)
                axes = axes.flatten()  # Flatten to easily index the subplots

                # Plot 4 bars (full, m1, w1, w2) in each subplot for each class
                for i in range(11):  # Loop through each class (0 to 10)
                    ax = axes[i]  # Select the subplot for class i
                    # Extract values for the current class
                    values = [data["full"][i], data["m1"][i], data["w1"][i], data["w2"][i]]
                    # Define the x positions for the bars
                    x = range(4)
                    # Plot bars
                    ax.bar(x, values, tick_label=["All", "M1", "W1", "W2"], color=["blue", "black", "gray", "darkgray"], alpha=0.7)
                    # Set title for the subplot
                    ax.set_title(f"Class {i}")
                    # Set x-axis label (optional, can be omitted since tick labels are set)
                    ax.set_xlabel("Nodes")
                    # Set y-axis label (only on the leftmost subplots for clarity)
                    if i % 3 == 0:
                        ax.set_ylabel(label)

                    ax.set_ylim(0.6, 1)  # Set y-axis limit for better comparison
                    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

                # Turn off the unused subplot (12th position)
                axes[-1].axis("off")

                # Adjust layout and display
                plt.tight_layout()
                plt.savefig(os.path.join(out_folder, metric + "_cnnPerNode.png"))
                # plt.show()
                plt.close()

            # 2. Plot Percentage Change
            for metric in eval_metrics:
                label = metric.capitalize()
                model = "CNN"

                data = {
                    "full": [],
                    "m1": [],
                    "w1": [],
                    "w2": [],
                }
                for i in range(11):
                    data["full"].append(float(results[full_config]["classes"][model][i][metric]))
                    data["m1"].append(float(results[m1_config]["classes"][model][i][metric]))
                    data["w1"].append(float(results[w1_config]["classes"][model][i][metric]))
                    data["w2"].append(float(results[w2_config]["classes"][model][i][metric]))

                # Create DataFrame with classes as index
                classes = list(range(11))
                df = pd.DataFrame(data, index=classes)
                print(df)

                # Create a figure with 11 subplots in a 4x3 grid
                fig, axes = plt.subplots(4, 3, figsize=(6, 10), sharey=True)
                axes = axes.flatten()
                # Plot 4 bars (full, m1, w1, w2) in each subplot for each class
                for i in range(11):  # Loop through each class (0 to 10)
                    ax = axes[i]  # Select the subplot for class i
                    # Extract values for the current class
                    values = [data["full"][i], data["m1"][i], data["w1"][i], data["w2"][i]]
                    # Calculate percentage change
                    percentage_change = [((v - data["full"][i]) / data["full"][i]) * 100 for v in values]
                    # Define the x positions for the bars
                    x = range(4)
                    # Assign colors based on percentage change (green for positive, red for negative)
                    colors = ["green" if change > 0 else "red" for change in percentage_change]
                    # Plot bars
                    ax.bar(x, percentage_change, tick_label=["All", "M1", "W1", "W2"], color=colors, alpha=0.7)
                    # Set title for the subplot
                    ax.set_title(f"Class {i}")
                    # Set x-axis label (optional, can be omitted since tick labels are set)
                    ax.set_xlabel("Nodes")
                    # Set y-axis label (only on the leftmost subplots for clarity)
                    if i % 3 == 0:
                        ax.set_ylabel(label)
                    ax.set_ylim(-25, 25)  # Set y-axis limit for better comparison
                    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

                # Turn off the unused subplot (12th position)
                axes[-1].axis("off")
                # Adjust layout and display

                plt.tight_layout()
                plt.savefig(os.path.join(out_folder, metric + "_cnnPerNodeChanges.png"))
                # plt.show()
                plt.close()

    print("############# TRAING vs INFERENCE TIME #############")
    # MODELS: Train Time vs Inference Time
    if True:
        config = "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no"
        if config in results.keys():
            df = pd.DataFrame.from_dict(results[config]["final"], orient="index")
            df = df.sort_values(by="fit_time", ascending=False)
            print(df)
            # create vertical bar chart with 7 bars each with 2 subbars: fit_time and eval_time

            def format_time(seconds):
                # ms to seconds
                # seconds = ms / 1000
                if seconds >= 60:  # Use minutes if >= 60 seconds
                    minutes = seconds / 60
                    return f"{minutes:.1f}m"
                # if smaller than 1 second, show in milliseconds
                elif seconds < 1:
                    milliseconds = seconds * 1000
                    return f"{milliseconds:.0f}ms"
                return f"{seconds:.1f}s"

            ### 1. by train_time
            # Data for plotting (fit_time and eval_time)
            models = df.index
            fit_times = df["fit_time"] / 1000_000_000  # seconds

            # Set up the bar chart
            x = range(len(models))  # Position for each model
            bar_width = 0.7  # Width of each bar
            fig, ax = plt.subplots(figsize=(6, 6))
            # Plot fit_time and eval_time bars
            fit_bars = ax.bar(x, fit_times, bar_width, label="Fit Time", color=[color_map[model] for model in models], alpha=0.7)

            for bar, time in zip(fit_bars, fit_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height * 0.85, format_time(time), ha="center", va="center", color="white", fontsize=10, rotation=0)

            ax.set_xlabel("Models")
            ax.set_ylabel("Time in seconds")
            ax.set_title("Training Latency by Model")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=0)
            ax.set_yscale("log")

            plt.tight_layout()
            plt.savefig(os.path.join(out_folder, config + "_fit_time.png"))
            # plt.show()
            plt.close()

            ### 2. by eval_time
            # sort df by eval_time
            df = df.sort_values(by="eval_time", ascending=False)
            models = df.index
            eval_times = df["eval_time"] / 1000_000_000  # seconds

            x = range(len(models))  # Position for each model
            fig, ax = plt.subplots(figsize=(6, 6))
            # Plot fit_time and eval_time bars
            eval_bars = ax.bar(x, eval_times, bar_width, label="Eval Time", color=[color_map[model] for model in models], alpha=0.7)

            for bar, time in zip(eval_bars, eval_times):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 0.8,
                    format_time(time),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    rotation=0,
                )

            ax.set_xlabel("Models")
            ax.set_ylabel("Time in seconds")
            ax.set_title("Prediction Latency by Model")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=0)
            ax.set_yscale("log")

            plt.tight_layout()
            plt.savefig(os.path.join(out_folder, config + "_pred_time.png"))
            # plt.show()
            plt.close()

            # format thoudannds as K
            def format_thousands(x):
                if x >= 1_000_000:
                    return f"{x / 1_000_000:.1f}M"
                elif x >= 1_000:
                    return f"{x / 1_000:.1f}K"
                else:
                    return str(int(x))

            ### 3. by eval_time by samples per second
            df["samples_per_second"] = df["support"] / (df["eval_time"] / 1_000_000_000)  # samples per second
            df = df.sort_values(by="samples_per_second", ascending=False)
            print(df)
            models = df.index

            x = range(len(models))  # Position for each model
            fig, ax = plt.subplots(figsize=(6, 6))
            # Plot fit_time and eval_time bars
            sps_bars = ax.bar(x, df["samples_per_second"], bar_width, label="Samples per Second", color=[color_map[model] for model in models], alpha=0.7)
            for bar, speed in zip(sps_bars, df["samples_per_second"]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 0.8,
                    f"{format_thousands(speed)}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    rotation=0,
                )
            ax.set_xlabel("Models")
            ax.set_ylabel("Samples per second")
            ax.set_title("Prediction Throughput by Model")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=0)
            ax.set_yscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(out_folder, config + "_samples_per_second.png"))
            # plt.show()
            plt.close()

    print("############# ZSCORE vs MINMAX #############")
    # MODELS: zscore vs minmax.
    if True:
        eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg"]
        labels = ["Accuracy", "Average Precision (macro)", "Average Recall (macro)", "Average F1 Score (macro)", "Average Weighted F1 Score"]
        logs_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no"
        nologs_config = "rollwindow_2+5+10+50+100-norm_minmax-logs_tfidf-oversmpl_no"
        if logs_config in results.keys() and nologs_config in results.keys():
            for metric in eval_metrics:
                label = labels[eval_metrics.index(metric)]
                onlynet_df = pd.DataFrame.from_dict(results[logs_config]["final"], orient="index")
                onlynet_df = onlynet_df.sort_values(by=metric, ascending=False)
                print(onlynet_df)

                eval_df = pd.DataFrame.from_dict(results[nologs_config]["final"], orient="index")
                # sort same order as window_df
                eval_df = eval_df.reindex(onlynet_df.index)
                print(eval_df)

                # plot Rolling Windows vs Non-Rolling Windows: 7 bars with 2 subbars Roll/NonRoll

                # Evaluation metrics and labels
                eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg"]
                labels = ["Accuracy", "Average Precision (macro)", "Average Recall (macro)", "Average F1 Score (macro)", "Average Weighted F1 Score"]

                models = [str(model) for model in onlynet_df.index]
                n_models = len(models)
                bar_width = 0.35  # Width of each bar
                print(models)

                for metric, label in zip(eval_metrics, labels):
                    # Get data for the metric
                    roll_values = onlynet_df[metric]
                    nonroll_values = eval_df[metric]

                    # Create a new figure
                    plt.figure(figsize=(10, 6))

                    # Set up bar positions
                    x = np.arange(n_models)

                    # Plot bars
                    plt.bar(x - bar_width / 2, roll_values, bar_width, label="Z-Score", color="blue", alpha=0.7)
                    plt.bar(x + bar_width / 2, nonroll_values, bar_width, label="Min-Max", color="teal", alpha=0.7)

                    # Customize plot
                    plt.title(f"Z-Score vs Min-Max: {label}")
                    plt.xlabel("Models")
                    plt.ylabel(label)
                    plt.xticks(x, models)
                    plt.ylim(0, 1)  # Set y-axis limit for better comparison
                    plt.legend()
                    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

                    # Save the plot
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "_zscoreVSminmax.png"))
                    # plt.show()
                    plt.close()

                # 2. Plot Percentage Change
                for metric, label in zip(eval_metrics, labels):
                    # Calculate percentage change
                    roll_values = onlynet_df[metric]
                    nonroll_values = eval_df[metric]
                    # Avoid division by zero
                    percentage_change = ((roll_values - nonroll_values) / nonroll_values.replace(0, np.nan)) * 100

                    # Create a new figure for percentage change
                    plt.figure(figsize=(10, 6))

                    # Plot bars
                    x = np.arange(n_models)
                    # Assign colors based on percentage change (green for positive, red for negative)
                    colors = ["green" if change > 0 else "red" for change in percentage_change]
                    plt.bar(x, percentage_change, color=colors, alpha=0.7)

                    # Customize plot
                    plt.title(f"Percentage Change (Z-Score vs Min-Max): {label}")
                    plt.xlabel("Models")
                    plt.ylabel("Percentage Change (%)")
                    plt.xticks(x, models)
                    # Set y-axis limits dynamically based on data, with some padding
                    max_abs_change = max(abs(percentage_change.max()), abs(percentage_change.min()))
                    plt.ylim(-max_abs_change * 1.1, max_abs_change * 1.1)
                    plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Add zero line
                    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
                    plt.savefig(os.path.join(out_folder, config + "_" + metric + "_zscoreVSminmaxchanges.png"))
                    # plt.show()
                    plt.close()

                break  # break outer for. only one it needed

    # Better:
    print("############# FULL vs ONLYNET vs ONLYSYS vs ONLYPROM #############")
    if True:
        eval_metrics = ["accuracy", "macro_avg_prec", "macro_avg_recall", "macro_f1_avg", "weighted_f1_avg"]
        labels = ["Accuracy", "Average Precision (macro)", "Average Recall (macro)", "Average F1 Score (macro)", "Average Weighted F1 Score"]
        full_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_no-oversmpl_no"
        onlynet_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_no-oversmpl_no-exp_onlynet"
        onlysys_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_no-oversmpl_no-exp_onlysys"
        onlyprom_config = "rollwindow_2+5+10+50+100-norm_zscore-logs_no-oversmpl_no-exp_onlyprom"
        if onlynet_config in results.keys() and onlysys_config in results.keys() and onlyprom_config in results.keys():
            for metric in eval_metrics:
                label = labels[eval_metrics.index(metric)]

                full_df = pd.DataFrame.from_dict(results[full_config]["final"], orient="index")
                full_df = full_df.sort_values(by=metric, ascending=False)
                print(full_df)

                onlynet_df = pd.DataFrame.from_dict(results[onlynet_config]["final"], orient="index")
                onlynet_df = onlynet_df.sort_values(by=metric, ascending=False)
                print(onlynet_df)

                onlysys_df = pd.DataFrame.from_dict(results[onlysys_config]["final"], orient="index")
                onlysys_df = onlysys_df.sort_values(by=metric, ascending=False)
                print(onlysys_df)

                onlyprom_df = pd.DataFrame.from_dict(results[onlyprom_config]["final"], orient="index")
                onlyprom_df = onlyprom_df.sort_values(by=metric, ascending=False)
                print(onlyprom_df)

                # remove all models except SVM
                # onlynet_df = onlynet_df[onlynet_df.index.str.contains("SVM")]
                # onlysys_df = onlysys_df[onlysys_df.index.str.contains("SVM")]
                # onlyprom_df = onlyprom_df[onlyprom_df.index.str.contains("SVM")]
                # full_df = full_df[full_df.index.str.contains("SVM")]

                # Ensure all dataframes have the same model order (use full_df's order)
                model_order = full_df.index
                onlynet_df = onlynet_df.loc[model_order]
                onlysys_df = onlysys_df.loc[model_order]
                onlyprom_df = onlyprom_df.loc[model_order]

                # Prepare data for plotting
                data = {
                    "Full": full_df[metric].values,
                    "OnlyNet": onlynet_df[metric].values,
                    "OnlySys": onlysys_df[metric].values,
                    "OnlyProm": onlyprom_df[metric].values,
                }

                # Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                bar_width = 0.2  # Width of each sub-bar
                index = np.arange(len(model_order))  # X-axis positions for the 7 models

                # Plot each configuration's bars
                plt.bar(index, data["Full"], bar_width, label="All features", color="black", alpha=0.7)
                plt.bar(index + bar_width, data["OnlyNet"], bar_width, label="Network only", color="blue", alpha=0.7)
                plt.bar(index + 2 * bar_width, data["OnlySys"], bar_width, label="Syscalls only", color="red", alpha=0.7)
                plt.bar(index + 3 * bar_width, data["OnlyProm"], bar_width, label="Prometheus only", color="darkgreen", alpha=0.7)

                # grid
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                # Customize the plot
                plt.xlabel("Models")
                plt.ylabel(label)
                plt.title(f"{label} comparison across individual features")
                plt.xticks(index + 1.5 * bar_width, model_order, rotation=0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_folder, config + "_" + metric + "_fullVSonly.png"))
                # plt.show()
                plt.close()


def parse_kfold():
    for root, dirs, files in os.walk(in_folder):
        for file in files:
            # skip if last root after / not in file
            # if root.split("/")[-1] not in file:
            #     print(f"Skipping {file} in {root} (root not in file)")
            #     continue
            if file.endswith(".txt") and "kfoldcv5_cv" in file:
                # if "rollwindow_2+5+10+50+100-norm_zscore-logs_tfidf-oversmpl_no" in file:
                print(file)
                model, config = file.replace("skip99_", "").split("-", 1)
                config = config.replace(".txt", "")

                # temp hack for svm
                # if "svm" in model:
                #     config = config.replace("minmax", "zscore")

                # Stratified Cross-Validation Summary:
                # Average Accuracy: 0.9991 ± 0.0001
                # Average Balanced Accuracy: 0.9990 ± 0.0008
                # Average Precision: 0.9953 ± 0.0005
                # Average Recall: 0.9990 ± 0.0008
                # Average F1 Score: 0.9972 ± 0.0004

                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Average Accuracy" in line:
                            accuracy = float(line.split()[2])
                            accuracy_std = float(line.split()[4])
                            print(f"Accuracy f1: {accuracy} ± {accuracy_std}")
                            continue
                        if "Average Balanced Accuracy" in line:
                            balanced_accuracy = float(line.split()[3])
                            balanced_accuracy_std = float(line.split()[5])
                            print(f"Balanced Accuracy f1: {balanced_accuracy} ± {balanced_accuracy_std}")
                            continue
                        if "Average Precision" in line:
                            precision = float(line.split()[2])
                            precision_std = float(line.split()[4])
                            print(f"Precision f1: {precision} ± {precision_std}")
                            continue
                        if "Average Recall" in line:
                            recall = float(line.split()[2])
                            recall_std = float(line.split()[4])
                            print(f"Recall f1: {recall} ± {recall_std}")
                            continue
                        if "Average F1 Score" in line:
                            f1 = float(line.split()[3])
                            f1_std = float(line.split()[5])
                            print(f"F1 Score f1: {f1} ± {f1_std}")
                            continue

                # svm default wihtout windows so replace window_0 with window_2+5+10+50+100
                if model == "knn" and "window_0" in config:
                    config = config.replace("window_0", "window_2+5+10+50+100")
                elif model == "knn" and "window_2+5+10+50+100" in config:
                    config = config.replace("window_2+5+10+50+100", "window_0")

                kfold_results[config] = {} if config not in kfold_results else kfold_results[config]
                kfold_results[config][model.upper()] = {
                    "accuracy": accuracy,
                    "accuracy_std": accuracy_std,
                    "balanced_accuracy": balanced_accuracy,
                    "balanced_accuracy_std": balanced_accuracy_std,
                    "precision": precision,
                    "precision_std": precision_std,
                    "recall": recall,
                    "recall_std": recall_std,
                    "f1": f1,
                    "f1_std": f1_std,
                }

    print(kfold_results)
    print("############# K-FOLD RESULTS #############")
    for config in kfold_results.keys():
        for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]:
            label = metric.replace("_", " ").capitalize()
            df = pd.DataFrame.from_dict(kfold_results[config], orient="index")
            df = df.sort_values(by=metric, ascending=False)
            print(df)

            # plot vertical bar chart: scale y axis to start at 0.5
            plt.figure(figsize=(6, 6))
            # show value insdie center of each bar
            for i, v in enumerate(df[metric]):
                plt.text(i, v - 0.03, str(round(v, 4)), ha="center", va="bottom", color="white")
                # plt.text(i, v - 0.05, " ±" + str(round(df[metric + "_std"].iloc[i], 4)), ha="center", va="bottom", color="white")

            # use color map to color bars
            colors = [color_map.get(model, "gray") for model in df.index]
            # make rrorbar have horizontal lines at ends
            plt.errorbar(df.index, df[metric], yerr=df[metric + "_std"], fmt=",", color="black", capsize=5, capthick=2)
            plt.bar(df.index, df[metric], color=colors, alpha=0.7)
            plt.xticks(rotation=0, ha="center")
            min_val = df[metric].min()
            plt.ylim(max(0, min_val - 0.05), 1)
            plt.title("Mean " + label + " 5-Fold Cross Validation")  # mean=average
            plt.xlabel("Model")
            plt.ylabel(label)
            plt.tight_layout()
            plt.savefig(os.path.join(out_folder, config + "_" + metric + "_kfold.png"))
            # plt.show()
            plt.close()


def main():
    os.makedirs(out_folder, exist_ok=True)
    # parse_dataset_distribution()
    parse_conf_mat()
    # parse_kfold()


if __name__ == "__main__":
    main()
