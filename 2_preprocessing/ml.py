# keras
# precision_recall_fscore_support sklearn.metrics
# code https://github.com/a18499/KubAnomaly_DataSet/blob/master/KubAnomaly_Paper.py#L14
# code https://de.wikipedia.org/wiki/Keras https://keras.io/examples/vision/mnist_convnet/

import math
import os
import argparse
import pandas as pd
from sklearn import clone
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
import numpy as np
from contextlib import redirect_stdout
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from time import time, time_ns

from sklearn.tree import DecisionTreeClassifier
from podlogs import keywords_list
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import gc
from pickle import dump, load
from datetime import datetime

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
import keras_tuner as kt


def to_categorical(x, num_classes=None):
    return np.eye(num_classes)[x]


def make_rolling_window(original_df: pd.DataFrame, rolling_window: int):
    # Create a copy of the dataframe to avoid modifying the original
    df = original_df.copy()

    num_rows = df.shape[0]
    # Assign time windows (3 rows per window)
    df["time"] = np.repeat(np.arange(0, num_rows // 3) * 10, 3)

    # Assign a row index within each time window (0,1,2 repeated)
    df["row_within_window"] = df.groupby("time").cumcount()

    # Save the original index for later alignment
    original_index = df.index.copy()

    # Set hierarchical index (time, row_within_window) to maintain row structure
    # df = df.set_index(["time", "row_within_window"])
    df = df.set_index(["attack", "time", "row_within_window"])

    ### select rolling window columns
    # from featureselection import keep_columns
    # base_include_columns = keep_columns
    # # remove ["attack", "node_k8s-master-1", "node_k8s-worker-1", "node_k8s-worker-2", "Pod_Logs"] from list keep_columns
    # # prometheus already removed
    # base_include_columns = [
    #     col for col in base_include_columns if col not in ["attack", "node_k8s-master-1", "node_k8s-worker-1", "node_k8s-worker-2", "Pod_Logs"]
    # ]
    # # add network2 columns
    # base_net2_columns = [
    #     "fl_dur",
    #     "tot_fw_pk",
    #     "tot_bw_pk",
    #     "tot_l_fw_pkt",
    #     "fw_pkt_l_max",
    #     "fw_pkt_l_min",
    #     "fw_pkt_l_avg",
    #     "fw_pkt_l_std",
    #     "bw_pkt_l_max",
    #     "bw_pkt_l_min",
    #     "bw_pkt_l_avg",
    #     "bw_pkt_l_std",
    #     "fl_byt_s",
    #     "fl_pkt_s",
    #     "fl_iat_avg",
    #     "fl_iat_std",
    #     "fl_iat_max",
    #     "fl_iat_min",
    #     "fin_cnt",
    #     "syn_cnt",
    #     "rst_cnt",
    #     "pst_cnt",
    #     "ack_cnt",
    #     "urg_cnt",
    #     "down_up_ratio",
    #     "pkt_size_avg",
    #     "fw_seg_avg",
    #     "bw_seg_avg",
    #     "subfl_fw_pk",
    #     "subfl_fw_byt",
    #     "subfl_bw_pkt",
    #     "subfl_bw_byt",
    #     "fw_win_byt",
    #     "bw_win_byt",
    #     "fw_act_pkt",
    #     "fw_seg_min",
    # ]
    # base_include_columns += base_net2_columns
    #####
    base_include_columns = [
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
        "setuid",
        "setgid",
        "mount",
        "umount",
        "chown",
        "chmod",
        "NetInCount",
        "NetOutCount",
        "NetInSize",
        "NetOutSize",
        "fl_dur",
        "tot_fw_pk",
        "tot_bw_pk",
        "tot_l_fw_pkt",
        "fw_pkt_l_max",
        "fw_pkt_l_min",
        "fw_pkt_l_avg",
        "fw_pkt_l_std",
        "bw_pkt_l_max",
        "bw_pkt_l_min",
        "bw_pkt_l_avg",
        "bw_pkt_l_std",
        "fl_byt_s",
        "fl_pkt_s",
        "fl_iat_avg",
        "fl_iat_std",
        "fl_iat_max",
        "fl_iat_min",
        "fin_cnt",
        "syn_cnt",
        "rst_cnt",
        "pst_cnt",
        "ack_cnt",
        "urg_cnt",
        "down_up_ratio",
        "pkt_size_avg",
        "fw_seg_avg",
        "bw_seg_avg",
        "subfl_fw_pk",
        "subfl_fw_byt",
        "subfl_bw_pkt",
        "subfl_bw_byt",
        "fw_win_byt",
        "bw_win_byt",
        "fw_act_pkt",
        "fw_seg_min",
    ]

    # Apply rolling windows separately for each row position across time windows
    grouped_data = df[base_include_columns].copy().groupby(["attack", "row_within_window"])

    # Calculate mean (average)
    window_mean_df = (
        grouped_data.rolling(rolling_window, min_periods=1).mean().reset_index(level=[1, 2], drop=True)  # Drop extra index levels from rolling
    )
    window_mean_df.columns = [f"{col}_roll{rolling_window}_mean" for col in window_mean_df.columns]

    # Caluclate median
    window_median_df = (
        grouped_data.rolling(rolling_window, min_periods=1).median().reset_index(level=[1, 2], drop=True)  # Drop extra index levels from rolling
    )
    window_median_df.columns = [f"{col}_roll{rolling_window}_median" for col in window_median_df.columns]

    # Calculate standard deviation
    window_std_df = (
        grouped_data.rolling(rolling_window, min_periods=1)
        .std()
        .reset_index(level=[1, 2], drop=True)  # Drop extra index levels from rolling
        .fillna(0)  # Rolling std might produce NaN if window has only 1 value
    )
    window_std_df.columns = [f"{col}_roll{rolling_window}_std" for col in window_std_df.columns]

    # Combine the aggregated features
    window_df = pd.concat([window_mean_df, window_median_df, window_std_df], axis=1)

    # print(window_df.dtypes)

    window_df.reset_index(drop=True, inplace=True)  # Reset index to avoid multi-index
    window_df.index = original_index  # Restore original index
    return window_df


# +++++++++++++++++++++++++++++++++++
def perform_kfold_cv(model_func, data, meta_path, n_splits=5):
    """
    Perform stratified K-fold cross-validation

    Args:
        model_func: Function to create and train a model
        data: Tuple containing (X, y, num_classes, class_weights_dict, scaler, oversample)
        meta_path: Path for saving results
        n_splits: Number of folds for cross-validation
    """
    X, y, num_classes, class_weights_dict, scaler, oversample = data

    # Initialize metrics storage
    fold_metrics = {"accuracy": [], "balanced_accuracy": [], "precision": [], "recall": [], "f1": []}

    # Create stratified K-fold split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Performing {n_splits}-fold stratified cross-validation...")

    # Create a file to store fold results
    cv_meta_path = meta_path.replace(".txt", "_cv.txt")
    with open(cv_meta_path, "w") as cv_file:
        cv_file.write(f"Stratified {n_splits}-Fold Cross-Validation Results\n")
        cv_file.write("=" * 60 + "\n\n")
        cv_file.write("Note: In stratified K-fold CV, each split maintains the class distribution of the dataset.\n\n")

    # Perform stratified K-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")

        # Split data for this fold
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

        # Log class distribution in this fold
        from collections import Counter

        print(f"Fold {fold + 1} class distribution in training set:")
        print(Counter(y_train))
        print(f"Fold {fold + 1} class distribution in test set:")
        print(Counter(y_test))

        # Apply SMOTE if specified (only to training data)
        if oversample == "smote":
            from imblearn.over_sampling import SMOTE

            print(f"Applying SMOTE to fold {fold + 1} training data...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE, class distribution in training set:")
            print(Counter(y_train))

        # Calculate class weights for this fold's training data
        fold_class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        fold_weights_dict = dict(enumerate(fold_class_weights))

        # Apply normalization if specified
        if scaler is not None:
            fold_scaler = clone(scaler)
            X_train = fold_scaler.fit_transform(X_train)
            X_test = fold_scaler.transform(X_test)

            # Convert back to DataFrame if necessary
            if isinstance(X, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=X.columns)
                X_test = pd.DataFrame(X_test, columns=X.columns)

        # Create fold-specific meta path
        fold_path = meta_path.replace(".txt", f"_fold{fold + 1}.txt")

        # Train and evaluate model on this fold
        fold_data = (X_train, X_test, y_train, y_test, num_classes, fold_weights_dict)

        # Modify model functions to return predictions
        model, y_pred = train_and_evaluate_model(model_func, fold_data, fold_path)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Store metrics
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["balanced_accuracy"].append(balanced_acc)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["f1"].append(f1)

        # Save fold metrics to file
        with open(cv_meta_path, "a") as cv_file:
            cv_file.write(f"Fold {fold + 1} Results:\n")
            cv_file.write(f"Accuracy: {accuracy:.4f}\n")
            cv_file.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
            cv_file.write(f"Precision (macro): {precision:.4f}\n")
            cv_file.write(f"Recall (macro): {recall:.4f}\n")
            cv_file.write(f"F1 Score (macro): {f1:.4f}\n\n")

            # Save classification report
            cv_file.write(f"Classification Report for Fold {fold + 1}:\n")
            cv_file.write(classification_report(y_test, y_pred, digits=4))
            cv_file.write("\n" + "=" * 50 + "\n\n")

    # Calculate average metrics
    avg_accuracy = np.mean(fold_metrics["accuracy"])
    std_accuracy = np.std(fold_metrics["accuracy"])
    avg_balanced_accuracy = np.mean(fold_metrics["balanced_accuracy"])
    std_balanced_accuracy = np.std(fold_metrics["balanced_accuracy"])
    avg_precision = np.mean(fold_metrics["precision"])
    std_precision = np.std(fold_metrics["precision"])
    avg_recall = np.mean(fold_metrics["recall"])
    std_recall = np.std(fold_metrics["recall"])
    avg_f1 = np.mean(fold_metrics["f1"])
    std_f1 = np.std(fold_metrics["f1"])

    # Save average metrics
    with open(cv_meta_path, "a") as cv_file:
        cv_file.write("Stratified Cross-Validation Summary:\n")
        cv_file.write(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
        cv_file.write(f"Average Balanced Accuracy: {avg_balanced_accuracy:.4f} ± {std_balanced_accuracy:.4f}\n")
        cv_file.write(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}\n")
        cv_file.write(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}\n")
        cv_file.write(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n")

    # Train a final model on all data for deployment
    print("\nTraining final model on all data...")
    if scaler is not None:
        X_transformed = scaler.fit_transform(X)
        if isinstance(X, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed, columns=X.columns)
    else:
        X_transformed = X

    # For final model, we don't split the data
    final_data = (X_transformed, None, y, None, num_classes, class_weights_dict)
    final_model = train_final_model(model_func, final_data, meta_path)

    return {
        "avg_metrics": {"accuracy": avg_accuracy, "balanced_accuracy": avg_balanced_accuracy, "precision": avg_precision, "recall": avg_recall, "f1": avg_f1},
        "std_metrics": {"accuracy": std_accuracy, "balanced_accuracy": std_balanced_accuracy, "precision": std_precision, "recall": std_recall, "f1": std_f1},
        "fold_metrics": fold_metrics,
        "final_model": final_model,
    }


def perform_time_series_cv(model_func, data, meta_path, n_splits=5):
    raise NotImplementedError("Time series cross-validation is broken")


def train_and_evaluate_model(model_func, data, meta_path):
    """Helper function to train a model and get predictions"""
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    # This is a wrapper for your existing model functions that needs to return the model and predictions
    model_name = model_func.__name__.replace("make_model_", "")
    print(f"Training {model_name} model...")

    model_path = meta_path.replace(".txt", ".pickle")
    if model_name == "cnn":
        model_path = meta_path.replace(".txt", ".keras")

    # Train the model based on model type
    if model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        model.fit(X_train, y_train)
    elif model_name == "nb":
        model = GaussianNB()
        sample_weights = np.ones(len(y_train))
        for idx, y_val in enumerate(y_train):
            sample_weights[idx] = class_weights_dict[y_val]
        model.fit(X_train, y_train, sample_weight=sample_weights)
    elif model_name == "gb":
        model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
    elif model_name == "dt":
        model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
        model.fit(X_train, y_train)
    elif model_name == "svm":
        # model = OneVsRestClassifier(LinearSVC(class_weight="balanced"), n_jobs=4)
        # model.fit(X_train, y_train)
        # n_estimators = 10
        # model = BaggingClassifier(  # ~11min
        #     estimator=SVC(class_weight="balanced", probability=True, random_state=42, kernel="linear", verbose=True),
        #     n_estimators=n_estimators,
        #     max_samples=1.0 / n_estimators,
        #     max_features=1.0,
        #     bootstrap=True,
        #     n_jobs=6,
        #     verbose=1,
        # )
        model = SVC(class_weight="balanced", probability=True, random_state=42, kernel="linear", verbose=True)  # RUn ml.py WITH -m sklearnex !!!
        model.fit(X_train, y_train)
    elif model_name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", max_iter=200, random_state=42)
        model.fit(X_train, y_train)
    elif model_name == "mlp3":
        y_train_oh = to_categorical(y_train, num_classes)
        y_test_oh = to_categorical(y_test, num_classes)
        X_train_reshaped = np.expand_dims(X_train, axis=-1)
        X_test_reshaped = np.expand_dims(X_test, axis=-1)

        model_path = meta_path.replace(".txt", ".keras")
        # TODO: if _best exists use it instead
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = keras.models.load_model(model_path)
        else:
            print("Creating new MLP model...")

            model = keras.Sequential(
                [
                    layers.Dense(64, activation="elu", input_shape=(X_train_reshaped.shape[1],)),
                    layers.Dropout(0.4),
                    layers.Dense(32, activation="elu"),
                    layers.Dropout(0.4),
                    layers.Dense(16, activation="elu"),
                    layers.Dropout(0.5),
                    layers.Dense(num_classes, activation="softmax"),  # multiclass
                ]
            )

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy", keras.metrics.F1Score(average="macro", threshold=None, name="f1_macro", dtype=None)],
            )

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            )
            early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=model_path.replace(".keras", "_best.keras"),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            )
            history = model.fit(
                X_train_reshaped,
                y_train_oh,
                epochs=50,
                batch_size=128,
                validation_data=(X_test_reshaped, y_test_oh),
                callbacks=[reduce_lr, early_stopping, model_checkpoint],
                # class_weight=class_weights_dict,#TODO:good enough?
            )

        # Evaluate model
        y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
    elif model_name == "cnn":
        y_train_oh = to_categorical(y_train, num_classes)
        y_test_oh = to_categorical(y_test, num_classes)  # Needed for evaluation
        X_train_reshaped = np.expand_dims(X_train, axis=-1)
        X_test_reshaped = np.expand_dims(X_test, axis=-1)

        # Check if a pre-trained model exists
        if os.path.exists(model_path):
            print(f"Loading existing CNN model from {model_path}")
            model = keras.models.load_model(model_path)
        else:
            print("Creating new CNN model...")
            model = keras.Sequential(
                [
                    layers.Conv1D(filters=16, kernel_size=5, input_shape=(X_train_reshaped.shape[1], 1), padding="same", activation="elu"),
                    layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="elu"),
                    layers.MaxPooling1D(pool_size=2),
                    # layers.Dropout(0.2),
                    layers.Flatten(),
                    layers.Dense(32, activation="elu"),
                    layers.Dropout(0.3),
                    layers.Dense(16, activation="elu"),
                    layers.Dropout(0.3),
                    layers.Dense(num_classes, activation="softmax"),  # multiclass
                ]
            )

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy", keras.metrics.F1Score(average="macro", threshold=None, name="f1_macro", dtype=None)],
            )

            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
            early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            print("Training CNN model...")
            history = model.fit(
                X_train_reshaped,
                y_train_oh,
                epochs=30,  # 50
                batch_size=128,
                validation_data=(X_test_reshaped, y_test_oh),
                callbacks=[reduce_lr, early_stopping],
                # class_weight=class_weights_dict, #TODO:good enough?
                verbose=1,  # Set verbosity as needed
            )
            # Optionally save training history plots for each fold if desired

        # Get predictions for CNN
        y_pred_prob = model.predict(X_test_reshaped)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Get predictions for non-CNN models
    if model_name != "cnn":
        y_pred = model.predict(X_test)

    # Save the model
    if model_name == "cnn":
        model.save(model_path)
    else:
        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)

    # Generate classification report
    save_classification_report(y_test, y_pred, meta_path)

    # # Generate graphics if needed (broken dont use)
    # if X_test is not None and y_test is not None:
    #     save_graphics(model, X_test, y_test, y_pred, meta_path)

    return model, y_pred


def train_final_model(model_func, data, meta_path):
    print(f" [!] Train final model on all data using the make_model_{model_func.__name__.replace('make_model_', '')} function.")
    return


# +++++++++++++++++++++++++++++++++++++
# TODO split every attack class+benign. normalize each SEPERATELY
# split every dataset by attacks/benign -> for each file do {rolling windows -> normalize -> save} -> join all datasets tfidf
# ^ like kubanomaly
# keep 99.csv just in case so if needed can just set to 0 and load gain


def sample_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ##### EXPERIMENT: random undersample class 0 to 20k
    # remove random rows until 20k sampeles left of class 0
    class_0_indices = df[df["attack"] == 0].index
    num_class_0 = len(class_0_indices)
    target_class_0_count = 51624  # 72912  # 50001

    if num_class_0 > target_class_0_count:
        print(f"Undersampling class 0 from {num_class_0} to {target_class_0_count} samples (preserving 3-row blocks)...")

        if target_class_0_count % 3 != 0:
            raise ValueError(f"target_class_0_count ({target_class_0_count}) must be divisible by 3 for block sampling.")

        # Identify the unique blocks (groups of 3 indices) for class 0
        # Assuming index corresponds to row number and blocks are 0-2, 3-5, etc.
        class_0_blocks = class_0_indices // 3
        unique_class_0_blocks = np.unique(class_0_blocks)

        num_blocks_to_keep = target_class_0_count // 3

        if len(unique_class_0_blocks) < num_blocks_to_keep:
            print(
                f"Warning: Not enough unique blocks ({len(unique_class_0_blocks)}) for class 0 to reach target count {target_class_0_count}. Keeping all available blocks."
            )
            num_blocks_to_keep = len(unique_class_0_blocks)

        # Randomly select block IDs to keep
        rng = np.random.default_rng(seed=42)
        keep_block_ids = rng.choice(unique_class_0_blocks, num_blocks_to_keep, replace=False)

        # Get all row indices belonging to the selected blocks
        keep_class_0_indices = df.index[np.isin(df.index // 3, keep_block_ids) & (df["attack"] == 0)]

        # Ensure we didn't accidentally select more/less than intended due to block overlaps or incomplete blocks
        # This check might be necessary if blocks aren't perfectly aligned with class 0
        if len(keep_class_0_indices) != target_class_0_count and len(unique_class_0_blocks) >= num_blocks_to_keep:
            print(f"Warning: Kept {len(keep_class_0_indices)} indices for class 0, expected {target_class_0_count}. Adjusting target.")

        # Get indices of other classes
        other_class_indices = df[df["attack"] != 0].index

        # Combine the indices to keep
        keep_indices = np.concatenate([keep_class_0_indices, other_class_indices])

        # Select the rows using the combined indices and sort by original index
        df = df.loc[keep_indices].sort_index()
        print(f"Data shape after undersampling class 0: {df.shape}")
        print(df["attack"].value_counts())

    # limit class 8 to target_class_8_count samples
    class_8_indices = df[df["attack"] == 8].index
    num_class_8 = len(class_8_indices)
    target_class_8_count = 5001  # Must be divisible by 3
    if num_class_8 > target_class_8_count:
        print(f"Undersampling class 8 from {num_class_8} to {target_class_8_count} samples (preserving 3-row blocks)...")

        if target_class_8_count % 3 != 0:
            raise ValueError(f"target_class_8_count ({target_class_8_count}) must be divisible by 3 for block sampling.")

        # Identify the unique blocks (groups of 3 indices) for class 8
        class_8_blocks = class_8_indices // 3
        unique_class_8_blocks = np.unique(class_8_blocks)

        num_blocks_to_keep = target_class_8_count // 3

        if len(unique_class_8_blocks) < num_blocks_to_keep:
            print(
                f"Warning: Not enough unique blocks ({len(unique_class_8_blocks)}) for class 8 to reach target count {target_class_8_count}. Keeping all available blocks."
            )
            num_blocks_to_keep = len(unique_class_8_blocks)

        # Randomly select block IDs to keep
        rng = np.random.default_rng(seed=42)  # Use the same seed for consistency if desired, or different otherwise
        keep_block_ids = rng.choice(unique_class_8_blocks, num_blocks_to_keep, replace=False)

        # Get all row indices belonging to the selected blocks
        keep_class_8_indices = df.index[np.isin(df.index // 3, keep_block_ids) & (df["attack"] == 8)]

        # Optional: Add check similar to class 0 if exact count is critical
        if len(keep_class_8_indices) != target_class_8_count and len(unique_class_8_blocks) >= num_blocks_to_keep:
            print(f"Warning: Kept {len(keep_class_8_indices)} indices for class 8, expected {target_class_8_count}. Adjusting target.")

        # Get indices of other classes (excluding class 8 this time)
        other_class_indices = df[df["attack"] != 8].index

        # Combine the indices to keep
        keep_indices = np.concatenate([keep_class_8_indices, other_class_indices])

        # Select the rows using the combined indices and sort by original index
        df = df.loc[keep_indices].sort_index()
        print(f"Data shape after undersampling class 8: {df.shape}")
        print(df["attack"].value_counts())  # Print counts again after class 8 sampling
    return df


# make benign & attack normalzation/windows SEPERATE
def make_dataset(
    attacks_csv: str,
    benign_csv: str,
    normalize_type: str = "no",
    podlogs_type: str = "no",
    rolling_windows: list[int] = None,
    oversample: str = "no",
    split_type: str = "random",
    attack_already_include_windows: bool = False,
    attacks_include_normalize: bool = True,  # only when attacks_csv is a folder
    skip99: bool = True,
    experiment: str = "",
) -> tuple[list, list, list, list, int]:
    # load data

    # check if attacks_csv is a folder
    if os.path.isdir(attacks_csv):
        print(f"Loading individual sections from folder {attacks_csv}")
        if attacks_include_normalize:
            normalize_type = "no"
            print("Setting normalize_type to 'no' by force.")

        df = None
        # MUST LOAD all attacks even if skip99 otherwise widnowing will make wrong resutls -> too easy guess
        for file in os.listdir(attacks_csv):
            if file.endswith(".csv"):  # and (not file.startswith("99") or not skip99)
                file_path = os.path.join(attacks_csv, file)
                print(f"Loading {file_path}")
                temp_df = pd.read_csv(file_path)

                # FIXME EXPERIEMNTS
                # drop columns indices from 66 to 355 inclusive
                # temp_df = temp_df.drop(temp_df.columns[66:356], axis=1)

                # only use node 1 (where node_k8s-master-1 is 1)
                # temp_df = temp_df[temp_df["node_k8s-master-1"] == 1]

                # temp_df["attack"] = temp_df["attack"].replace(99, 0)

                if df is None:
                    df = temp_df
                else:
                    df = pd.concat([df, temp_df], ignore_index=True)

        df.reset_index(drop=True, inplace=True)
        print(f"Combined data shape: {df.shape}")
    else:
        print(f"Loading attacks data from {attacks_csv}")
        attacks_df = pd.read_csv(attacks_csv)
        print(f"Attacks data shape: {attacks_df.shape}")
        print(attacks_df.head())
        if benign_csv != "":
            print(f"Loading benign data from {benign_csv}")
            benign_df = pd.read_csv(benign_csv)
            print(f"Benign data shape: {benign_df.shape}")
            print(benign_df.head())
            # reset index
            attacks_df = attacks_df.reset_index(drop=True)
            benign_df = benign_df.reset_index(drop=True)

            # assert columns are the same
            if not attacks_df.columns.equals(benign_df.columns):
                print(attacks_df.columns.difference(benign_df.columns))
                print(benign_df.columns.difference(attacks_df.columns))
                raise ValueError("Columns are not the same.")

            # Combine attacks and benign data
            df = pd.concat([attacks_df, benign_df])
            print(f"Combined data shape: {df.shape}")
            del attacks_df
            del benign_df
            gc.collect()
        else:
            print("No separate benign data provided. Assuming --attacks contains all data.")
            df = attacks_df

    print(df["attack"].value_counts())

    # if experiment == "promv2":  # Prom v2 filter by node. to prevent leakage between nodes
    # now default!
    master1ip = "192.168.122.216"
    worker1ip = "192.168.122.233"
    worker2ip = "192.168.122.228"
    master1node = "k8s-master-1"
    worker1node = "k8s-worker-1"
    worker2node = "k8s-worker-2"
    prom_metrics_range = (66, 356)  # 356 excluding
    cols_to_drop = []
    stats = (0, 0, 0)
    force_same_features_across_nodes = False  # also incl when metric only on some node

    # Dictionary to store column names without IPs or node identifiers and their counts
    colname_counts = {}

    # First pass: Normalize column names and count occurrences
    for i in range(prom_metrics_range[0], prom_metrics_range[1]):
        colname = df.columns[i]
        # Remove all IP addresses and node identifiers to get the base metric name
        normalized_colname = colname
        for ip in [master1ip, worker1ip, worker2ip]:
            normalized_colname = normalized_colname.replace(ip, "")
        for node in [master1node, worker1node, worker2node]:
            normalized_colname = normalized_colname.replace(node, "")
        colname_counts[normalized_colname] = colname_counts.get(normalized_colname, 0) + 1

    # Find columns shared by all three nodes
    shared_prom_cols = [col for col, count in colname_counts.items() if count == 3]
    # print all shared cols
    # print(shared_prom_cols)

    stats_included_prom = []
    # Second pass: Filter columns and apply node-specific zeroing
    for i in range(prom_metrics_range[0], prom_metrics_range[1]):
        colname = df.columns[i]
        normalized_colname = colname
        for ip in [master1ip, worker1ip, worker2ip]:
            normalized_colname = normalized_colname.replace(ip, "")
        for node in [master1node, worker1node, worker2node]:
            normalized_colname = normalized_colname.replace(node, "")
        prefix_ip = "instance="
        prefix_node = "node="

        # Only process columns that are shared by all nodes
        if force_same_features_across_nodes and normalized_colname not in shared_prom_cols:
            cols_to_drop.append(i)
            continue

        if prefix_ip + master1ip in colname or prefix_node + master1node in colname:
            # Set to 0 if not node_k8s-master-1
            df[colname] = df[colname].where(df["node_k8s-master-1"] == 1, 0)
            stats = (stats[0] + 1, stats[1], stats[2])
        elif prefix_ip + worker1ip in colname or prefix_node + worker1node in colname:
            # Set to 0 if not node_k8s-worker-1
            df[colname] = df[colname].where(df["node_k8s-worker-1"] == 1, 0)
            stats = (stats[0], stats[1] + 1, stats[2])
        elif prefix_ip + worker2ip in colname or prefix_node + worker2node in colname:
            # Set to 0 if not node_k8s-worker-2
            df[colname] = df[colname].where(df["node_k8s-worker-2"] == 1, 0)
            stats = (stats[0], stats[1], stats[2] + 1)
        else:
            cols_to_drop.append(i)
            continue

        stats_included_prom.append(colname)

    # Drop columns that aren't shared or can't identify node
    df = df.drop(df.columns[cols_to_drop], axis=1)
    print(f"* Prom v2 filter applied. Kept {stats[0]} master, {stats[1]} worker1, {stats[2]} worker2 metrics.")
    print(f"* Prom v2 filter applied. Dropped columns: {len(cols_to_drop)} (non-shared or cannot identify node)")
    print(f"* {len(shared_prom_cols)} shared metric types across all nodes")
    print(f"* {len(stats_included_prom)} total metrics after filtering")

    # save to file _promv2.txt
    with open(attacks_csv.replace(".csv", "_promv2.txt"), "w") as f:
        f.write(f"Prom v2 filter applied. Kept {stats[0]} master, {stats[1]} worker1, {stats[2]} worker2 metrics.\n")
        f.write(f"Prom v2 filter applied. Dropped columns: {len(cols_to_drop)} (non-shared or cannot identify node)\n")
        f.write(f"Prom v2 filter applied. {len(shared_prom_cols)} shared metric types across all nodes\n")
        f.write(f"Prom v2 filter applied. {len(stats_included_prom)} total metrics after filtering\n")
        for c in stats_included_prom:
            f.write(c + "\n")

    # print the value and node_* columns to verify it workd
    # print(df[[stats_included_prom[0]] + [col for col in df.columns if "node_k8s-" in col]].head(10)) # ok  works
    # exit(42)

    nonwindowsampledcsv = attacks_csv.replace(".csv", "_sampled.csv")
    if not os.path.exists(nonwindowsampledcsv) and not os.path.isdir(attacks_csv):
        dfs = sample_dataset(df.copy())
        print(f" (will only run once for input dataset) Saving non-windowed sampled DF to {nonwindowsampledcsv}")
        dfs.to_csv(nonwindowsampledcsv, index=False)
        del dfs

    # # concat rolling window to df
    if rolling_windows is None or attack_already_include_windows:
        rolling_windows = []
        if attack_already_include_windows:
            print("Attack data already includes rolling windows. Skipping rolling window creation.")
    # Store all window dataframes to concatenate at once
    all_window_dfs = []
    for window_size in rolling_windows:
        if window_size > 0:
            print(f"Applying rolling window over {window_size} samples per node...")
            window_df = make_rolling_window(df, window_size)
            all_window_dfs.append(window_df)

    # Concatenate all window features at once
    if all_window_dfs:
        # concat BEFORE "node_k8s-"...
        nodeassign_idx = df.columns.get_loc("node_k8s-master-1")
        left_cols = df.columns[:nodeassign_idx]
        right_cols = df.columns[nodeassign_idx:]
        # Concatenate all window dataframes with the original dataframe
        df = pd.concat([df[left_cols]] + all_window_dfs + [df[right_cols]], axis=1)

        # df = pd.concat([df] + all_window_dfs, axis=1)
        print(f"Data shape after adding all rolling windows: {df.shape}")

    df = sample_dataset(df)

    # SAVE to csv _sampled
    sampled_csvpath = attacks_csv.replace(".csv", "_windows_sampled.csv")
    if all_window_dfs and not os.path.exists(sampled_csvpath) and not os.path.isdir(attacks_csv):
        print(f" (will only run once for input dataset) Saving sampled DF to {sampled_csvpath}")
        df.to_csv(sampled_csvpath, index=False)
    ############
    # after rolling windows created
    # if experiment == "nonet":
    #     # 26-65
    #     print("Experiment: dropping columns 26 to 65 inclusive NET")
    #     df = df.drop(df.columns[26:66], axis=1)
    # elif experiment == "nosys":
    #     # 0-24
    #     print("Experiment: dropping columns 0 to 24 inclusive SYS")
    #     df = df.drop(df.columns[0:25], axis=1)

    #### Experiments
    # onlyprom,onlynet,onlysys
    to_drop_syscalls = [
        "clone",
        "fork",
        "mount",
        "umount",
        "mmap",
        "chdir",
        "close",
        "setgid",
        "chmod",
        "unlink",
        "write",
        "connect",
        "brk",
        "munmap",
        "open",
        "rename",
        "setuid",
        "select",
        "execve",
        "read",
        "creat",
        "kill",
        "poll",
        "accept",
        "chown",
    ]
    # store indices of syscall columns to drop (syscall just need to be in column name. not exact match)
    to_drop_syscalls_idx = []
    for syscall in to_drop_syscalls:
        # get index of syscall column
        syscall_idx = df.columns[df.columns.str.contains(syscall)].tolist()
        if syscall_idx:
            to_drop_syscalls_idx.extend(syscall_idx)
    to_drop_net = [
        "NetInCount",
        "NetOutCount",
        "NetInSize",
        "NetOutSize",
        "fl_dur",
        "tot_fw_pk",
        "tot_bw_pk",
        "tot_l_fw_pkt",
        "fw_pkt_l_max",
        "fw_pkt_l_min",
        "fw_pkt_l_avg",
        "fw_pkt_l_std",
        "bw_pkt_l_max",
        "bw_pkt_l_min",
        "bw_pkt_l_avg",
        "bw_pkt_l_std",
        "fl_byt_s",
        "fl_pkt_s",
        "fl_iat_avg",
        "fl_iat_std",
        "fl_iat_max",
        "fl_iat_min",
        "fin_cnt",
        "syn_cnt",
        "rst_cnt",
        "pst_cnt",
        "ack_cnt",
        "urg_cnt",
        "down_up_ratio",
        "pkt_size_avg",
        "fw_seg_avg",
        "bw_seg_avg",
        "subfl_fw_pk",
        "subfl_fw_byt",
        "subfl_bw_pkt",
        "subfl_bw_byt",
        "fw_win_byt",
        "bw_win_byt",
        "fw_act_pkt",
        "fw_seg_min",
    ]
    to_drop_net_idx = []
    for net in to_drop_net:
        # get index of net column
        net_idx = df.columns[df.columns.str.contains(net)].tolist()
        if net_idx:
            to_drop_net_idx.extend(net_idx)
    # all columns cotnaining { or :
    to_drop_prom = []
    for col in df.columns:
        if "{" in col or ":" in col:
            to_drop_prom.append(col)
    if experiment == "onlyprom":
        # drop 0-65
        print("Experiment: dropping columns 0 to 65 inclusive ONLY PROM")
        # df = df.drop(df.columns[0:66], axis=1)
        print(f"len to drop syscalls: {len(to_drop_syscalls_idx)}")
        print(f"len to drop net: {len(to_drop_net_idx)}")
        df = df.drop(to_drop_syscalls_idx + to_drop_net_idx, axis=1)
    elif experiment == "onlynet":
        # drop 0-25, 66-355
        print("Experiment: dropping columns 0 to 25 inclusive and 66 to 355 inclusive ONLY NET")
        # Check test42.py MUST ALSO include index of rolling windows!!  356-446 syscall window!
        # to_drop = list(range(0, 26)) + list(range(66, 356))
        print(f"len to drop prom: {len(to_drop_prom)}")
        print(f"len to drop syscalls: {len(to_drop_syscalls_idx)}")
        df = df.drop(to_drop_syscalls_idx + to_drop_prom, axis=1)
    elif experiment == "onlysys":
        # drop 26-355
        print("Experiment: dropping columns 26 to 355 inclusive ONLY SYS")
        # to_drop = list(range(26, 356))
        # df = df.drop(df.columns[to_drop], axis=1)
        print(f"len to drop prom: {len(to_drop_prom)}")
        print(f"len to drop net: {len(to_drop_net_idx)}")
        df = df.drop(to_drop_net_idx + to_drop_prom, axis=1)
    elif experiment == "noprom":
        print(f"len to drop prom: {len(to_drop_prom)}")
        df = df.drop(to_drop_prom, axis=1)
    elif experiment == "nonet":
        print(f"len to drop net: {len(to_drop_net_idx)}")
        df = df.drop(to_drop_net_idx, axis=1)
    elif experiment == "nosys":
        print(f"len to drop syscalls: {len(to_drop_syscalls_idx)}")
        df = df.drop(to_drop_syscalls_idx, axis=1)
    ### End experiments

    # (only used for SPLIT) delete if skip99. after windowing. otherwise wrong windows!
    if os.path.isdir(attacks_csv):
        print("Treating time between attacks as benign" if not skip99 else "Ignoring time between attacks")
        if skip99:
            df = df[df["attack"] != 99]
        else:
            df["attack"] = df["attack"].replace(99, 0)

    if "binary" in experiment:
        # convert all attacks class 1-10 to 1
        print("Experiment: converting all attacks to 1")
        df["attack"] = df["attack"].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)

    print(f"Number of benign samples: {len(df[df['attack'] == 0])}")
    print(f"Number of attack samples: {len(df[df['attack'] != 0])}")
    print(f"Ratio of benign to attack samples: {len(df[df['attack'] == 0]) / len(df[df['attack'] != 0]):.2f}")
    print(df["attack"].value_counts())

    print(f"Processing podlogs...{podlogs_type}")
    if podlogs_type == "no":
        df = df.drop(columns=["Pod_Logs"], errors="ignore")  # ignore when used with experiment only*
    elif podlogs_type == "count":  # like v1
        # fillna
        df["Pod_Logs"] = df["Pod_Logs"].fillna("")
        # count number of matched keywords in Pod_Logs col for each row
        df["Pod_Logs"] = df["Pod_Logs"].apply(lambda x: sum(x.lower().count(keyword) for keyword in keywords_list[0]))

    elif podlogs_type == "tfidf":
        # fillna
        df["Pod_Logs"] = df["Pod_Logs"].fillna("")
        # tf-idf on Pod_Logs col
        tfidf = TfidfVectorizer(max_features=25, ngram_range=(1,10))
        tfidf_matrix = tfidf.fit_transform(df["Pod_Logs"])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
        # Add a prefix to the tfidf_df columns to avoid naming conflicts
        tfidf_df = tfidf_df.add_prefix("tfidf_")
        df = df.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)
        df = pd.concat([df, tfidf_df], axis=1)
        df = df.drop(columns=["Pod_Logs"])

    # raise if any NaN values
    if df.isnull().values.any():
        raise ValueError("Data contains NaN values.")

    # Experiment: master1,worker1,worker2
    if experiment == "master1":
        print("Experiment: master1")
        df = df[df["node_k8s-master-1"] == 1]
    elif experiment == "worker1":
        print("Experiment: worker1")
        df = df[df["node_k8s-worker-1"] == 1]
    elif experiment == "worker2":
        print("Experiment: worker2")
        df = df[df["node_k8s-worker-2"] == 1]

    print(f"Final shape: {df.shape}")

    # Split data into features (X) and labels (y)
    X = df.drop(columns=["attack"])  # target column name
    y = df["attack"].astype(int)
    num_classes = len(y.unique())

    print("Computing class weights...")
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weights_dict = dict(enumerate(class_weights))

    # Create scaler based on normalize_type
    scaler = None
    if normalize_type == "minmax":
        scaler = MinMaxScaler()
    elif normalize_type == "zscore":
        scaler = StandardScaler()
    elif normalize_type == "l2":
        scaler = Normalizer(norm="l2")

    if split_type == "random":
        # Traditional random train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply SMOTE if needed
        if oversample == "smote":
            from imblearn.over_sampling import SMOTE
            from collections import Counter

            print("Class distribution before SMOTE:")
            print(Counter(y_train))
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print("After SMOTE, the class distribution is:")
            print(Counter(y_train))

        # Apply normalization
        if scaler is not None:
            print(f"Applying {normalize_type} normalization")
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            print("Data is not normalized.")

        return X_train, X_test, y_train, y_test, num_classes, class_weights_dict
    else:
        # Return the full dataset for CV
        return X, y, num_classes, class_weights_dict, scaler, oversample


def save_model_summary(model: keras.Model, meta_path: str):
    with open(meta_path, "a") as f:
        with redirect_stdout(f):
            model.summary()


def save_classification_report(y_true, y_pred, meta_path: str):
    # print balanced accuracy score
    print(f"Balanced accuracy score: {balanced_accuracy_score(y_true, y_pred):.4f}")
    with open(meta_path, "a") as f:
        with redirect_stdout(f):
            print(classification_report(y_true, y_pred, digits=4))
            print(f"Balanced accuracy score: {balanced_accuracy_score(y_true, y_pred):.4f}")


def save_graphics(model, X_test, y_test, y_pred, meta_path):
    print("Saving graphics...")
    meta_path = meta_path.replace(".txt", "_")

    # Convert y_test to class labels if it's in one-hot encoding
    if y_test.ndim > 1:  # Check if y_test is one-hot encoded
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test

    num_classes = len(np.unique(y_test_labels))

    # Get predicted probabilities based on model type
    if hasattr(model, "predict_proba"):  # scikit-learn models
        y_prob = model.predict_proba(X_test)
    elif hasattr(model, "decision_function") and hasattr(model, "classes_"):  # For SVM models
        # Convert decision function output to pseudo-probabilities
        decisions = model.decision_function(X_test)
        if decisions.ndim == 1:  # Binary case
            y_prob = np.column_stack([1 - decisions, decisions])
        else:  # Multi-class case
            # Apply softmax to convert to pseudo-probabilities
            y_prob = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
            y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)
    elif hasattr(model, "predict"):  # Keras or models without predict_proba
        y_prob = model.predict(X_test)
    else:
        raise ValueError("Model must have either predict or predict_proba method")

    # Ensure y_prob is the right shape (samples, classes)
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        # Binary classification case, reshape to proper probability format
        y_prob = np.column_stack((1 - y_prob, y_prob))

    # Ensure y_test is in one-hot format for ROC and PR curves
    if y_test.ndim == 1:  # Convert to one-hot if it's not already
        y_test = to_categorical(y_test, num_classes=num_classes)

    # Compute confusion matrix
    print("> confusion matrix...")
    cm = confusion_matrix(y_test_labels, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test_labels), yticklabels=np.unique(y_test_labels))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(meta_path + "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Compute ROC-AUC score
    print("> ROC-AUC score...")
    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
        print(f"  ROC-AUC Score: {roc_auc}")

        # add AUC score to legend
        plt.figure(figsize=(4, 4))
        for i in range(y_prob.shape[1]):
            fpr, tpr, thresholds = roc_curve(y_test[:, i], y_prob[:, i], drop_intermediate=False)
            roc_auc = roc_auc_score(y_test[:, i], y_prob[:, i], multi_class="ovo", average="macro")
            print(roc_auc)
            # plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.4f})")
            plt.plot(fpr, tpr, label=f"Class {i} (AUC={math.trunc(roc_auc * 10000) / 10000})")

        plt.plot([0, 1], [0, 1], "k--")  # Random guessing line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves per Class")
        plt.legend()
        # plt.show()
        plt.savefig(meta_path + "roc_curve.png", bbox_inches="tight")
        plt.close()
    # print("> ROC-AUC score V2...")
    # try:
    #     roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
    #     print(f"  ROC-AUC Score: {roc_auc}")

    #     # add AUC score to legend
    #     plt.figure(figsize=(4, 4))
    #     all_fpr = []
    #     all_tpr = []
        
    #     for i in range(y_prob.shape[1]):
    #         fpr, tpr, thresholds = roc_curve(y_test[:, i], y_prob[:, i], drop_intermediate=False)
    #         roc_auc = roc_auc_score(y_test[:, i], y_prob[:, i])
    #         print(roc_auc)
    #         plt.plot(fpr, tpr, label=f"Class {i} (AUC={math.trunc(roc_auc * 10000) / 10000})")
    #         all_fpr.append(fpr)
    #         all_tpr.append(tpr)
        
    #     # Compute macro-average ROC
    #     # Interpolate all ROC curves at this points
    #     mean_fpr = np.linspace(0, 1, 100)
    #     tprs = []
    #     for fpr, tpr in zip(all_fpr, all_tpr):
    #         tprs.append(np.interp(mean_fpr, fpr, tpr))
        
    #     # Compute mean TPR
    #     mean_tpr = np.mean(tprs, axis=0)
    #     macro_auc = auc(mean_fpr, mean_tpr)
        
    #     # Plot macro-average ROC
    #     plt.plot(mean_fpr, mean_tpr, 'k:', 
    #             label=f'Macro-average (AUC={math.trunc(macro_auc * 10000) / 10000})',
    #             linewidth=2)

    #     plt.plot([0, 1], [0, 1], "k--")  # Random guessing line
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title("ROC Curves per Class")
    #     plt.legend()
    #     plt.savefig(meta_path + "roc_curve.png", bbox_inches="tight")
    #     plt.show()
    #     plt.close()
    except Exception as e:
        print(f"Error computing ROC curve: {e}")

    # Compute Precision-Recall curve
    print("> Precision-Recall curve...")
    try:
        plt.figure(figsize=(4, 4))
        for i in range(y_prob.shape[1]):  # Loop through each class
            precision, recall, _ = precision_recall_curve(y_test[:, i], y_prob[:, i])
            avg_precision = average_precision_score(y_test[:, i], y_prob[:, i])
            plt.plot(recall, precision, label=f"Class {i} (AP={avg_precision:.4f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves per Class")
        plt.legend()
        plt.savefig(meta_path + "precision_recall_curve.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error computing precision-recall curve: {e}")

    # pca not good
    return
    # Create t-SNE visualization of classification space
    print("> Creating classification space visualization...")
    try:
        from sklearn.manifold import TSNE

        print("  X_test length:", len(X_test))

        # Sample points if dataset is too large
        max_samples = 15000
        if len(X_test) > max_samples:
            print(f"  Sampling {max_samples} points for visualization...")
            indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_test_sample = X_test[indices]
            y_test_sample_labels = y_test_labels[indices]
            y_pred_sample = y_pred[indices]
        else:
            X_test_sample = X_test
            y_test_sample_labels = y_test_labels
            y_pred_sample = y_pred

        # Check if X_test_sample has more than 2 dimensions (e.g., from CNN models)
        if X_test_sample.ndim > 2:
            print("  Reshaping 3D data to 2D for t-SNE...")
            n_samples = X_test_sample.shape[0]
            X_test_sample = X_test_sample.reshape(n_samples, -1)

        # Reduce dimensionality to 2D for visualization
        print("  Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42)
        X_test_2d = tsne.fit_transform(X_test_sample)

        # Plot the samples
        plt.figure(figsize=(12, 10))

        # Determine if we should use a simplified approach for many classes
        unique_classes = np.unique(y_test_sample_labels)
        use_simplified = False  # len(unique_classes) > 10

        if use_simplified:
            # Simplified approach: just show correct vs incorrect
            correct = y_pred_sample == y_test_sample_labels
            plt.scatter(X_test_2d[correct, 0], X_test_2d[correct, 1], c="green", marker="o", alpha=0.6, label="Correct predictions")
            plt.scatter(X_test_2d[~correct, 0], X_test_2d[~correct, 1], c="red", marker="x", alpha=0.6, label="Incorrect predictions")

            # Add some annotations for misclassifications
            misclass_indices = np.where(~correct)[0]
            if len(misclass_indices) > 0:
                n_annotations = min(20, len(misclass_indices))
                for idx in np.random.choice(misclass_indices, n_annotations, replace=False):
                    plt.annotate(
                        f"T:{y_test_sample_labels[idx]}→P:{y_pred_sample[idx]}",
                        (X_test_2d[idx, 0], X_test_2d[idx, 1]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                    )
        else:
            # Use a colormap for the classes
            # cmap = plt.cm.get_cmap("Paired", len(unique_classes))
            cmap = plt.cm.get_cmap("tab20", len(unique_classes))

            # Detailed approach: show each class with consistent colors
            for i, cls in enumerate(unique_classes):
                # Get indices for this class
                indices = np.where(y_test_sample_labels == cls)[0]
                color = cmap(i)

                # Get correct predictions for this class
                correct_idx = indices[y_pred_sample[indices] == cls]
                if len(correct_idx) > 0:
                    plt.scatter(X_test_2d[correct_idx, 0], X_test_2d[correct_idx, 1], color=color, marker="o", alpha=0.6, label=f"Class {cls} (correct)")

                # Get incorrect predictions
                incorrect_idx = indices[y_pred_sample[indices] != cls]
                if len(incorrect_idx) > 0:
                    plt.scatter(X_test_2d[incorrect_idx, 0], X_test_2d[incorrect_idx, 1], color=color, marker="x", alpha=0.8)

                    # Add some annotations for misclassifications
                    n_annotations = min(5000, len(incorrect_idx))
                    for idx in np.random.choice(incorrect_idx, n_annotations, replace=False):
                        plt.annotate(
                            f"P:{y_pred_sample[idx]}",
                            # "",
                            (X_test_2d[idx, 0], X_test_2d[idx, 1]),
                            textcoords="offset points",
                            xytext=(0, 5),
                            ha="center",
                            color=color,
                        )

        plt.title("t-SNE Visualization of Test Samples and Classification Results")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")

        # Handle legend
        if not use_simplified:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc="best", bbox_to_anchor=(1.05, 1), ncol=1)
        else:
            plt.legend(loc="best")

        plt.tight_layout()
        plt.savefig(meta_path + "classification_space.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating classification space visualization: {e}")


def write_to_meta_file(meta_path: str, text: str):
    with open(meta_path, "a") as f:
        f.write(text + "\n")


# k-nearest neighbors model
def make_model_knn(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    model_path = meta_path.replace(".txt", ".pickle")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model = load(f)
    else:
        print("Creating new KNN model...")
        model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        fittime = time_ns()
        model.fit(X_train, y_train)
        fittime = time_ns() - fittime
        write_to_meta_file(meta_path, "fit_time " + str(fittime))
        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)

    # Evaluate model
    evaltime = time_ns()
    y_pred = model.predict(X_test)
    evaltime = time_ns() - evaltime
    write_to_meta_file(meta_path, "eval_time " + str(evaltime))
    print(classification_report(y_test, y_pred, digits=4))
    save_classification_report(y_test, y_pred, meta_path)

    save_graphics(model, X_test, y_test, y_pred, meta_path)


# naive bayes model (gaussian)
def make_model_nb(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    model_path = meta_path.replace(".txt", ".pickle")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model = load(f)
    else:
        print("Creating new Naive Bayes model...")
        model = GaussianNB()

        # class weights to sample weights
        sample_weights = np.ones(len(y_train))
        for idx, y_val in enumerate(y_train):
            sample_weights[idx] = class_weights_dict[y_val]

        fittime = time_ns()
        model.fit(X_train, y_train, sample_weight=sample_weights)  # no difference with/without weight
        fittime = time_ns() - fittime
        write_to_meta_file(meta_path, "fit_time " + str(fittime))
        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)

    # Evaluate model
    evaltime = time_ns()
    y_pred = model.predict(X_test)
    evaltime = time_ns() - evaltime
    write_to_meta_file(meta_path, "eval_time " + str(evaltime))

    # classification report
    print(classification_report(y_test, y_pred, digits=4))
    save_classification_report(y_test, y_pred, meta_path)

    save_graphics(model, X_test, y_test, y_pred, meta_path)


# gradient boosting model
def make_model_gb(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    model_path = meta_path.replace(".txt", ".pickle")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model = load(f)
    else:
        print("Creating new Gradient Boosting model...")
        model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
        fittime = time_ns()
        model.fit(X_train, y_train)
        fittime = time_ns() - fittime
        write_to_meta_file(meta_path, "fit_time " + str(fittime))
        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)

    evaltime = time_ns()
    y_pred = model.predict(X_test)
    evaltime = time_ns() - evaltime
    write_to_meta_file(meta_path, "eval_time " + str(evaltime))

    # classification report
    print(classification_report(y_test, y_pred, digits=4))
    save_classification_report(y_test, y_pred, meta_path)

    save_graphics(model, X_test, y_test, y_pred, meta_path)


# decision tree model
def make_model_dt(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    model_path = meta_path.replace(".txt", ".pickle")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model = load(f)
    else:
        print("Creating new Decision Tree model...")
        model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
        fittime = time_ns()
        model.fit(X_train, y_train)
        fittime = time_ns() - fittime
        write_to_meta_file(meta_path, "fit_time " + str(fittime))
        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)

    # Evaluate model
    evaltime = time_ns()
    y_pred = model.predict(X_test)
    evaltime = time_ns() - evaltime
    write_to_meta_file(meta_path, "eval_time " + str(evaltime))
    # print most relevant features
    # print("Most relevant features:")
    # feature_importances = model.feature_importances_
    # indices = np.argsort(feature_importances)[::-1]
    # for i in range(30):
    #     print(f"{i + 1}. Feature {indices[i]}: {feature_importances[indices[i]]}")

    # print decision tree
    from sklearn.tree import plot_tree

    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=[f"Feature #{i}" for i in range(X_test.shape[1])], class_names=[str(i) for i in range(num_classes)], fontsize=5)
    plt.savefig(meta_path + "decision_tree.png", bbox_inches="tight")
    # plt.show()
    plt.close()

    # classification report
    print(classification_report(y_test, y_pred, digits=4))
    save_classification_report(y_test, y_pred, meta_path)

    save_graphics(model, X_test, y_test, y_pred, meta_path)


# random forest model
def make_model_rf(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    model_path = meta_path.replace(".txt", ".pickle")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model = load(f)
    else:
        print("Creating new Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
        fittime = time_ns()
        model.fit(X_train, y_train)
        fittime = time_ns() - fittime
        write_to_meta_file(meta_path, "fit_time " + str(fittime))
        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)

    # Evaluate model
    evaltime = time_ns()
    y_pred = model.predict(X_test)
    evaltime = time_ns() - evaltime
    write_to_meta_file(meta_path, "eval_time " + str(evaltime))
    # print most relevant features
    print("Most relevant features:")
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    for i in range(30):
        print(f"{i + 1}. Feature {indices[i]}: {feature_importances[indices[i]]}")
        # print feature NAME

    # print plot showing most relevant features column chart (50 features max). get their names
    max = 50
    plt.figure(figsize=(10, 6))
    plt.axhline(y=np.mean(feature_importances), color="r", linestyle="--", label="Mean Importance")
    plt.bar(range(max), feature_importances[indices[:max]], align="center")
    plt.xticks(range(max), indices[:max], rotation=90)
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Importance")
    plt.title(f"Top {max} of {len(feature_importances)} Feature Importances")
    plt.tight_layout()
    plt.savefig(meta_path + "feature_importances.png", bbox_inches="tight")
    plt.close()

    # classification report
    print(classification_report(y_test, y_pred, digits=4))
    save_classification_report(y_test, y_pred, meta_path)

    save_graphics(model, X_test, y_test, y_pred, meta_path)


def make_model_mlp(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    # use scikit-learn MLPClassifier
    model_path = meta_path.replace(".txt", ".pickle")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model = load(f)
    else:
        print("Creating new MLP model...")
        model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", max_iter=200, random_state=42)  # 5min
        fittime = time_ns()
        model.fit(X_train, y_train)
        fittime = time_ns() - fittime
        write_to_meta_file(meta_path, "fit_time " + str(fittime))
        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)
    # Evaluate model
    evaltime = time_ns()
    y_pred = model.predict(X_test)
    evaltime = time_ns() - evaltime
    write_to_meta_file(meta_path, "eval_time " + str(evaltime))

    print(classification_report(y_test, y_pred, digits=4))
    save_classification_report(y_test, y_pred, meta_path)
    save_graphics(model, X_test, y_test, y_pred, meta_path)


# https://github.com/a18499/KubAnomaly_DataSet/blob/master/KubAnomaly_Paper.py#L157
#  Kubaniomaly model
def make_model_mlp3(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    # make one hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model_path = meta_path.replace(".txt", ".keras")
    # TODO: if _best exists use it instead
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = keras.models.load_model(model_path)
    else:
        print("Creating new MLP model...")
        # Reshape data for CNN input (assuming 1D data per sample)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        model = keras.Sequential(
            [
                layers.Dense(64, activation="elu", input_shape=(X_train.shape[1],)),
                layers.Dropout(0.4),
                layers.Dense(32, activation="elu"),
                layers.Dropout(0.4),
                layers.Dense(16, activation="elu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),  # multiclass
            ]
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.F1Score(average="macro", threshold=None, name="f1_macro", dtype=None)],
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_path.replace(".keras", "_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=128,
            validation_data=(X_test, y_test),
            callbacks=[reduce_lr, early_stopping, model_checkpoint],
            # class_weight=class_weights_dict,#TODO:good enough?
        )
        # print histroy in graph validation loss and training loss
        accuracy = history.history["accuracy"]
        val_accuracy = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        f1_macro = history.history["f1_macro"]
        val_f1_macro = history.history["val_f1_macro"]
        epochs = range(len(accuracy))

        plt.plot(epochs, accuracy, "b", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.savefig(meta_path + "training_validation_accuracy.png", bbox_inches="tight")
        plt.close()

        plt.plot(epochs, f1_macro, "b", label="Training f1_macro")
        plt.plot(epochs, val_f1_macro, "r", label="Validation f1_macro")
        plt.title("Training and validation f1_macro")
        plt.legend()
        plt.savefig(meta_path + "training_validation_f1_macro.png", bbox_inches="tight")
        plt.close()

        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.savefig(meta_path + "training_validation_loss.png", bbox_inches="tight")
        plt.close()

        # Save the model
        print(f"Saving model to {model_path}")
        model.save(model_path)

    # Reshape test data if needed (when loading a model)
    if X_test.ndim == 2:
        X_test = np.expand_dims(X_test, axis=-1)

    model.summary()

    save_model_summary(model, meta_path)

    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred, digits=4))
    save_classification_report(y_true, y_pred, meta_path)

    save_graphics(model, X_test, y_test, y_pred, meta_path)


# zscore better than minmax here
def make_model_cnn(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    # make one hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model_path = meta_path.replace(".txt", ".keras")
    # TODO: if _best exists use it instead
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = keras.models.load_model(model_path)
    else:
        print("Creating new CNN model...")
        # Reshape data for CNN input (assuming 1D data per sample)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        model = keras.Sequential(
            [
                layers.Conv1D(
                    filters=16, kernel_size=5, input_shape=(X_train.shape[1], 1), padding="same", activation="elu"
                ),  # (,1) = 1 channel weil ein wert nur eine value ist nicht (r,g,b) ode rso
                layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="elu"),
                layers.MaxPooling1D(pool_size=2),
                # layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(32, activation="elu"),
                layers.Dropout(0.3),
                layers.Dense(16, activation="elu"),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation="softmax"),  # multiclass
            ]
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.F1Score(average="macro", threshold=None, name="f1_macro", dtype=None)],
        )

        # Train model using rollwindow!
        # 128 from kubanomaly => 0.6 macro @ 10ep,  @ 30ep = 0.66
        # 512 @ 10ep = 0.54, 2048 @ 30ep = 0.69
        # epochs was 10. kubanomaly = 30
        # keras tut =2048batch
        # No Roll Windows. 2048 @ 30ep = 0.66
        # print(class_weights_dict)

        # ohne windows schon 0.75 @30ep zscore a_split
        # exit(99)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_path.replace(".keras", "_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )
        fittime = time_ns()
        history = model.fit(
            X_train,
            y_train,
            epochs=30,  # was 30
            batch_size=128,
            validation_data=(X_test, y_test),
            callbacks=[reduce_lr, early_stopping, model_checkpoint],
            # class_weight=class_weights_dict,#TODO:good enough? makes worse
        )
        fittime = time_ns() - fittime
        write_to_meta_file(meta_path, "fit_time " + str(fittime))
        # print histroy in graph validation loss and training loss
        accuracy = history.history["accuracy"]
        val_accuracy = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        f1_macro = history.history["f1_macro"]
        val_f1_macro = history.history["val_f1_macro"]
        epochs = range(len(accuracy))

        plt.plot(epochs, accuracy, "b", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.savefig(meta_path + "training_validation_accuracy.png", bbox_inches="tight")
        plt.close()

        plt.plot(epochs, f1_macro, "b", label="Training f1_macro")
        plt.plot(epochs, val_f1_macro, "r", label="Validation f1_macro")
        plt.title("Training and validation f1_macro")
        plt.legend()
        plt.savefig(meta_path + "training_validation_f1_macro.png", bbox_inches="tight")
        plt.close()

        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.savefig(meta_path + "training_validation_loss.png", bbox_inches="tight")
        plt.close()

        # Save the model
        print(f"Saving model to {model_path}")
        model.save(model_path)

    # Reshape test data if needed (when loading a model)
    if X_test.ndim == 2:
        X_test = np.expand_dims(X_test, axis=-1)

    model.summary()

    save_model_summary(model, meta_path)

    # Evaluate model
    evaltime = time_ns()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    evaltime = time_ns() - evaltime
    write_to_meta_file(meta_path, "eval_time " + str(evaltime))
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred, digits=4))
    save_classification_report(y_true, y_pred, meta_path)

    save_graphics(model, X_test, y_test, y_pred, meta_path)


# linear svm one vs rest
# https://scikit-learn.org/stable/modules/svm.html#multi-class-classification
# https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/
def make_model_svm(data: tuple, meta_path: str):
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    model_path = meta_path.replace(".txt", ".pickle")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model = load(f)
    else:
        print("Creating new linear SVM model...")
        # model = OneVsRestClassifier(LinearSVC(class_weight="balanced"), n_jobs=4, verbose=1)
        # model = SVC(class_weight="balanced", probability=True, random_state=42, kernel="rbf") #no
        # use BaggingClassifier with SVC
        # n_estimators = 10
        # model = BaggingClassifier(  # ~11min
        #     estimator=SVC(class_weight="balanced", probability=True, random_state=42, kernel="linear", verbose=True),
        #     n_estimators=n_estimators,
        #     max_samples=1.0 / n_estimators,
        #     max_features=1.0,
        #     bootstrap=True,
        #     n_jobs=6,
        #     verbose=1,
        # )
        model = SVC(class_weight="balanced", probability=True, random_state=42, kernel="linear", verbose=True)  # RUn ml.py WITH -m sklearnex !!!
        fittime = time_ns()
        model.fit(X_train, y_train)
        fittime = time_ns() - fittime
        write_to_meta_file(meta_path, "fit_time " + str(fittime))
        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)

    evaltime = time_ns()
    y_pred = model.predict(X_test)
    evaltime = time_ns() - evaltime
    write_to_meta_file(meta_path, "eval_time " + str(evaltime))

    print(classification_report(y_test, y_pred, digits=4))
    save_classification_report(y_test, y_pred, meta_path)

    save_graphics(model, X_test, y_test, y_pred, meta_path)


def main():
    parser = argparse.ArgumentParser(description="ML")
    parser.add_argument(
        "--attacks",
        type=str,
        help="File path to optimized all_datasets_*.csv. Or all in one file.",
        required=True,
    )
    parser.add_argument("--benign", type=str, help="File path to optimized benign_dataset.csv", required=False, default="")
    parser.add_argument(
        "--output",
        type=str,
        help="Destination folder path for datasets",
        required=True,
    )
    parser.add_argument(
        "--rolling",
        type=str,
        help="Number of rolling windows: 2 = 20ms window = 2 samples window per node. Recommended: 2,5,10,50,100",
        default="0",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Models to run",
        default="cnn,rf,knn,nb,gb,svm",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        help="Normalization types: no,minmax,zscore",
        default="minmax,zscore",
    )
    parser.add_argument(
        "--oversample",
        type=str,
        help="Oversampling types: no,smote",
        default="no",
    )
    parser.add_argument(
        "--podlogs",
        type=str,
        help="Podlogs types",
        default="no,count,tfidf",
    )
    parser.add_argument(
        "--force",
        type=lambda x: x.lower() == "true",
        help="Force overwrite exting meta files (re-run again)",
        default=False,
    )
    parser.add_argument(
        "--retrain-model",
        type=lambda x: x.lower() == "true",
        help="Train model even if model file exists",
        default=False,
    )
    parser.add_argument(
        "--recreate-dataset",
        type=lambda x: x.lower() == "true",
        help="Force recreate dataset",
        default=False,
    )
    parser.add_argument(
        "--cv",
        type=str,
        help="Cross-validation strategy: 'no' for no CV, 'time' for time-based CV, or 'kfold' CV",
        default="no",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        help="Number of folds for cross-validation (default: 5)",
        default=5,
    )
    parser.add_argument(
        "--attacks-include-windows",
        type=lambda x: x.lower() == "true",
        help="skip rolling window creation if ds already include rolling windows",
        default=False,
    )
    parser.add_argument(
        "--attacks-include-normalize",
        type=lambda x: x.lower() == "true",
        help="skip rolling window creation if ds already include normalization. dont normalize again",
        default=True,
    )
    parser.add_argument(
        "--skip99",  # has no effect on dataset_labeled_net2* because between is already =0 benign
        type=lambda x: x.lower() == "true",
        help="time between attacks as benign (false) or ignore (true)",
        default=True,
    )
    parser.add_argument("--experiment", type=str, help="master1,worker1,worker2,nonet,nosys,noprom,onlyprom,onlynet,onlysys,binary", default="")

    args = parser.parse_args()
    # attacks
    print(f"Input file: {args.attacks}")
    attacks_csv = os.path.abspath(args.attacks)
    benign_csv = os.path.abspath(args.benign) if args.benign != "" else ""
    DESTROOT = os.path.abspath(args.output)
    os.makedirs(DESTROOT, exist_ok=True)
    print(f"Destination folder: {DESTROOT}")

    rolling_window_sizes = [int(x) for x in args.rolling.split(",")]
    models = args.models.split(",")
    normalize_types = args.normalize.split(",")
    podlogs_types = args.podlogs.split(",")
    oversample_types = args.oversample.split(",")
    print(f"Rolling windows: {rolling_window_sizes}")
    print(f"Models: {models}")
    print(f"Normalization types: {normalize_types}")
    print(f"Podlogs types: {podlogs_types}")
    print(f"Oversample types: {oversample_types}")

    attacks_include_windows = args.attacks_include_windows
    if attacks_include_windows:
        print("Skipping rolling window creation as dataset already includes them.")

    for model in models:
        for normalize_type in normalize_types:
            for oversample_type in oversample_types:
                for podlogs_type in podlogs_types:
                    # if model == "svm" and normalize_type == "zscore":
                    #     print(f"( Temp skip model {model} norm {normalize_type} )")
                    #     # ConvergenceWarning: Liblinear failed to converge, increase the number of iterations. when zscore
                    #     print(" !! force minmax for svm (was zscore)")
                    #     normalize_type = "minmax"
                    #     # continue

                    cv_suffix = ""
                    if args.cv != "no":
                        cv_type = "time" if args.cv == "time" else "kfold"
                        cv_suffix = f"-{cv_type}cv{args.cv_folds}"

                    exp_suffix = ""
                    if args.experiment:
                        exp_suffix = f"-exp_{args.experiment}"

                    id = (
                        f"{'skip99' if args.skip99 else '99'}_{model}-rollwindow_"
                        + "+".join(str(x) for x in rolling_window_sizes)
                        + f"-norm_{normalize_type}-logs_{podlogs_type}-oversmpl_{oversample_type}{cv_suffix}{exp_suffix}"
                    )
                    os.makedirs(os.path.join(DESTROOT, model, id), exist_ok=True)
                    meta_path = os.path.join(
                        DESTROOT,
                        model,
                        id,
                        id + ".txt",
                    )
                    print(
                        f"Model: {model}, Window: {rolling_window_sizes}, Normalize: {normalize_type}, Podlogs: {podlogs_type}, Oversample: {oversample_type}"
                    )
                    if os.path.exists(meta_path) and not args.force:
                        print(f"Model {meta_path} already exists. Skipping...")
                        continue

                    datasettempfile = os.path.join(
                        DESTROOT,
                        ".cache",
                        f"{'skip99' if args.skip99 else '99'}_rollwindow_"
                        + "+".join(str(x) for x in rolling_window_sizes)
                        + f"-norm_{normalize_type}-logs_{podlogs_type}-oversmpl_{oversample_type}{cv_suffix}{exp_suffix}.dataset.pickle",
                    )
                    os.makedirs(os.path.dirname(datasettempfile), exist_ok=True)
                    if not os.path.exists(datasettempfile) or args.recreate_dataset:
                        split_type = "random" if args.cv == "no" else "cv"
                        data = make_dataset(
                            attacks_csv=attacks_csv,
                            benign_csv=benign_csv,
                            normalize_type=normalize_type,
                            podlogs_type=podlogs_type,
                            rolling_windows=rolling_window_sizes,
                            oversample=oversample_type,
                            split_type=split_type,
                            attack_already_include_windows=attacks_include_windows,
                            attacks_include_normalize=args.attacks_include_normalize,
                            skip99=args.skip99,
                            experiment=args.experiment,
                        )
                        with open(datasettempfile, "wb") as f:
                            dump(data, f, protocol=5)
                    else:
                        print(f"Loading existing dataset from {datasettempfile}")
                        with open(datasettempfile, "rb") as f:
                            data = load(f)

                    if args.retrain_model:
                        print("(--retrain-model) Deleting model if exists...")
                        model_path = meta_path.replace(".txt", ".pickle")
                        os.remove(model_path) if os.path.exists(model_path) else None
                        model_path = meta_path.replace(".txt", ".keras")
                        os.remove(model_path) if os.path.exists(model_path) else None

                    os.remove(meta_path) if os.path.exists(meta_path) else None

                    model_func = globals()[f"make_model_{model}"]
                    if args.cv == "time":
                        raise NotImplementedError("Time-series cross-validation is not implemented yet.")
                        # Perform time-series cross-validation
                        print("Performing time-series cross-validation...")
                        cv_results = perform_time_series_cv(model_func, data, meta_path, n_splits=args.cv_folds)

                        # Write summary to meta file
                        with open(meta_path, "w") as f:
                            f.write(f"Time Series {args.cv_folds}-Fold Cross-Validation Summary\n")
                            f.write("=" * 60 + "\n\n")
                            f.write(f"Average Accuracy: {cv_results['avg_metrics']['accuracy']:.4f} ± {cv_results['std_metrics']['accuracy']:.4f}\n")
                            f.write(
                                f"Average Balanced Accuracy: {cv_results['avg_metrics']['balanced_accuracy']:.4f} ± {cv_results['std_metrics']['balanced_accuracy']:.4f}\n"
                            )
                            f.write(f"Average Precision: {cv_results['avg_metrics']['precision']:.4f} ± {cv_results['std_metrics']['precision']:.4f}\n")
                            f.write(f"Average Recall: {cv_results['avg_metrics']['recall']:.4f} ± {cv_results['std_metrics']['recall']:.4f}\n")
                            f.write(f"Average F1 Score: {cv_results['avg_metrics']['f1']:.4f} ± {cv_results['std_metrics']['f1']:.4f}\n")
                    # Replace the existing kfold CV implementation in main()
                    elif args.cv == "kfold":
                        # Perform standard k-fold cross-validation
                        k_folds = int(args.cv_folds)
                        print(f"Performing {k_folds}-fold standard cross-validation...")
                        cv_results = perform_kfold_cv(model_func, data, meta_path, n_splits=k_folds)

                        # Write summary to meta file
                        with open(meta_path, "w") as f:
                            f.write(f"Stratified {k_folds}-Fold Cross-Validation Summary\n")
                            f.write("=" * 60 + "\n\n")
                            f.write(f"Average Accuracy: {cv_results['avg_metrics']['accuracy']:.4f} ± {cv_results['std_metrics']['accuracy']:.4f}\n")
                            f.write(
                                f"Average Balanced Accuracy: {cv_results['avg_metrics']['balanced_accuracy']:.4f} ± {cv_results['std_metrics']['balanced_accuracy']:.4f}\n"
                            )
                            f.write(f"Average Precision: {cv_results['avg_metrics']['precision']:.4f} ± {cv_results['std_metrics']['precision']:.4f}\n")
                            f.write(f"Average Recall: {cv_results['avg_metrics']['recall']:.4f} ± {cv_results['std_metrics']['recall']:.4f}\n")
                            f.write(f"Average F1 Score: {cv_results['avg_metrics']['f1']:.4f} ± {cv_results['std_metrics']['f1']:.4f}\n")
                    else:
                        # Run traditional single train/test split
                        print(f"Dataset created. Running make_model_{model}...")
                        print("Running traditional train/test split (no CV)...")
                        model_func(data, meta_path)

                    print(f"Meta saved => {meta_path}")


if __name__ == "__main__":
    main()
