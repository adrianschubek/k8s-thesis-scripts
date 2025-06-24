### copy to ml.py as needed

############# CNN Tuning
def build_tunable_cnn_model(hp, input_shape, num_classes):
    """Build a tunable CNN model with hyperparameters defined by Keras Tuner."""
    model = keras.Sequential()

    hp.Int("batch_size", min_value=64, max_value=512, step=64)

    # First Conv1D layer with tunable filters and kernel size
    model.add(
        layers.Conv1D(
            filters=hp.Int("conv1_filters", min_value=16, max_value=128, step=16),
            kernel_size=hp.Choice("conv1_kernel", values=[3, 5, 7]),
            activation=hp.Choice("conv1_activation", values=["relu", "tanh", "elu"]),
            input_shape=input_shape,
        )
    )

    # MaxPooling with tunable pool size
    model.add(layers.MaxPooling1D(pool_size=hp.Int("pool1_size", min_value=2, max_value=4, step=1)))

    # Optional second Conv1D layer
    if hp.Boolean("use_conv2"):
        model.add(
            layers.Conv1D(
                filters=hp.Int("conv2_filters", min_value=32, max_value=256, step=32),
                kernel_size=hp.Choice("conv2_kernel", values=[3, 5]),
                activation=hp.Choice("conv2_activation", values=["relu", "tanh", "elu"]),
            )
        )
        model.add(layers.MaxPooling1D(pool_size=hp.Int("pool2_size", min_value=2, max_value=4, step=1)))

    model.add(layers.Flatten())

    # Tunable Dense layers
    for i in range(hp.Int("num_dense_layers", 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Int(f"dense_{i}_units", min_value=32, max_value=512, step=32), activation=hp.Choice(f"dense_{i}_activation", ["relu", "tanh", "elu"])
            )
        )

        # Apply dropout with tunable rate
        # model.add(layers.Dropout(rate=hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)))

    # add 0.5 dropout layer
    model.add(layers.Dropout(rate=0.5))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile with tunable learning rate and optimizer
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def make_model_cnn_tuned(data: tuple, meta_path: str):
    # !!!
    # requires pip install tensorflow
    # !!!
    X_train, X_test, y_train, y_test, num_classes, class_weights_dict = data

    # Convert to one-hot encoding
    y_train_categorical = to_categorical(y_train, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)

    # Reshape data for CNN input
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Check if model already exists
    model_path = meta_path.replace(".txt", ".keras")
    tuner_dir = meta_path.replace(".txt", "_tuner")

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = keras.models.load_model(model_path)
    else:
        print("Starting hyperparameter tuning for CNN model...")
        start_time = time()

        # Define the tuner
        tuner = kt.BayesianOptimization(
            lambda hp: build_tunable_cnn_model(hp, input_shape=(X_train.shape[1], 1), num_classes=num_classes),
            objective="val_accuracy",
            max_trials=25,  # Adjust based on your compute resources
            directory=tuner_dir,
            project_name="cnn_tuning",
            overwrite=True,
            seed=42,
            executions_per_trial=1,  # To save time, can be increased for better reliability
        )

        # Display search space summary
        tuner.search_space_summary()

        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        # Search for best hyperparameters
        tuner.search(X_train, y_train_categorical, epochs=10, batch_size=256, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights_dict)

        # Get best hyperparameters and build model
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Save best hyperparameters to meta file
        with open(meta_path, "a") as f:
            f.write("Best Hyperparameters:\n")
            f.write(str(best_hp.values) + "\n\n")

        # Build model with best hyperparameters
        model = build_tunable_cnn_model(best_hp, input_shape=(X_train.shape[1], 1), num_classes=num_classes)

        # Train the model with the best hyperparameters
        print("Training final model with best hyperparameters...")
        model.fit(
            X_train,
            y_train_categorical,
            epochs=10,
            batch_size=best_hp.get("batch_size"),
            validation_split=0.2,
            callbacks=[early_stopping],
            class_weight=class_weights_dict,
        )

        # Save the model
        model.save(model_path)

        # Log tuning time
        tuning_time = time() - start_time
        with open(meta_path, "a") as f:
            f.write(f"Hyperparameter tuning took {tuning_time:.2f} seconds\n")
            f.write(f"Tuning completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Model summary
    model.summary()
    save_model_summary(model, meta_path)

    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test_categorical, axis=1)

    # Save classification report
    print(classification_report(y_true, y_pred, digits=4))
    save_classification_report(y_true, y_pred, meta_path)

    # Save model graphics
    save_graphics(model, X_test, y_test_categorical, y_pred, meta_path)


#############
