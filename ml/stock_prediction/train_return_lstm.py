from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "TensorFlow is not installed in the current environment. "
        "Install it in the project environment before running this return-target LSTM training script."
    ) from exc

X_TRAIN_PATH = Path("data/processed/return_X_train.npy")
Y_TRAIN_PATH = Path("data/processed/return_y_train.npy")
X_VAL_PATH = Path("data/processed/return_X_val.npy")
Y_VAL_PATH = Path("data/processed/return_y_val.npy")
MODEL_PATH = Path("ml/stock_prediction/return_lstm_stock_model.keras")
SEQUENCE_LENGTH = 60
FEATURE_COUNT = 13


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return np.load(path)


def validate_inputs(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D; got shape {X_train.shape}.")
    if X_train.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(f"X_train shape mismatch: expected {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {X_train.shape[1:]}.")
    if len(y_train) != len(X_train):
        raise ValueError(f"y_train length does not match X_train: {len(y_train)} vs {len(X_train)}.")

    if X_val.ndim != 3:
        raise ValueError(f"X_val must be 3D; got shape {X_val.shape}.")
    if X_val.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(f"X_val shape mismatch: expected {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {X_val.shape[1:]}.")
    if len(y_val) != len(X_val):
        raise ValueError(f"y_val length does not match X_val: {len(y_val)} vs {len(X_val)}.")

    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("NaN values found in training arrays.")
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        raise ValueError("NaN values found in validation arrays.")
    if np.isinf(X_train).any() or np.isinf(y_train).any():
        raise ValueError("Infinite values found in training arrays.")
    if np.isinf(X_val).any() or np.isinf(y_val).any():
        raise ValueError("Infinite values found in validation arrays.")


def build_model() -> Model:
    inputs = Input(shape=(SEQUENCE_LENGTH, FEATURE_COUNT))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def main() -> None:
    X_train = load_array(X_TRAIN_PATH)
    y_train = load_array(Y_TRAIN_PATH)
    X_val = load_array(X_VAL_PATH)
    y_val = load_array(Y_VAL_PATH)

    validate_inputs(X_train, y_train, X_val, y_val)

    model = build_model()
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        shuffle=False,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    final_epoch_count = len(history.history["loss"])
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    print(f"training samples: {len(X_train)}")
    print(f"validation samples: {len(X_val)}")
    print(f"input shape: {X_train.shape[1:]}")
    print(f"epochs actually completed: {final_epoch_count}")
    print(f"final training loss: {final_train_loss}")
    print(f"final validation loss: {final_val_loss}")
    print(f"saved model path: {MODEL_PATH}")


if __name__ == "__main__":
    main()
