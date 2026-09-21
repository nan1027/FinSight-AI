from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dropout, Dense, LSTM
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "TensorFlow is not installed in the current environment. "
        "Install it in the project environment before running this LSTM training script."
    ) from exc


X_TRAIN_PATH = Path("data/processed/stock_X_train.npy")
Y_TRAIN_PATH = Path("data/processed/stock_y_train.npy")
X_VAL_PATH = Path("data/processed/stock_X_val.npy")
Y_VAL_PATH = Path("data/processed/stock_y_val.npy")
X_TEST_PATH = Path("data/processed/stock_X_test.npy")
Y_TEST_PATH = Path("data/processed/stock_y_test.npy")
MODEL_PATH = Path("ml/stock_prediction/lstm_stock_model.keras")
SEQUENCE_LENGTH = 60
FEATURES = 1


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return np.load(path)


def main() -> None:
    X_train = load_array(X_TRAIN_PATH)
    y_train = load_array(Y_TRAIN_PATH)
    X_val = load_array(X_VAL_PATH)
    y_val = load_array(Y_VAL_PATH)
    X_test = load_array(X_TEST_PATH)
    y_test = load_array(Y_TEST_PATH)

    X_train = X_train.reshape(X_train.shape[0], SEQUENCE_LENGTH, FEATURES)
    X_val = X_val.reshape(X_val.shape[0], SEQUENCE_LENGTH, FEATURES)
    X_test = X_test.reshape(X_test.shape[0], SEQUENCE_LENGTH, FEATURES)

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
    )

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

    print(f"training samples: {X_train.shape[0]}")
    print(f"validation samples: {X_val.shape[0]}")
    print(f"test samples: {X_test.shape[0]}")
    print(f"input shape: {X_train.shape[1:]}" )
    print(f"epochs actually completed: {final_epoch_count}")
    print(f"final training loss: {final_train_loss}")
    print(f"final validation loss: {final_val_loss}")
    print(f"saved model path: {MODEL_PATH}")


if __name__ == "__main__":
    main()
