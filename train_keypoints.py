
"""Retrain keypoint classifier from CSV.

For trustworthy evaluation, collect multiple independent sessions per class
and pass --session-dir path/to/sessions (each session in its own CSV).
Without independent sessions, a stratified frame split is a smoke test only.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "hand-gesture-recognition-mediapipe" / "model" / "keypoint_classifier"
LABELS = ['Crouch', 'Jump', 'Neutral', 'LeftSpin', 'RightSpin']


def load_table(path):
    table = np.loadtxt(path, delimiter=',', dtype=np.float32, ndmin=2)
    if table.shape[1] != 43:
        raise ValueError(f'{path}: expected 43 CSV columns (class ID + 42 features), got {table.shape[1]}')
    if not np.all(np.isfinite(table)):
        raise ValueError(f'{path}: non-finite values')
    targets = table[:, 0].astype(np.int32)
    if not np.all(table[:, 0] == targets) or not np.all((targets >= 0) & (targets < len(LABELS))):
        raise ValueError(f'{path}: class IDs must be integers 0..4')
    return table[:, 1:].astype(np.float32), targets


def data_split(args):
    if args.session_dir:
        files = sorted(Path(args.session_dir).glob('*.csv'))
        if len(files) < 10:
            raise ValueError('Provide at least 10 independent session files, with more than one per gesture.')
        Xs, ys, groups = [], [], []
        for idx, path in enumerate(files):
            x, y = load_table(path)
            Xs.append(x)
            ys.append(y)
            groups.append(np.full(len(y), idx))
        X, y = np.vstack(Xs), np.concatenate(ys)
        group_ids = np.concatenate(groups)
        split = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=42)
        chosen = None
        for train_i, test_i in split.split(X, y, group_ids):
            if set(y[train_i]) == set(range(5)) and set(y[test_i]) == set(range(5)):
                chosen = train_i, test_i
                break
        if chosen is None:
            raise ValueError('Sessions cannot be split with all five classes in each portion. Record more sessions per class.')
        train_i, test_i = chosen
        print('Session-disjoint validation enabled (recording files do not overlap).')
        return X[train_i], X[test_i], y[train_i], y[test_i]
    X, y = load_table(MODEL_DIR / 'keypoint.csv')
    print('WARNING: Random frame split only. Validation may be greatly optimistic!')
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-dir', default=None, help='Folder with separate per-session CSV recordings')
    parser.add_argument('--epochs', type=int, default=250)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    X_train, X_test, y_train, y_test = data_split(args)
    print('Training counts:', dict(zip(*np.unique(y_train, return_counts=True))))
    print('Validation counts:', dict(zip(*np.unique(y_test, return_counts=True))))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(len(LABELS), activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    best_path = MODEL_DIR / 'keypoint_classifier_best.keras'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
    ]
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs,
              batch_size=64, callbacks=callbacks, class_weight=weight_map, verbose=2)
    model = tf.keras.models.load_model(best_path)
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    print(classification_report(y_test, preds, labels=list(range(5)), target_names=LABELS, zero_division=0))
    print('Confusion matrix:\n', confusion_matrix(y_test, preds, labels=list(range(5))))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    out_path = MODEL_DIR / 'keypoint_classifier_candidate.tflite'
    out_path.write_bytes(converter.convert())
    print(f'Candidate saved: {out_path}')
    print('Keep your existing .tflite intact until live testing confirms this candidate is better.')

if __name__ == '__main__':
    main()

