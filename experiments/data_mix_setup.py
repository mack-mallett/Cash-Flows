from custom_pipeline.inject_dummy_rows import inject_dummy_rows
from sklearn.model_selection import train_test_split
import numpy as np

def large_initial_training(X, y, user_labels, n_splits):
    """
    Splits the data into an inital training set of 50% of data and 5 subsequent batches each with 10% of the data.
    Each chunk contains at least as many examples of each class as n_splits.
    """
    X_initial, X_remain, y_initial, y_remain = train_test_split(X, y, test_size=0.5)
    chunk_indices = np.array_split(np.arange(len(X_remain)), 5)
    # Initial chunk
    X_train_init, y_train_init = inject_dummy_rows(
        X_initial, 
        y_initial, 
        user_labels, 
        n_splits=n_splits
    )

    # Subsequent batches
    subsequent_X_batches = []
    subsequent_y_batches = []

    for idx in chunk_indices:
        X_b, y_b = inject_dummy_rows(X.iloc[idx], y.iloc[idx], user_labels, n_splits=n_splits)
        subsequent_X_batches.append(X_b)
        subsequent_y_batches.append(y_b)

    return X_train_init, y_train_init, subsequent_X_batches, subsequent_y_batches

def equal_portions(X, y, user_labels, n_splits):
    """
    Splits the data into 10 equal sized portions.
    Each chunk contains at least as many examples of each class as n_splits.
    """
    chunk_indices = np.array_split(np.arange(len(X)), 10)
    # Initial chunk
    X_train_init, y_train_init = inject_dummy_rows(
        X.iloc[chunk_indices[0]], 
        y.iloc[chunk_indices[0]], 
        user_labels, 
        n_splits=n_splits
    )

    # Subsequent batches
    subsequent_X_batches = []
    subsequent_y_batches = []

    for idx in chunk_indices[1:]:
        X_b, y_b = inject_dummy_rows(X.iloc[idx], y.iloc[idx], user_labels, n_splits=n_splits)
        subsequent_X_batches.append(X_b)
        subsequent_y_batches.append(y_b)

    chunk_indices = np.array_split(np.arange(len(X)), 10)

    X_train_init = X.iloc[chunk_indices[0]]
    y_train_init = y.iloc[chunk_indices[0]]

    subsequent_X_batches = [X.iloc[idx] for idx in chunk_indices[1:]]
    subsequent_y_batches = [y.iloc[idx] for idx in chunk_indices[1:]]

    return X_train_init, y_train_init, subsequent_X_batches, subsequent_y_batches
