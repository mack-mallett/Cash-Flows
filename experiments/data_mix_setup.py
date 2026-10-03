import numpy as np

def equal_portions(X, y):#, user_labels, n_splits):
    """
    Splits the data into 10 equal sized portions.
    Each chunk contains at least as many examples of each class as n_splits.
    """
    chunk_indices = np.array_split(np.arange(len(X)), 10)

    X_train_init = X.iloc[chunk_indices[0]]
    y_train_init = y.iloc[chunk_indices[0]]

    subsequent_X_batches = [X.iloc[idx] for idx in chunk_indices[1:]]
    subsequent_y_batches = [y.iloc[idx] for idx in chunk_indices[1:]]

    return X_train_init, y_train_init, subsequent_X_batches, subsequent_y_batches, chunk_indices
