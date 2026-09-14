import pandas as pd

def inject_dummy_rows(X_chunk, y_chunk, user_labels, n_splits=8):
    """
    Ensures every class in user_labels has at least n_splits representation 
    in the chunk by appending correctly-typed dummy rows.
    Mistake here. The dummy rows are all the same, so if a real class needs more labels then it will get the same 
    rows as a generic dummy label. This will probably lower the accuracy for that class.
    """
    counts = y_chunk.value_counts()
    dummies_X, dummies_y = [], []

    for label in user_labels:
        current_count = counts.get(label, 0)
        needed = max(0, n_splits - current_count)
        
        if needed > 0:
            dummies_y.extend([label] * needed)

    if not dummies_y:
        return X_chunk.reset_index(drop=True), y_chunk.reset_index(drop=True)

    total_dummies = len(dummies_y)
    dummy_X = pd.DataFrame(index=range(total_dummies), columns=X_chunk.columns)

    for col in X_chunk.columns:
        dtype = X_chunk[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            dummy_X[col] = 0.0
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            dummy_X[col] = pd.Timestamp('1970-01-01')
        else:
            dummy_X[col] = ""

    X_augmented = pd.concat([X_chunk, dummy_X], ignore_index=True)
    y_augmented = pd.concat([y_chunk, pd.Series(dummies_y)], ignore_index=True)

    return X_augmented, y_augmented