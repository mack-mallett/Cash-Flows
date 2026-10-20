"""
A clone of ColumnTransformer with the same concept as skpartial.pipeline PartialPipeline. 
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

class PartialColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers):
        """
        transformers: list of tuples (name, transformer, column_names_or_indices)
        """
        self.transformers = transformers

    def partial_fit(self, X, y=None, **kwargs):
        for name, trans, cols in self.transformers:
            # Select column subset (supports DataFrame column names or array indices)
            X_cols = X[cols] if hasattr(X, "iloc") or hasattr(X, "loc") else X[:, cols]
            
            # Delegate partial_fit if supported, otherwise fall back to fit
            if hasattr(trans, "partial_fit"):
                trans.partial_fit(X_cols, y, **kwargs)
            elif hasattr(trans, "fit"):
                trans.fit(X_cols, y)
        return self

    def fit(self, X, y=None):
        return self.partial_fit(X, y)

    def transform(self, X):
        outputs = []
        for name, trans, cols in self.transformers:
            X_cols = X[cols] if hasattr(X, "iloc") or hasattr(X, "loc") else X[:, cols]
            outputs.append(trans.transform(X_cols))

        # Check if outputs contain sparse matrices (e.g., from HashingVectorizer)
        if any(sparse.issparse(o) for o in outputs):
            # Convert dense blocks to CSR matrices before stacking
            outputs_sparse = [
                sparse.csr_matrix(o) if not sparse.issparse(o) else o 
                for o in outputs
            ]
            return sparse.hstack(outputs_sparse, format="csr")
        
        return np.hstack(outputs)

    def fit_transform(self, X, y=None):
        self.partial_fit(X, y)
        return self.transform(X)