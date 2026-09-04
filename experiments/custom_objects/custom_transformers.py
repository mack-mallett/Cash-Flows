"""
Stateless preprocessing transformers to prep transaction data for classification.
Transformers are compatible with sklearn.pipeline.Pipeline. 
All include a .partial_fit() method to be compatible with skpartial.pipeline PartialPipeline wrapper for Pipeline.
OneHotTransformer is unfinished, as new categories are problamatic in incremental learning. Will have to rely on e-transfer markings in 'Location' feature.
"""
#general data transformation libraries
import pandas as pd
import numpy as np
#base classes for estimators
from sklearn.base import BaseEstimator, TransformerMixin
#Testing
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import validate_data, check_is_fitted  # pyright: ignore[reportAttributeAccessIssue]

# class FactorizeLabels(TransformerMixin, BaseEstimator):
#     def __init__(self) -> None:
#         super().__init__()

#     def fit(self, X, y = None):
#         """Basic validation, for API convention."""
#         X = validate_data(self, X, accept_sparse=False)
#         self.categories_ = []
#         for col in range(X.shape[1]):
#             _, uniques = pd.factorize(X[:, col])
#             self.categories_.append(uniques)
#         self.n_features_in_ = X.shape[1]
#         return self
    
#     def partial_fit(self, X, y=None, **kwargs):
#         """Allows compatibility with skpartial by calling fit on incoming batches."""
#         return self.fit(X, y)

#     def transform(self, X):
#         """Factorize Budget Labels. TODO: save label names somewhere."""
#         X = validate_data(self, X, accept_sparse=False, reset=False)
#         check_is_fitted(self)
#         X_out = np.empty(X.shape, dtype=int)
#         for col in range(X.shape[1]):
#             # X_mod, _ = pd.factorize(X[:, col])
#             cat = pd.Categorical(X[:, col], categories=self.categories_[col])
#             X_out[:,col] = cat.codes
#         return X_out

#     def get_feature_names_out(self, input_features=None):
#       check_is_fitted(self)
#       if input_features is None:
#         # Fallback to feature indices matching fitted input shape
#         return np.array(
#             [f'x{i}' for i in range(self.n_features_in_)], dtype=object
#         )
#       return np.asarray(input_features, dtype=object)

class CurrencyBasics(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y = None):
        """Basic validation, for API convention. Set n_features_in_ for get_feature_names_out() (Note: this feature is unused)"""
        X = validate_data(self, X, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        return self
    
    def partial_fit(self, X, y=None, **kwargs):
        """Allows compatibility with skpartial by calling fit on incoming batches."""
        return self.fit(X, y)

    def transform(self, X):
        """Fill missing values and remove values below 0. For safety."""
        X = validate_data(self, X, accept_sparse=False, reset=False)
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.fillna(0).clip(lower=0)

    def get_feature_names_out(self, input_features=None):
      check_is_fitted(self)
      if input_features is None:
        # Fallback to feature indices matching fitted input shape
        return np.array(
            [f'x{i}' for i in range(self.n_features_in_)], dtype=object
        )
      return np.asarray(input_features, dtype=object)

class DateParsing(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__() #type:ignore
        # Tells check_estimator that this transformer expects string/categorical inputs
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        return tags
    
    def fit(self, X, y = None):
        """Basic validation, for API convention."""
        X = validate_data(self, X, accept_sparse=False, dtype=None)
        return self

    def partial_fit(self, X, y=None, **kwargs):
        """Allows compatibility with skpartial by calling fit on incoming batches."""
        return self.fit(X, y)

    def transform(self, X):
        """From a date column parse month and day columns"""
        X = validate_data(self, X, accept_sparse=False, reset=False, dtype=None)
        check_is_fitted(self)
        extracted = []
        for col in range(X.shape[1]):
            dt_series = pd.to_datetime(pd.Series(X[:, col]), errors='coerce')

            # Extract month and day, filling invalid/missing parses with 0
            extracted.append(dt_series.dt.month.fillna(0).to_numpy())
            extracted.append(dt_series.dt.day.fillna(0).to_numpy())

            return np.column_stack(extracted)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
          # Fallback default if input_features aren't provided
          return np.array(['month', 'day'], dtype=object)

        feature_names = []
        for feature in input_features:
          feature_names.append(f'{feature}_month')
          feature_names.append(f'{feature}_day')
        return np.array(feature_names, dtype=object)

class YearCycle(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__() #type:ignore
        # Tells check_estimator that this transformer expects string/categorical inputs
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        return tags

    def fit(self, X, y = None):
        """Basic validation, for API convention."""
        X = validate_data(self, X, accept_sparse=False, dtype=None)
        return self

    def partial_fit(self, X, y=None, **kwargs):
        """Allows compatibility with skpartial by calling fit on incoming batches."""
        return self.fit(X, y)

    def transform(self, X):
        """From a date column produce the Sine and CoSine position of the month in the year."""
        X = validate_data(self, X, accept_sparse=False, reset=False, dtype=None)
        check_is_fitted(self)
        extracted = []
        for col in range(X.shape[1]):
            dt_series = pd.to_datetime(pd.Series(X[:, col]), errors='coerce')
            extracted.append(np.sin(2 * np.pi * dt_series.dt.month / 12))
            extracted.append(np.cos(2 * np.pi * dt_series.dt.month / 12))
        return np.column_stack(extracted)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
          # Fallback default if input_features aren't provided
          return np.array(['monthSin', 'monthCos'], dtype=object)

        feature_names = []
        for feature in input_features:
          feature_names.append(f'{feature}_monthSin')
          feature_names.append(f'{feature}_monthCos')
        return np.array(feature_names, dtype=object)

class MonthCycle(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y = None):
        """Basic validation, for API convention."""
        X = validate_data(self, X, accept_sparse=False, dtype=None)
        return self

    def partial_fit(self, X, y=None, **kwargs):
        """Allows compatibility with skpartial by calling fit on incoming batches."""
        return self.fit(X, y)

    def transform(self, X):
        """From a date column produce the Sine and CoSine position of the day in the month."""
        X = validate_data(self, X, accept_sparse=False, reset=False, dtype=None)
        check_is_fitted(self)
        extracted = []
        for col in range(X.shape[1]):
            dt_series = pd.to_datetime(pd.Series(X[:, col]), errors='coerce')
            extracted.append(np.sin(2 * np.pi * dt_series.dt.day / 31))
            extracted.append(np.cos(2 * np.pi * dt_series.dt.day / 31))
        return np.column_stack(extracted)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
          # Fallback default if input_features aren't provided
          return np.array(['daySin', 'dayCos'], dtype=object)

        feature_names = []
        for feature in input_features:
          feature_names.append(f'{feature}_daySin')
          feature_names.append(f'{feature}_dayCos')
        return np.array(feature_names, dtype=object)

class Log1pTransformer(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__() #type:ignore
        # Tells check_estimator to pass strictly positive numbers during checks
        tags.input_tags.positive_only = True
        return tags
    
    def fit(self, X, y = None):
        """Basic validation, for API convention. Set n_features_in_ for get_feature_names_out() (Note: this feature is unused)"""
        X = validate_data(self, X, accept_sparse=False)
        if (X < 0).any():
            raise ValueError("Negative values in data")
        self.n_features_in_ = X.shape[1]
        return self

    def partial_fit(self, X, y=None, **kwargs):
        """Allows compatibility with skpartial by calling fit on incoming batches."""
        return self.fit(X, y)

    def transform(self, X):
        """Applies log1p transformation to skewed numerical features."""
        X = validate_data(self, X, accept_sparse=False, reset=False)
        check_is_fitted(self)
        if (X < 0).any():
            raise ValueError("Negative values in data")
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
      check_is_fitted(self)
      if input_features is None:
        # Fallback to feature indices matching fitted input shape
        return np.array(
            [f'x{i}' for i in range(self.n_features_in_)], dtype=object
        )
      return np.asarray(input_features, dtype=object)



#Skip for now since I don't have a good way to handle new categories in incremental processing.
class OneHotTransformer(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y = None):
        """Basic validation, for API convention."""
        X = validate_data(self, X, accept_sparse=True)
        return self

    def transform(self, X):
        """Applies log1p transformation to skewed numerical features."""
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return np.log1p(X)

if __name__ == '__main__':
    my_estimators = [
        # FactorizeLabels(), 
        CurrencyBasics(), 
        DateParsing(),
        YearCycle(),
        MonthCycle(),
        Log1pTransformer(),
        ]
    for estimator in my_estimators:
        results = check_estimator(estimator, on_fail="warn") #type:ignore
        for test in results:
            print(f"\n{test}\n")