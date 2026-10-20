"""
Incramental Learning pipeline for classifier
"""
#General
# from pandas import pd.DataFrame, pd.Series, pd.read_csv
import pandas as pd
import numpy as np
from pathlib import Path
#Data Processing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator

from skpartial.pipeline import (
    PartialPipeline,
    make_partial_pipeline,
)
from .partial_column_transformer import PartialColumnTransformer
import copy
#Classifiers
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
#Reporting
from sklearn.metrics import classification_report
#exporting 
import joblib

class BaseClassifier():
    def __init__(
            self,
            # initial_X:pd.DataFrame,
            # initial_y:pd.Series,
            # subsequent_X:list[pd.DataFrame],
            # subsequent_y:list[pd.Series],
            user_labels:list,
            preprocessor:PartialPipeline | PartialColumnTransformer,
            classifier:BaseEstimator,
            # GridSearchCV_kwargs:dict
            
    ):
        # self.initial_X = initial_X
        # self.initial_y = initial_y
        # self.subsequent_X = subsequent_X
        # self.subsequent_y = subsequent_y
        self.user_labels = user_labels
        self.user_map = {key:i for i, key in enumerate(self.user_labels)}

        self.label_encoder = LabelEncoder()

        self.raw_preprocessor = preprocessor
        self.preprocessor_pipe = make_partial_pipeline(preprocessor)
        self.clf = classifier

        # self.grid_CV = GridSearchCV(
        #     estimator=self.clf, 
        #     **GridSearchCV_kwargs,
        #     refit=True,
        #     return_train_score=True
        #     )

        #check for fitting
        self.fitted = False



    # def _prep_labels(self):
    #     """
    #     Create a set of dummy labels, map to the user_lables. Allows for 20 categories or 10 more than defined by the user, whichever is less.
    #     Fit the LabelEncoder() to the base_labels, and transform the initial labels
    #     """
    #     if len(self.user_labels) >=20:
    #         self.base_labels = list(range(0,len(self.user_labels)+10))
    #     else:
    #         self.base_labels = list(range(0,21))

    #     self.user_map = {key:i for i, key in enumerate(self.user_labels)}

    #     self.label_encoder.fit(self.base_labels)

    #     return self.label_encoder.transform(self.initial_y.map(self.user_map))
    
    def _format_X(self, X):
        """Transform X and update the transformer for next time"""
        X_trans = self.preprocessor_pipe.transform(X=X)
        self.preprocessor_pipe.partial_fit(X=X)
        return X_trans

    def _fit_prpp(self, X):
        """Fit and transform the initial dataset with PartialPipeline. Use Monkey Patch to overwrite the is_fitted parameter of the underlying Pipeline object"""
        initial_X_encoded = self.preprocessor_pipe.fit_transform(X=X)
        self.preprocessor_pipe.__sklearn_is_fitted__ = lambda: True
        return initial_X_encoded

    def _update_labels(self, y):
        """Update self.user_map and self.user_labels with previously unseen labels"""
        new_labels = y.unique()
        unseen_labels = [label for label in new_labels if label not in self.user_map.keys()]
        self.user_labels.extend(unseen_labels)
        for label in unseen_labels:
            self.user_map[label] = len(self.user_map)

    # def param_tune_CV(self):
    #     """
    #     Train Classifier on an initial dataset. Hyperparameter tuning via GridSearchCV.
    #     """
    #     #Encode the category labels, with lots of headroom for new labels
    #     self.initial_y_encoded = self._prep_labels()
    #     # Fit and transform the initial dataset. Use Monkey Patch to overwrite the is_fitted parameter of the underlying Pipeline object
    #     self.initial_X_encoded = self._fit_prpp(self.initial_X)
    #     # Find the best hyperparameters
    #     self.grid_CV.fit(self.initial_X_encoded, self.initial_y_encoded)
    #     best_estimator = self.grid_CV.best_estimator_
    #     self.best_params_ = best_estimator.get_params()

    # def init_from_params(self, params_file:Path):
    #     """Only works with one row of 'best parameters'. As there should be if run inside this envorinment."""
    #     params = pd.read_csv(params_file, index_col=0)
    #     params = params.replace({np.nan: None})
    #     params_dict = params.T.to_dict()
    #     for key in params_dict.values():
    #         self.best_params_ = key
    #     #set up labels
    #     self.initial_y_encoded = self._prep_labels()

    def acc_report(self, next_batch_y, y_pred):
        y_next_batch_trans = self.label_encoder.transform(next_batch_y.map(self.user_map))
        present_idx = np.unique(np.concatenate([y_next_batch_trans, y_pred]))
        rev_map = {i:key for key, i in self.user_map.items()}
        present_cat = [rev_map.get(i, f"Extra_Cat_{i}") for i in present_idx]
        #create_report
        report = pd.DataFrame(
            classification_report(
                y_next_batch_trans,
                y_pred,
                labels=present_idx,
                target_names=present_cat,
                output_dict=True,
                zero_division=np.nan #type:ignore #Pylance issue
            )
        )
        return report

class IncrementalMultinomialNBClassifier(BaseClassifier):
    """
    Incramental training workflow class for sklearn's SGDClassifier. Uses GridSearchCV to optimize fit around an initial training set.
    Over-saves variables for easy access during experiments.
    """
    def __init__(
            self,
            #user_data_and_lables
            # initial_X:pd.DataFrame,
            # initial_y:pd.Series,
            # subsequent_X:list[pd.DataFrame],
            # subsequent_y:list[pd.Series],
            user_labels:list,
            #preprocessing_pipeline
            preprocessor:PartialPipeline | PartialColumnTransformer,
            # GridSearchCV_kwargs:dict,
            nb_classifier:MultinomialNB,
            ) -> None:

        super().__init__(
            # initial_X=initial_X,
            # initial_y=initial_y,
            # subsequent_X=subsequent_X,
            # subsequent_y=subsequent_y,
            user_labels=user_labels,
            preprocessor=preprocessor,
            classifier=nb_classifier,
            # GridSearchCV_kwargs=GridSearchCV_kwargs
        )

        self.clf = nb_classifier

    def load_trained_model(self, path):
        load_path = path / 'multinomial_nb_model.joblib'
        self.clf = joblib.load(load_path)

    # def train_init_batch(self):
    #     """Done previous to user experience."""
    #     self.initial_y_encoded = self._prep_labels()
    #     self.initial_X_plus = self.initial_X.copy()

    #     self.initial_X_encoded = self._fit_prpp(self.initial_X_plus)

    #     clf_class = self.clf.__class__ #special to SGDClassifier
    #     self.clf = clf_class(**self.best_params_)

    #     self.clf.fit(self.initial_X_encoded, self.initial_y_encoded) #type:ignore Pylance Error

    #     self.baseline_clf = copy.deepcopy(self.clf)
    #     self.fitted = True

    def train_subsequent_batches(self, X, y):
        """
        Incramentally train the preprocessing pipeline and XGBoost tree on the next batch of user data. 
        Depending on experiment results, it may be necessary to change the xgb.Booster params to something more suitable for incramental learning.
        """
        if not self.fitted:
            raise RuntimeError("Initial Classifier has not been fit!")

        #Add new user labels to the list of user labels
        self._update_labels(y)
        y_next_batch_trans = self.label_encoder.transform(y.map(self.user_map))

        # Update the preprocessor pipe and transform the next batch of X
        X_trans = self._format_X(X=X)
        # X_trans = self._fit_prpp(X=X)
        self.clf.partial_fit(X_trans, y_next_batch_trans)

    def save_model(self, path):
        "Save the model to the users folder"
        save_path = path / 'multinomial_nb_model.joblib'
        joblib.dump(self.clf, save_path)


    def predict_new(self, next_batch_X):
        #prediction
        next_batch_X_trans = self._fit_prpp(X=next_batch_X)
        y_pred = self.clf.predict(next_batch_X_trans)
        rev_map = {i:key for key, i in self.user_map.items()}

        vectorized_reverse_map = np.vectorize(rev_map.get)
        y_pred_nl = vectorized_reverse_map(y_pred)

        # report = self.acc_report(next_batch_y, y_pred)
        return y_pred_nl