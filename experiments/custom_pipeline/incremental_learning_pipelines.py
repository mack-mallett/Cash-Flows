"""
Incramental learning pipelines for experiments. Both pipelines have the same core functionality and depolyment process:
 1. Declare class with desired parameters.
 2. Call train_init_batch(). This will map the user_lables to a pre-defined set of labels, either 20 or 10 more than the initial amount provided by the user. \
 GridSearchCV() is used to optimize specified hyperparameters to the initial training set. The .best_estimator_ is extracted and used for subsequent batches.
 3. Call train_subsequent_batches() with parameters specifying the incoming data.
 4. Use accuracy_report() to predict on an unseen batch and report on accuracy.
 Example: 

 ```python
 mlp_classifier = IncrementalSklearnClassifier(
    initial_X=X_train_init,
    initial_y=y_train_init,
    subsequent_X=subsequent_X_batches,
    subsequent_y=subsequent_y_batches,
    user_lables=user_labels,
    preprocessor=preprocessor,
    GridSearchCV_kwargs={
        'verbose':1,
        'param_grid':[
                {
            'mlpclassifier__alpha': list(np.logspace(-5, 3, num=10))
            },
            ],
        #Cross Validation Params
        'cv':StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        'scoring':'f1_weighted', #try None, or other methods
        'refit':True,
    },
    classifier_w_partial_fit=MLPClassifier(
        random_state=0,
        solver='adam',
        tol=1e-4,
        max_iter=1000
    )
)

#Train Initial Batch and report on accuracy using a test set

mlp_classifier.train_init_batch()
df, fig = mlp_classifier.accuracy_report(next_batch_X=subsequent_X_batches[0], next_batch_y=subsequent_y_batches[0])

#Train subsequent batches and report on accuracy

for i in range(len(subsequent_y_batches)-1):
    mlp_classifier.train_subsequent_batches(X=subsequent_X_batches[i], y=subsequent_y_batches[i])
    df, fig = mlp_classifier.accuracy_report(next_batch_X=subsequent_X_batches[i+1], next_batch_y=subsequent_y_batches[i+1])
```
"""
#General
from pandas import DataFrame, Series, read_csv
import numpy as np
from pathlib import Path
#Data Processing
# from pipelines import preprocessor
from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from skpartial.pipeline import (
    PartialPipeline,
    make_partial_pipeline,
)
from .partial_column_transformer import PartialColumnTransformer
import copy
from sklearn.base import clone
#Classifiers
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
#Reporting
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class IncrementalXGBoostClassifier():
    """
    Incramental training workflow class for XGBoost Classifier. Uses GridSearchCV to optimize fit around an initial training set.
    Over-saves variables for easy access during experiments.
    """
    def __init__(
            self,
            #user_data_and_lables
            initial_X:DataFrame,
            initial_y:Series,
            subsequent_X:list[DataFrame],
            subsequent_y:list[Series],
            user_labels:list,
            #preprocessing_pipeline
            preprocessor:PartialPipeline | PartialColumnTransformer,
            GridSearchCV_kwargs:dict,
            xgboost_model:xgb.XGBClassifier,
            ) -> None:
        self.initial_X = initial_X
        self.initial_y = initial_y
        self.subsequent_X = subsequent_X
        self.subsequent_y = subsequent_y
        self.user_labels = user_labels

        self.label_encoder = LabelEncoder()
        self.clf = xgboost_model
        self.raw_preprocessor = preprocessor
        self.preprocessor_pipe = make_partial_pipeline(preprocessor)
        self.grid_CV = GridSearchCV(
            estimator=self.clf, 
            **GridSearchCV_kwargs,
            refit=True,
            return_train_score=True
            )
        #check for fitting
        self.fitted = False
        self.n_trees = None

    def _prep_labels(self):
        """
        Create a set of dummy labels, map to the user_lables. Allows for 20 categories or 10 more than defined by the user, whichever is less.
        Fit the LabelEncoder() to the base_labels, and transform the initial labels
        """
        if len(self.user_labels) >=20:
            self.base_labels = list(range(0,len(self.user_labels)+10))
        else:
            self.base_labels = list(range(0,21))

        self.user_map = {key:i for i, key in enumerate(self.user_labels)}
        self.label_encoder.fit(self.base_labels)

        self.initial_y_encoded = self.label_encoder.transform(self.initial_y.map(self.user_map))

    def _format_X(self, X):
        """Transform X and update the transformer for next time"""
        X_trans = self.preprocessor_pipe.transform(X=X)
        self.preprocessor_pipe.partial_fit(X=X)
        return X_trans

    def param_tune_CV(self):
        """
        Train XGBoost on an initial dataset. Hyperparameter tuning via GridSearchCV. Booster is set to have dummy labels, giving room for users to add new labels via user_lables.
        """
        #Encode the category labels, with lots of headroom for new labels
        self._prep_labels()
        total_classes = len(self.base_labels)
        
        # Fit and transform the initial dataset. Use Monkey Patch to overwrite the is_fitted parameter of the underlying Pipeline object
        self.initial_X_encoded = self.preprocessor_pipe.fit_transform(X=self.initial_X)
        self.preprocessor_pipe.__sklearn_is_fitted__ = lambda: True

        # Find the best hyperparameters using sklearn wrapper
        self.grid_CV.fit(self.initial_X_encoded, self.initial_y_encoded)
        best_estimator = self.grid_CV.best_estimator_

        #Extract the best hyperparameters. Force the num_class to match the base_labels
        self.best_params_ = best_estimator.get_xgb_params()
        self.best_params_['num_class'] = total_classes
        # self.xgb_params['objective'] = 'multi:softprob' # Ensure it expects probabilities for multi-class
        
        # Extract how many trees GridSearchCV decided on (defaults to 100 if missing)
        self.n_trees = getattr(best_estimator, 'n_estimators', 30)
        if self.n_trees is None:
            self.n_trees = 30

    def init_from_params(self, params_file:Path):
        """Only works with one row of 'best parameters'. As there should be if run inside this envorinment."""
        params = read_csv(params_file, index_col=0)
        params = params.replace({np.nan: None})
        params_dict = params.T.to_dict()
        # print(params_dict)
        for key in params_dict.values():
            self.best_params_ = key
        #set up labels
        self._prep_labels()
        if self.n_trees is None:
            self.n_trees = 30

    def train_init_batch(self):
        #reset the preprocessing pipeline
        self.initial_X_encoded = self.preprocessor_pipe.fit_transform(X=self.initial_X)
        self.preprocessor_pipe.__sklearn_is_fitted__ = lambda: True

        #Now switch out of the sklearn wrapper. Declare a DMatrix.
        dtrain_init = xgb.DMatrix(self.initial_X_encoded, label=self.initial_y_encoded)
        # Train a XGBooster object on the training data. Redundent training is necessary because when GridSearhCV['refit'] = False .best_estimator_ is disabled. This redundency could be removed by finding a substitute for sklearn.
        self.booster = xgb.train(
            params=self.best_params_,
            dtrain=dtrain_init,
            num_boost_round=self.n_trees
        )
        self.baseline_clf = copy.deepcopy(self.booster)
        self.fitted = True


    def train_subsequent_batches(self, X, y):
        """
        Incramentally train the preprocessing pipeline and XGBoost tree on the next batch of user data. 
        Depending on experiment results, it may be necessary to change the xgb.Booster params to something more suitable for incramental learning.
        """
        if not self.fitted:
            raise RuntimeError("Initial Classifier has not been fit!")
            
        y_next_batch_encoded = self.label_encoder.transform(y.map(self.user_map))

        # Update the preprocessor pipe and transform the next batch of X
        X_trans = self._format_X(X=X)
        dtrain = xgb.DMatrix(X_trans, label=y_next_batch_encoded)

        # Continue boosting on top of existing trees using the saved parameters
        self.booster = xgb.train(
            params=self.best_params_,
            dtrain=dtrain,
            num_boost_round=10,
            xgb_model=self.booster 
        )

    def accuracy_report(self, next_batch_X, next_batch_y, incremental:bool=True):
        #format next_batch_X
        next_batch_X_trans = self._format_X(X=next_batch_X)
        #prediction. From XGBoost Documentation: "To have cached results for incremental prediction, please use the xgboost.Booster.predict() method instead."
        dtest = xgb.DMatrix(next_batch_X_trans)

        if incremental:
            y_prob = self.booster.predict(dtest)
        else:
            y_prob = self.baseline_clf.predict(dtest)

        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_pred = np.argmax(y_prob, axis=1)
        #get_categories_present_in_this_batch
        y_next_batch_trans = self.label_encoder.transform(next_batch_y.map(self.user_map))
        present_idx = np.unique(np.concatenate([y_next_batch_trans, y_pred]))
        rev_map = {i:key for key, i in self.user_map.items()}
        present_cat = [rev_map.get(i, f"Extra_Cat_{i}") for i in present_idx]
        #create_report
        report = DataFrame(
            classification_report(
                y_next_batch_trans,
                y_pred,
                labels=present_idx,
                target_names=present_cat,
                output_dict=True,
                zero_division=np.nan #type:ignore #Pylance issue
            )
        )

        #confusion_matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        matrix = ConfusionMatrixDisplay.from_predictions(
            y_next_batch_trans,
            y_pred,
            labels=present_idx,
            display_labels=present_cat,
            xticks_rotation='vertical',
            cmap='Blues',
            ax=ax,
            colorbar=True,
        )
        plt.title('Budget Classification Confusion Matrix (Batch)')
        plt.tight_layout()
        plt.close(fig)

        return report.T, fig

class IncrementalKMeansClassifier():
    """
    Incramental training workflow class for KNN Classifier. Uses GridSearchCV to optimize fit around an initial training set.
    Over-saves variables for easy access during experiments.
    """
    def __init__(
            self,
            #user_data_and_lables
            initial_X:DataFrame,
            initial_y:Series,
            subsequent_X:list[DataFrame],
            subsequent_y:list[Series],
            user_labels:list,
            #preprocessing_pipeline
            preprocessor:PartialPipeline | PartialColumnTransformer,
            GridSearchCV_kwargs:dict,
            k_means_classifier:MiniBatchKMeans,
            ) -> None:
        self.initial_X = initial_X
        self.initial_y = initial_y
        self.subsequent_X = subsequent_X
        self.subsequent_y = subsequent_y
        self.user_labels = user_labels

        self.label_encoder = LabelEncoder()
        self.clf = k_means_classifier
        self.raw_preprocessor = preprocessor
        self.preprocessor_pipe = make_partial_pipeline(preprocessor)
        self.grid_CV = GridSearchCV(
            estimator=self.clf, 
            **GridSearchCV_kwargs,
            refit=True,
            return_train_score=True
            )
        #check for fitting
        self.fitted = False

    def _prep_labels(self):
        """
        Create a set of dummy labels, map to the user_lables. Allows for 20 categories or 10 more than defined by the user, whichever is less.
        Fit the LabelEncoder() to the base_labels, and transform the initial labels
        """
        if len(self.user_labels) >=20:
            self.base_labels = list(range(0,len(self.user_labels)+10))
        else:
            self.base_labels = list(range(0,21))

        self.user_map = {key:i for i, key in enumerate(self.user_labels)}
        self.label_encoder.fit(self.base_labels)

        self.initial_y_encoded = self.label_encoder.transform(self.initial_y.map(self.user_map))

    def _format_X(self, X):
        """Transform X and update the transformer for next time"""
        X_trans = self.preprocessor_pipe.transform(X=X)
        self.preprocessor_pipe.partial_fit(X=X)
        return X_trans

    def param_tune_CV(self):
        """
        Train XGBoost on an initial dataset. Hyperparameter tuning via GridSearchCV. Booster is set to have dummy labels, giving room for users to add new labels via user_lables.
        """
        #Encode the category labels, with lots of headroom for new labels
        self._prep_labels()
        
        # Fit and transform the initial dataset. Use Monkey Patch to overwrite the is_fitted parameter of the underlying Pipeline object
        self.initial_X_encoded = self.preprocessor_pipe.fit_transform(X=self.initial_X) #GOOD
        self.preprocessor_pipe.__sklearn_is_fitted__ = lambda: True

        # Find the best hyperparameters using sklearn wrapper. GOOD
        self.grid_CV.fit(self.initial_X_encoded, self.initial_y_encoded)
        best_estimator = self.grid_CV.best_estimator_

        #Extract the best hyperparameters. Force the num_class to match the base_labels
        self.best_params_ = best_estimator.get_params()


    def init_from_params(self, params_file:Path):
        """Only works with one row of 'best parameters'. As there should be if run inside this envorinment."""
        params = read_csv(params_file, index_col=0)
        params = params.replace({np.nan: None})
        params_dict = params.T.to_dict()
        # print(params_dict)
        for key in params_dict.values():
            self.best_params_ = key
        #set up labels
        self._prep_labels()

    def train_init_batch(self):
        #reset the preprocessing pipeline. GOOD
        self.initial_X_encoded = self.preprocessor_pipe.fit_transform(X=self.initial_X)
        self.preprocessor_pipe.__sklearn_is_fitted__ = lambda: True

        #Now switch out of the sklearn wrapper. Declare a DMatrix. CHANGE
        #Declare a new classifier here
        self.clf = MiniBatchKMeans(**self.best_params_)
        # self.clf.set_params(**self.best_params_) #type:ignore Pylance Error
        self.clf.fit(self.initial_X_encoded)

        self.baseline_clf = copy.deepcopy(self.clf)
        self.fitted = True

    def train_subsequent_batches(self, X, y):
        """
        Incramentally train the preprocessing pipeline and XGBoost tree on the next batch of user data. 
        Depending on experiment results, it may be necessary to change the xgb.Booster params to something more suitable for incramental learning.
        """
        if not self.fitted:
            raise RuntimeError("Initial Classifier has not been fit!")
            
        y_next_batch_trans = self.label_encoder.transform(y.map(self.user_map))

        # Update the preprocessor pipe and transform the next batch of X
        X_trans = self._format_X(X=X)
        self.clf.partial_fit(X_trans, y_next_batch_trans)


    def accuracy_report(self, next_batch_X, next_batch_y, incremental:bool=True):
        #format next_batch_X
        next_batch_X_trans = self._format_X(X=next_batch_X)
        #prediction. From XGBoost Documentation: "To have cached results for incremental prediction, please use the xgboost.Booster.predict() method instead."
        # dtest = xgb.DMatrix(next_batch_X_trans)

        if incremental:
            y_pred = self.clf.predict(next_batch_X_trans)
        else:
            y_pred = self.baseline_clf.predict(next_batch_X_trans)

        # if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        #     y_pred = np.argmax(y_prob, axis=1)
        #get_categories_present_in_this_batch
        next_batch_y_encoded = self.label_encoder.transform(next_batch_y.map(self.user_map))
        present_idx = np.unique(np.concatenate([next_batch_y_encoded, y_pred]))
        rev_map = {i:key for key, i in self.user_map.items()}
        present_cat = [rev_map.get(i, f"Extra_Cat_{i}") for i in present_idx]
        #create_report
        report = DataFrame(
            classification_report(
                next_batch_y_encoded,
                y_pred,
                labels=present_idx,
                target_names=present_cat,
                output_dict=True,
                zero_division=np.nan #type:ignore #Pylance issue
            )
        )

        #confusion_matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        matrix = ConfusionMatrixDisplay.from_predictions(
            next_batch_y_encoded,
            y_pred,
            labels=present_idx,
            display_labels=present_cat,
            xticks_rotation='vertical',
            cmap='Blues',
            ax=ax,
            colorbar=True,
        )
        plt.title('Budget Classification Confusion Matrix (Batch)')
        plt.tight_layout()
        plt.close(fig)

        return report.T, fig

class IncrementalSGDClassifier():
    """
    Incramental training workflow class for sklearn's SGDClassifier. Uses GridSearchCV to optimize fit around an initial training set.
    Over-saves variables for easy access during experiments.
    """
    def __init__(
            self,
            #user_data_and_lables
            initial_X:DataFrame,
            initial_y:Series,
            subsequent_X:list[DataFrame],
            subsequent_y:list[Series],
            user_labels:list,
            #preprocessing_pipeline
            preprocessor:PartialPipeline | PartialColumnTransformer,
            GridSearchCV_kwargs:dict,
            # SGDClassifier_kwargs:dict,
            classifier_w_partial_fit,
            ) -> None:
        self.initial_X = initial_X
        self.initial_y = initial_y
        self.subsequent_X = subsequent_X
        self.subsequent_y = subsequent_y
        self.user_labels = user_labels

        self.label_encoder = LabelEncoder()
        self.clf = classifier_w_partial_fit
        # self.clf = SGDClassifier(**SGDClassifier_kwargs)
        self.raw_preprocessor = preprocessor
        self.pipe = make_partial_pipeline(preprocessor, self.clf)
        self.classifier_step_name = list(self.pipe.named_steps.keys())[-1]
        self.grid_CV = GridSearchCV(
            estimator=self.pipe, 
            **GridSearchCV_kwargs,
            return_train_score=True,
            refit=True,
            )
        #check for fitting
        self.fitted = False

    def _prep_labels(self):
        if len(self.user_labels) >=20:
            self.base_labels = list(range(0,len(self.user_labels)+10))
        else:
            self.base_labels = list(range(0,21))

        self.user_map = {key:i for i, key in enumerate(self.user_labels)}
        self.label_encoder.fit(self.base_labels)

        self.initial_y_encoded = self.label_encoder.transform(self.initial_y.map(self.user_map))
        self.subsequent_y_enc = []
        for batch in self.subsequent_y:
            encoded = self.label_encoder.transform(batch.map(self.user_map))
            self.subsequent_y_enc.append(encoded)

    def _add_dummy_cats(self):
        sgd_clf = self.pipe.named_steps[self.classifier_step_name]

        all_classes = np.array(self.base_labels)  # array([0, 1, 2, ..., 20])
        unseen_classes = np.setdiff1d(all_classes, sgd_clf.classes_)

        if len(unseen_classes) > 0:
            n_features = sgd_clf.coef_.shape[1]
            n_new = len(unseen_classes)

            # Pad weights and intercepts with zeros for dummy classes (11 to 20)
            sgd_clf.coef_ = np.vstack([sgd_clf.coef_, np.zeros((n_new, n_features))])
            sgd_clf.intercept_ = np.append(sgd_clf.intercept_, np.zeros(n_new))

            # Update internal classes array to match base_labels
            sgd_clf.classes_ = all_classes

    def param_tune_CV(self):
        #prep_lables
        self._prep_labels()
        #initial_fit
        self.grid_CV.fit(self.initial_X, self.initial_y_encoded)
        self.best_params_ = self.grid_CV.best_params_

    def init_from_params(self, params_file:Path):
        """Only works with one row of 'best parameters'. As there should be if run inside this envorinment."""
        params = read_csv(params_file, index_col=0)
        params = params.replace({np.nan: None})
        params_dict = params.T.to_dict()
        # print(params_dict)
        for key in params_dict.values():
            self.best_params_ = key
        #set up labels
        self._prep_labels()

    def train_init_batch(self):
        #add extra categories
        # self.pipe.set_params(**self.best_params_)
        # self.pipe = clone(self.grid_CV.best_estimator_)
        clf_class = self.clf.__class__
        fresh_clf = clf_class(**self.clf.get_params())
        self.pipe = make_partial_pipeline(self.raw_preprocessor, fresh_clf)

        self.pipe.set_params(**self.best_params_) #type:ignore Pylance Error

        self.pipe.fit(self.initial_X, self.initial_y_encoded) #type:ignore Pylance Error
        self._add_dummy_cats()

        self.baseline_clf = copy.deepcopy(self.pipe)
        self.fitted = True

    def train_subsequent_batches(self, X, y):
        if not self.fitted:
            raise RuntimeError("Initial Classifier has not been fit!")
        y_next_batch_encoded = self.label_encoder.transform(y.map(self.user_map))
        self.pipe.partial_fit(X, y_next_batch_encoded, self.label_encoder.classes_)

    def accuracy_report(self, next_batch_X, next_batch_y, incremental:bool=True):
        #prediction
        if incremental:
            y_pred = self.pipe.predict(next_batch_X)
        else:
            y_pred = self.baseline_clf.predict(next_batch_X)
        
        #get_categories_present_in_this_batch
        y_next_batch_trans = self.label_encoder.transform(next_batch_y.map(self.user_map))
        present_idx = np.unique(np.concatenate([y_next_batch_trans, y_pred]))
        rev_map = {i:key for key, i in self.user_map.items()}
        present_cat = [rev_map.get(i, f"Extra_Cat_{i}") for i in present_idx]
        #create_report
        report = DataFrame(
            classification_report(
                y_next_batch_trans,
                y_pred,
                labels=present_idx,
                target_names=present_cat,
                output_dict=True,
                zero_division=np.nan #type:ignore #Pylance issue
            )
        )

        #confusion_matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        matrix = ConfusionMatrixDisplay.from_predictions(
            y_next_batch_trans,
            y_pred,
            labels=present_idx,
            display_labels=present_cat,
            xticks_rotation='vertical',
            cmap='Blues',
            ax=ax,
            colorbar=True,
        )
        plt.title('Budget Classification Confusion Matrix (Batch)')
        plt.tight_layout()
        plt.close(fig)

        return report.T, fig

class IncrementalMultinomialNBClassifier():
    """
    Incramental training workflow class for sklearn's SGDClassifier. Uses GridSearchCV to optimize fit around an initial training set.
    Over-saves variables for easy access during experiments.
    """
    def __init__(
            self,
            #user_data_and_lables
            initial_X:DataFrame,
            initial_y:Series,
            subsequent_X:list[DataFrame],
            subsequent_y:list[Series],
            user_labels:list,
            #preprocessing_pipeline
            preprocessor:PartialPipeline | PartialColumnTransformer,
            GridSearchCV_kwargs:dict,
            # SGDClassifier_kwargs:dict,
            classifier_w_partial_fit,
            ) -> None:
        self.initial_X = initial_X
        self.initial_y = initial_y
        self.subsequent_X = subsequent_X
        self.subsequent_y = subsequent_y
        self.user_labels = user_labels

        self.label_encoder = LabelEncoder()
        self.clf = classifier_w_partial_fit
        # self.clf = SGDClassifier(**SGDClassifier_kwargs)
        self.raw_preprocessor = preprocessor
        self.pipe = make_partial_pipeline(preprocessor, self.clf)
        self.classifier_step_name = list(self.pipe.named_steps.keys())[-1]
        self.grid_CV = GridSearchCV(
            estimator=self.pipe, 
            **GridSearchCV_kwargs,
            return_train_score=True,
            refit=True,
            )
        #check for fitting
        self.fitted = False

    def _prep_labels(self):
        if len(self.user_labels) >=20:
            self.base_labels = list(range(0,len(self.user_labels)+10))
        else:
            self.base_labels = list(range(0,21))

        self.user_map = {key:i for i, key in enumerate(self.user_labels)}
        self.label_encoder.fit(self.base_labels)

        self.initial_y_encoded = self.label_encoder.transform(self.initial_y.map(self.user_map))
        self.subsequent_y_enc = []
        for batch in self.subsequent_y:
            encoded = self.label_encoder.transform(batch.map(self.user_map))
            self.subsequent_y_enc.append(encoded)

    def _add_dummy_cats(self):
        sgd_clf = self.pipe.named_steps[self.classifier_step_name]

        all_classes = np.array(self.base_labels)  # array([0, 1, 2, ..., 20])
        unseen_classes = np.setdiff1d(all_classes, sgd_clf.classes_)

        if len(unseen_classes) > 0:
            n_features = sgd_clf.coef_.shape[1]
            n_new = len(unseen_classes)

            # Pad weights and intercepts with zeros for dummy classes (11 to 20)
            sgd_clf.coef_ = np.vstack([sgd_clf.coef_, np.zeros((n_new, n_features))])
            sgd_clf.intercept_ = np.append(sgd_clf.intercept_, np.zeros(n_new))

            # Update internal classes array to match base_labels
            sgd_clf.classes_ = all_classes

    def param_tune_CV(self):
        #prep_lables
        self._prep_labels()
        #initial_fit
        self.grid_CV.fit(self.initial_X, self.initial_y_encoded)
        self.best_params_ = self.grid_CV.best_params_

    def init_from_params(self, params_file:Path):
        """Only works with one row of 'best parameters'. As there should be if run inside this envorinment."""
        params = read_csv(params_file, index_col=0)
        params = params.replace({np.nan: None})
        params_dict = params.T.to_dict()
        # print(params_dict)
        for key in params_dict.values():
            self.best_params_ = key
        #set up labels
        self._prep_labels()

    def train_init_batch(self):
        #add extra categories
        self.pipe.set_params(**self.best_params_)
        # self.pipe = clone(self.grid_CV.best_estimator_)
        self.pipe.set_params(**self.best_params_) #type:ignore Pylance Error
        self.pipe.fit(self.initial_X, self.initial_y_encoded) #type:ignore Pylance Error
        # self._add_dummy_cats()

        self.baseline_clf = copy.deepcopy(self.pipe)
        self.fitted = True

    def train_subsequent_batches(self, X, y):
        if not self.fitted:
            raise RuntimeError("Initial Classifier has not been fit!")
        y_next_batch_encoded = self.label_encoder.transform(y.map(self.user_map))
        self.pipe.partial_fit(X, y_next_batch_encoded)

    def accuracy_report(self, next_batch_X, next_batch_y, incremental:bool=True):
        #prediction
        if incremental:
            y_pred = self.pipe.predict(next_batch_X)
        else:
            y_pred = self.baseline_clf.predict(next_batch_X)
        
        #get_categories_present_in_this_batch
        y_next_batch_trans = self.label_encoder.transform(next_batch_y.map(self.user_map))
        present_idx = np.unique(np.concatenate([y_next_batch_trans, y_pred]))
        rev_map = {i:key for key, i in self.user_map.items()}
        present_cat = [rev_map.get(i, f"Extra_Cat_{i}") for i in present_idx]
        #create_report
        report = DataFrame(
            classification_report(
                y_next_batch_trans,
                y_pred,
                labels=present_idx,
                target_names=present_cat,
                output_dict=True,
                zero_division=np.nan #type:ignore #Pylance issue
            )
        )

        #confusion_matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        matrix = ConfusionMatrixDisplay.from_predictions(
            y_next_batch_trans,
            y_pred,
            labels=present_idx,
            display_labels=present_cat,
            xticks_rotation='vertical',
            cmap='Blues',
            ax=ax,
            colorbar=True,
        )
        plt.title('Budget Classification Confusion Matrix (Batch)')
        plt.tight_layout()
        plt.close(fig)

        return report.T, fig