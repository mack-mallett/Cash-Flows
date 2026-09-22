"""
Entry Point for running experiments
"""
#General
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
#Data Processing
from custom_pipeline.pipelines import preprocessor, naive_bayes_preprocessor
from sklearn.model_selection import StratifiedKFold
from data_mix_setup import equal_portions, large_initial_training
#Classifiers
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import MultinomialNB
#Pipelines
from custom_pipeline.incremental_learning_pipelines import IncrementalXGBoostClassifier, IncrementalSGDClassifier, IncrementalKMeansClassifier, IncrementalMultinomialNBClassifier

#Data Import
user_data = Path.cwd() / "userData" / "Mack"
X = pd.read_csv(
    user_data / 'GL_2lvl.csv',
    usecols=['Date', 'Location', 'Tag1', 'Credit', 'Debit', 'Source','Balance', 'E_Transfer'],
    parse_dates=['Date']
    )
#Experiment Parameters
num_tests = 100
random_state = 0
n_splits = 8 # 8 is the maximum for this dataset
how_to_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
user_labels = list(np.unique(X['Tag1']))

#Data Splitting
y = X['Tag1']
X = X.drop('Tag1', axis=1)

## Data mix for experiments:
X_train_init, y_train_init, subsequent_X_batches, subsequent_y_batches = large_initial_training(X, y, user_labels, n_splits)
##uncomment this line and comment out prev line to test models on 10 equal sized increments
# X_train_init, y_train_init, subsequent_X_batches, subsequent_y_batches = equal_portions(X, y, user_labels, n_splits)

##Setup Classifiers
logistic_classifier = IncrementalSGDClassifier(
    initial_X=X_train_init,
    initial_y=y_train_init,
    subsequent_X=subsequent_X_batches,
    subsequent_y=subsequent_y_batches,
    user_labels=user_labels,
    preprocessor=preprocessor,
    GridSearchCV_kwargs={
        'verbose':1,
        'param_grid':
                {
            'sgdclassifier__alpha': list(np.logspace(-5, 3, num=10)),
            'sgdclassifier__penalty':['l2', 'l1', 'elasticnet', None]
            },
        #Cross Validation Params
        'cv':how_to_fold,
        'scoring':None, #try None, or other methods
    },
    classifier_w_partial_fit=SGDClassifier(
        loss='log_loss',
        penalty='l2',
        random_state=None,#random_state,
        max_iter=1000,
        tol=1e-4,
    )
)

svm_classifier = IncrementalSGDClassifier(
    initial_X=X_train_init,
    initial_y=y_train_init,
    subsequent_X=subsequent_X_batches,
    subsequent_y=subsequent_y_batches,
    user_labels=user_labels,
    preprocessor=preprocessor,
    GridSearchCV_kwargs={
        'verbose':1,
        'param_grid':
                {
            'sgdclassifier__alpha': list(np.logspace(-5, 3, num=10)),
            'sgdclassifier__penalty':['l2', 'l1', 'elasticnet', None]
            },
        #Cross Validation Params
        'cv':how_to_fold,
        'scoring':None, #try None, or other methods
    },
    classifier_w_partial_fit=SGDClassifier(
        loss='hinge',
        penalty='l2',
        random_state=None,#random_state,
        max_iter=1000,
        tol=1e-4,
    )
)
#KMeans and MiniBatchKMeans suffer from the phenomenon called the Curse of Dimensionality for high dimensional datasets such as text data.
#The solutions suggested by sklearn involve dimensionality reduction. Depending on the results maybe I'll put some light LSA in my pipeline
k_means_classifier = IncrementalKMeansClassifier(
    initial_X=X_train_init,
    initial_y=y_train_init,
    subsequent_X=subsequent_X_batches,
    subsequent_y=subsequent_y_batches,
    user_labels=user_labels,
    preprocessor=preprocessor,
    GridSearchCV_kwargs={
        'verbose':1,
        'param_grid':
                {
            'n_init': [1,2,3,4,5,'auto'],
            'batch_size':[2**3, 2**5, 2**7, 2**9, 2**10] #There's only 800 samples (2**9.644) in the training batch. 2**10 is default from the package.
            },
        #Cross Validation Params
        'cv':how_to_fold,
        'scoring':None, #try None, or other methods
    },
    k_means_classifier=MiniBatchKMeans(
        n_clusters=20, #I've set up the incremental_learning_pipelines to have 20 classes. Domain knowledge.
        max_iter=1000,
        random_state=None,
        tol=1e-4,
    )
)

naive_bayes_classifier = IncrementalMultinomialNBClassifier(
    initial_X=X_train_init,
    initial_y=y_train_init,
    subsequent_X=subsequent_X_batches,
    subsequent_y=subsequent_y_batches,
    user_labels=user_labels,
    preprocessor=naive_bayes_preprocessor,
    GridSearchCV_kwargs={
        'verbose':1,
        'param_grid':
                {
            'multinomialnb__alpha': [0.2, 0.4, 0.6, 0.8, 1],
            'multinomialnb__fit_prior':[True, False]
            },
        #Cross Validation Params
        'cv':how_to_fold,
        'scoring': None, #try None, or other methods
    },
    classifier_w_partial_fit=MultinomialNB(

    )
)

xgboost_classifier = IncrementalXGBoostClassifier(
    initial_X=X_train_init,
    initial_y=y_train_init,
    subsequent_X=subsequent_X_batches,
    subsequent_y=subsequent_y_batches,
    user_labels=user_labels,
    preprocessor=preprocessor,
    GridSearchCV_kwargs={
        'verbose':1,
        'param_grid':{
        # 2. Control how aggressively trees are added per batch
            'learning_rate': [0.01, 0.05, 0.1], 
        
        # 3. Prevent overfitting to tiny batch structures
            'subsample': [0.6, 0.8],        # Drop 0.2 and 0.4. Low subsamples on small batches introduce high variance.
            'colsample_bytree': [0.6, 0.8], # Re-enable this to decorrelate trees across batches.
        
        # 4. Strict regularization to stop structural overfit
            'reg_lambda': [1.0, 10.0],    # Focus on higher penalties to keep tree weights small.
            'min_child_weight': [1, 5]       # CRITICAL: Forces leaves to require more samples to split.
            },
        #Cross Validation Params
        'cv':how_to_fold,
        'scoring':None, #try ‘roc_auc’, and try googling best estimation methods for my data
    },
    xgboost_model=xgb.XGBClassifier(
        objective='multi:softprob',
        tree_method='hist',
        n_jobs=-1,
        max_depth=3,
        random_state=None,#random_state
    )
)

#Experiment Boilerplate
# classifiers = {'logistic':logistic_classifier}
classifiers = {'naive_bayes':naive_bayes_classifier, 'k_means':k_means_classifier, 'logistic':logistic_classifier, 'svm':svm_classifier, 'xgboost':xgboost_classifier}
#Experiment Loop
for key, clf in classifiers.items():
    results_dir = Path.cwd() / 'experiments' / 'results' / 'big_init_chunk_ss' / f"{key}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting tests for {key} at {datetime.now().strftime('%H:%M:%S')}")

    #train inital batch
    print(f"training initial {key} at {datetime.now().strftime('%H:%M:%S')}")
    clf_params = Path.cwd() / 'experiments' / 'results' / 'big_init_chunk_ss' / f"{key}" / f"{key}_best_params.csv"
    if clf_params.is_file():
        print(f"Initialize {key} from existing parameters.")
        clf.init_from_params(clf_params)
    else:
        clf.param_tune_CV()
        cv_results = pd.DataFrame(clf.grid_CV.cv_results_)
        #Save CV fitting results and best parameters
        cv_results.to_csv(results_dir / f"{key}_cv_results.csv")
        pd.DataFrame(clf.best_params_, index=[0]).to_csv(results_dir / f"{key}_best_params.csv")

    for test in range(num_tests):
        #Train Initial Batch
        clf.train_init_batch()

        #predict results on initial test batch and train batch.
        print(f"test {test}: predict initial accuracy for {key} at {datetime.now().strftime('%H:%M:%S')}")
        initial_test_accuracy, initial_test_confusion_matrix = clf.accuracy_report(
            next_batch_X=subsequent_X_batches[0], 
            next_batch_y=subsequent_y_batches[0]
            )
        initial_train_accuracy, initial_train_confusion_matrix = clf.accuracy_report(
            next_batch_X=X_train_init, 
            next_batch_y=y_train_init
            )

        #initial_test_accuracy is saved twice to allow incremental and baseline streams to be analyzed seperately
        #Save incremental learning train and test results
        initial_train_accuracy.to_csv(results_dir / f"{key}_inc_test{test}_batch0_train_accuracy.csv")
        initial_train_confusion_matrix.savefig(results_dir / f"{key}_inc_test{test}_batch0_train_conf_matrix.png")
        initial_test_accuracy.to_csv(results_dir / f"{key}_inc_test{test}_batch0_test_accuracy.csv")
        initial_test_confusion_matrix.savefig(results_dir / f"{key}_inc_test{test}_batch0_test_conf_matrix.png")

        #Save baseline model test results
        initial_test_accuracy.to_csv(results_dir / f"{key}_base_test{test}_batch0_test_accuracy.csv")
        initial_test_confusion_matrix.savefig(results_dir / f"{key}_base_test{test}_batch0_test_conf_matrix.png")

        #repeat prcess for subsequent batches
        for i, (X_b, y_b) in enumerate(zip(subsequent_X_batches, subsequent_y_batches), start=0):
            batch_n = i+1
            total_batches = len(subsequent_y_batches)
            print(f"test {test}: training batch {batch_n}/{total_batches} for {key} at {datetime.now().strftime('%H:%M:%S')}")

            if i + 1 < len(subsequent_X_batches):
                #train on an incoming batch
                clf.train_subsequent_batches(X=X_b, y=y_b)

                #evaluate training error for incremental model
                sub_train_accuracy, sub_train_confusion_matrix = clf.accuracy_report(next_batch_X=X_b, next_batch_y=y_b)
                #Save results
                sub_train_accuracy.to_csv(results_dir / f"{key}_inc_test{test}_batch{batch_n}_train_accuracy.csv")
                sub_train_confusion_matrix.savefig(results_dir / f"{key}_inc_test{test}_batch{batch_n}_train_conf_matrix.png")
                #record test results

                #test incremental model
                sub_test_accuracy, sub_test_confusion_matrix = clf.accuracy_report(
                    next_batch_X=subsequent_X_batches[i+1], 
                    next_batch_y=subsequent_y_batches[i+1]
                    )
                #Save incremental results
                sub_test_accuracy.to_csv(results_dir / f"{key}_inc_test{test}_batch{batch_n}_test_accuracy.csv")
                sub_test_confusion_matrix.savefig(results_dir / f"{key}_inc_test{test}_batch{batch_n}_test_conf_matrix.png")

                #test baseline model
                base_test_accuracy, base_test_confusion_matrix = clf.accuracy_report(
                    next_batch_X=subsequent_X_batches[i+1], 
                    next_batch_y=subsequent_y_batches[i+1], 
                    incremental=False
                    )
                #Save baseline results
                base_test_accuracy.to_csv(results_dir / f"{key}_base_test{test}_batch{batch_n}_test_accuracy.csv")
                base_test_confusion_matrix.savefig(results_dir / f"{key}_base_test{test}_batch{batch_n}_test_conf_matrix.png")