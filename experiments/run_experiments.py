"""
Entry Point for running experiments
"""
#General
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
#Data Processing
from custom_pipeline.pipelines import preprocessor
from sklearn.model_selection import train_test_split, StratifiedKFold
#Classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
#Pipelines
# from experiments.custom_objects.inc_logistic_clf import IncramentalSGDClassifier
from custom_pipeline.incremental_learning_pipelines import IncrementalXGBoostClassifier, IncrementalSklearnClassifier

#Data Import
user_data = Path.cwd() / "userData" / "Mack"
X = pd.read_csv(
    user_data / 'GL_2lvl.csv',
    usecols=['Date', 'Location', 'Tag1', 'Credit', 'Debit', 'Source','Balance', 'E_Transfer'],
    parse_dates=['Date']
    )
#Experiment Parameters
random_state = 0
how_to_fold = StratifiedKFold(n_splits=8, shuffle=True, random_state=random_state)
#try RepeatedStratifiedKFold. It might help with my exceptionally small datasets
user_labels = list(np.unique(X['Tag1']))

def inject_dummy_rows(X_chunk, y_chunk, user_labels, n_splits=8):
    """
    Ensures every class in user_labels has at least n_splits representation 
    in the chunk by appending correctly-typed dummy rows.
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

# --- Data Splitting with Injection ---

n_splits = 8
#Data Splitting
y = X['Tag1']
X = X.drop('Tag1', axis=1)

#big first portion test
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

# #equal portions test
# chunk_indices = np.array_split(np.arange(len(X)), 10)
# # Initial chunk
# X_train_init, y_train_init = inject_dummy_rows(
#     X.iloc[chunk_indices[0]], 
#     y.iloc[chunk_indices[0]], 
#     user_labels, 
#     n_splits=n_splits
# )

# # Subsequent batches
# subsequent_X_batches = []
# subsequent_y_batches = []

# for idx in chunk_indices[1:]:
#     X_b, y_b = inject_dummy_rows(X.iloc[idx], y.iloc[idx], user_labels, n_splits=n_splits)
#     subsequent_X_batches.append(X_b)
#     subsequent_y_batches.append(y_b)

# chunk_indices = np.array_split(np.arange(len(X)), 10)

# X_train_init = X.iloc[chunk_indices[0]]
# y_train_init = y.iloc[chunk_indices[0]]

# subsequent_X_batches = [X.iloc[idx] for idx in chunk_indices[1:]]
# subsequent_y_batches = [y.iloc[idx] for idx in chunk_indices[1:]]

#Setup Classifiers
logistic_classifier = IncrementalSklearnClassifier(
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
        random_state=random_state,
        max_iter=1000,
        tol=1e-4,
    )
)

svm_classifier = IncrementalSklearnClassifier(
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
        random_state=random_state,
        max_iter=1000,
        tol=1e-4,
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
        'param_grid':
                {
            'gamma': [0, 0.1, 1],
            'reg_lambda':[1e-3, 1, 10],
            'alpha': [1e-3, 1, 10]
            },
        #Cross Validation Params
        'cv':how_to_fold,
        'scoring':None, #try ‘roc_auc’, and try googling best estimation methods for my data
    },
    xgboost_model=xgb.XGBClassifier(
        objective='multi:softprob',
        tree_method='hist',
        n_jobs=-1,
        random_state=random_state
    )
)

classifiers = {'logistic':logistic_classifier, 'svm':svm_classifier, 'xgboost':xgboost_classifier}

#Experiment Loop
results_dir = Path.cwd() / 'experiments' / 'results' / 'big_init_chunk'
results_dir.mkdir(parents=True, exist_ok=True)
for key, clf in classifiers.items():
    print(f"Starting tests for {key} at {datetime.now().strftime('%H:%M:%S')}")
    #test for catastrophic forgetting add in
    # Prepare evaluation matrix: rows = training step, cols = batch evaluated
    num_batches = len(subsequent_X_batches) + 1  # Initial + subsequent
    eval_matrix = np.zeros((num_batches, num_batches))
    #train inital batch
    print(f"training initial {key}")
    clf.train_init_batch()
    #predict results on test batch
    print(f"predict initial accuracy for {key} at {datetime.now().strftime('%H:%M:%S')}")
    initial_test_accuracy, initial_confusion_matrix = clf.accuracy_report(next_batch_X=subsequent_X_batches[0], next_batch_y=subsequent_y_batches[0])
    cv_results = pd.DataFrame(clf.grid_CV.cv_results_)
    #test for catastrophic forgetting add in
    acc_0, _ = clf.accuracy_report(next_batch_X=X_train_init, next_batch_y=y_train_init)
    eval_matrix[0, 0] = acc_0.loc['weighted avg', 'f1-score']
    # Historical buffer tracking all seen data up to current step
    seen_X = [X_train_init]
    seen_y = [y_train_init]
    #Save results
    initial_test_accuracy.to_csv(results_dir / f"{key}_initial_test_accuracy.csv")
    initial_confusion_matrix.savefig(results_dir / f"{key}_initial_test_conf_matrix.png")
    cv_results.to_csv(results_dir / f"{key}_initial_cv_results.csv")
    
    #repeat prcess for subsequent batches
    # for i in range(len(subsequent_y_batches)-1):
    for i, (X_b, y_b) in enumerate(zip(subsequent_X_batches, subsequent_y_batches), start=0):
        print(f"training batch {i+1}/9 for {key} at {datetime.now().strftime('%H:%M:%S')}")
        #train on an incoming batch
        # clf.train_subsequent_batches(X=subsequent_X_batches[i], y=subsequent_y_batches[i])
        clf.train_subsequent_batches(X=X_b, y=y_b)
        #test for catastrophic forgetting add in
        seen_X.append(X_b)
        seen_y.append(y_b)
        for j in range(len(seen_X)):
            acc_j, _ = clf.accuracy_report(next_batch_X=seen_X[j], next_batch_y=seen_y[j])
            eval_matrix[i + 1, j] = acc_j.loc['weighted avg', 'f1-score']
        #record training results to monitor overfitting
        # sub_train_accuracy, sub_train_confusion_matrix = clf.accuracy_report(next_batch_X=subsequent_X_batches[i], next_batch_y=subsequent_y_batches[i])
        sub_train_accuracy, sub_train_confusion_matrix = clf.accuracy_report(next_batch_X=X_b, next_batch_y=y_b)
        #Save results
        sub_train_accuracy.to_csv(results_dir / f"{key}_batch{i}_train_accuracy.csv")
        sub_train_confusion_matrix.savefig(results_dir / f"{key}_batch{i}_train_conf_matrix.png")
        #record test results
        if i + 1 < len(subsequent_X_batches):
            sub_test_accuracy, sub_test_confusion_matrix = clf.accuracy_report(next_batch_X=subsequent_X_batches[i+1], next_batch_y=subsequent_y_batches[i+1])
            #Save results
            sub_test_accuracy.to_csv(results_dir / f"{key}_batch{i}_test_accuracy.csv")
            sub_test_confusion_matrix.savefig(results_dir / f"{key}_batch{i}_test_conf_matrix.png")

    #test for catastrophic forgetting add in
    final_step = num_batches - 1
    forgetting_per_batch = []

    for j in range(final_step):
        peak_score = np.max(eval_matrix[j:final_step+1, j]) # Best score achieved right after/near learning
        final_score = eval_matrix[final_step, j]
        forgetting_per_batch.append(peak_score - final_score)

    mean_forgetting = np.mean(forgetting_per_batch)
    print(f"Mean Catastrophic Forgetting: {mean_forgetting:.4f}")

    # Save Matrix to DataFrame
    df_matrix = pd.DataFrame(
        eval_matrix, 
        columns=[f"Batch_{j}" for j in range(num_batches)],
        index=[f"Trained_Up_To_Step_{i}" for i in range(num_batches)]
    )
    df_matrix.to_csv(results_dir / f"{key}_forgetting_matrix.csv")