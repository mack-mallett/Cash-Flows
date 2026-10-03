"""
EntryEntry Point for running experiments
"""
#General
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
#Data Processing
from custom_pipeline.pipelines import experiments
from sklearn.model_selection import StratifiedKFold
from data_mix_setup import equal_portions
#Classifiers
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import MultinomialNB
#Pipelines
from custom_pipeline.incremental_learning_pipelines import IncrementalXGBoostClassifier, IncrementalSGDClassifier, IncrementalKMeansClassifier, IncrementalMultinomialNBClassifier
#Plotting
import matplotlib.pyplot as plt


# Data Import
user_data = Path.cwd() / "userData" / "Mack"
X = pd.read_csv(
    user_data / 'GL_2lvl.csv',
    usecols=['Date', 'Location', 'Tag1', 'Credit', 'Debit','Balance'],
    parse_dates=['Date']
    )
#Experiment Parameters
num_tests = 100 #number of times each model is run
experiments = experiments #pipelines to test
random_state = 0 #only for k-fold
n_splits = 10 #for k-fold
how_to_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#Data Splitting
y = X['Tag1']
X = X.drop('Tag1', axis=1)
## Data mix for experiments:
X_train_init, y_train_init, subsequent_X_batches, subsequent_y_batches, _ = equal_portions(X, y)
user_labels = list(np.unique(y_train_init))

#Report on Label Distribution in Each Batch
progress = y_train_init.value_counts(normalize=True).rename("Batch_0")
batches = [progress]
for b, batch_y in enumerate(subsequent_y_batches):
    add = batch_y.value_counts(normalize=True).rename(f"Batch_{b + 1}")
    batches.append(add)
progress = pd.concat(batches, axis=1, )
progress = progress * 100
print(progress.head(50))

ax = progress.T.plot.line(figsize=(10,6))
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.xlabel('Batch Number')
plt.ylabel("Label Percent of Batch Total")
plt.title("Proportion of Label in Each Batch")
plt.tight_layout()
results_dir = Path.cwd() / 'experiments' / 'results' / 'equal_chunk2'
results_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(results_dir / 'Label_Proportion_Over_Time.png')
plt.close()

for prpp_name, preprocessor in experiments.items():
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
                'alpha': list(np.logspace(-5, 3, num=10)),
                'penalty':['l2', 'l1', 'elasticnet', None]
                },
            #Cross Validation Params
            'cv':how_to_fold,
            'scoring':None, 
        },
        sgd_classifier=SGDClassifier(
            loss='log_loss',
            penalty='l2',
            random_state=None,
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
                'alpha': list(np.logspace(-5, 3, num=10)),
                'penalty':['l2', 'l1', 'elasticnet', None]
                },
            #Cross Validation Params
            'cv':how_to_fold,
            'scoring':None, 
        },
        sgd_classifier=SGDClassifier(
            loss='hinge',
            penalty='l2',
            random_state=None,
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
            'scoring':None, 
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
        preprocessor=preprocessor,
        GridSearchCV_kwargs={
            'verbose':1,
            'param_grid':
                    {
                'alpha': [0.2, 0.4, 0.6, 0.8, 1],
                'fit_prior':[True, False]
                },
            #Cross Validation Params
            'cv':how_to_fold,
            'scoring': None,
        },
        nb_classifier=MultinomialNB(

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
                'subsample': [0.1, 0.2, 0.6, 0.8],        
                'colsample_bytree': [0.1, 0.2, 0.6, 0.8], 
            
            # 4. Strict regularization to stop structural overfit
                'reg_lambda': [1.0, 10.0],    # Focus on higher penalties to keep tree weights small.
                'min_child_weight': [1, 5]       # CRITICAL: Forces leaves to require more samples to split.
                },
            #Cross Validation Params
            'cv':how_to_fold,
            'scoring':None,
        },
        xgboost_model=xgb.XGBClassifier(
            objective='multi:softprob',
            tree_method='hist',
            n_jobs=-1,
            max_depth=3,
            random_state=None,
        )
    )

    #Experiment Boilerplate
    # classifiers = {'xgboost':xgboost_classifier}
    classifiers = {'naive_bayes':naive_bayes_classifier, 'k_means':k_means_classifier, 'logistic':logistic_classifier, 'svm':svm_classifier, 'xgboost':xgboost_classifier}
    
    #Experiment Loop
    for key, clf in classifiers.items():
        try:
            sresults_dir = results_dir / f"{prpp_name}" / f"{key}"
            sresults_dir.mkdir(parents=True, exist_ok=True)

            print(f"Starting tests for {key} at {datetime.now().strftime('%H:%M:%S')}")
            #train inital batch
            print(f"training initial {key} at {datetime.now().strftime('%H:%M:%S')}")
            clf_params = Path.cwd() / 'experiments' / 'results' / 'equal_chunk2' / f"{prpp_name}" / f"{key}" / f"{key}_best_params.csv"
            if clf_params.is_file():
                print(f"Initialize {key} from existing parameters.")
                clf.init_from_params(clf_params)
            else:
                clf.param_tune_CV()
                cv_results = pd.DataFrame(clf.grid_CV.cv_results_)
                
                #Save CV fitting results and best parameters
                cv_results.to_csv(sresults_dir / f"{key}_cv_results.csv")
                pd.DataFrame(clf.best_params_, index=[0]).to_csv(sresults_dir / f"{key}_best_params.csv") #type:ignore

            for test in range(num_tests):
                #Train Initial Batch
                clf.train_init_batch()

                #predict results on initial test batch and train batch.
                print(f"test {test}: predict initial accuracy for {key} at {datetime.now().strftime('%H:%M:%S')}")
                initial_test_accuracy = clf.accuracy_report(
                    next_batch_X=subsequent_X_batches[0], 
                    next_batch_y=subsequent_y_batches[0]
                    )
                initial_train_accuracy = clf.accuracy_report(
                    next_batch_X=X_train_init, 
                    next_batch_y=y_train_init
                    )

                #initial_test_accuracy is saved twice to allow incremental and baseline streams to be analyzed seperately
                #Save incremental learning train and test results
                initial_train_accuracy.to_csv(sresults_dir / f"{key}_inc_test{test}_batch0_train_accuracy.csv")
                initial_test_accuracy.to_csv(sresults_dir / f"{key}_inc_test{test}_batch0_test_accuracy.csv")

                #Save baseline model test results
                initial_test_accuracy.to_csv(sresults_dir / f"{key}_base_test{test}_batch0_test_accuracy.csv")

                #repeat prcess for subsequent batches
                for i, (X_b, y_b) in enumerate(zip(subsequent_X_batches, subsequent_y_batches), start=0):
                    batch_n = i+1
                    total_batches = len(subsequent_y_batches)
                    print(f"test {test}: training batch {batch_n}/{total_batches} for {key} at {datetime.now().strftime('%H:%M:%S')}")

                    if i + 1 < len(subsequent_X_batches):
                        #train on an incoming batch
                        clf.train_subsequent_batches(X=X_b, y=y_b)

                        #evaluate training error for incremental model
                        sub_train_accuracy = clf.accuracy_report(next_batch_X=X_b, next_batch_y=y_b)
                        #Save results
                        sub_train_accuracy.to_csv(sresults_dir / f"{key}_inc_test{test}_batch{batch_n}_train_accuracy.csv")
                        #record test results

                        #test incremental model
                        sub_test_accuracy = clf.accuracy_report(
                            next_batch_X=subsequent_X_batches[i+1], 
                            next_batch_y=subsequent_y_batches[i+1]
                            )
                        #Save incremental results
                        sub_test_accuracy.to_csv(sresults_dir / f"{key}_inc_test{test}_batch{batch_n}_test_accuracy.csv")
                        
                        base_test_accuracy = clf.accuracy_report(
                            next_batch_X=subsequent_X_batches[i+1], 
                            next_batch_y=subsequent_y_batches[i+1], 
                            incremental=False
                            )
                        #Save baseline results
                        base_test_accuracy.to_csv(sresults_dir / f"{key}_base_test{test}_batch{batch_n}_test_accuracy.csv")

        except Exception as e:
            print(f"An error occurred testing {key} on pipeline: {prpp_name}: {e}")