from pathlib import Path

from appLogic.settingsClass import genericSettings
from typing import Literal, Tuple, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

settingsFolder = Path(__file__).parent.parent / "Settings" / "modelSettings"


class tfidfSettings(genericSettings):
    """Settings for tfidf vectorizer"""
    analyzer:Literal['word', 'char', 'char_wb']
    ngram_range:Tuple[int, int]
    min_df:int
    max_features:int
    sublinear_tf:bool


class KFoldSettings(genericSettings):
    """Settings for StratifiedKFold"""
    n_splits:int
    shuffle:bool
    random_state:int    


class logregSettings(genericSettings):
    """Settings for LogisticRegressionCV"""
    #Model Fit Params
    max_iter: int
    solver: Literal['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    l1_ratios: Sequence[float] 
    tol: float
    class_weight: str
    #Cross Validation Params
    Cs: int #100 in original experiment
    # cv:StratifiedKFold
    scoring:str
    refit:bool
    use_legacy_attributes:bool

class Categorizer():
    """Comes with hardcoded settings"""
    def __init__(self):
        self.tfidfSettings = tfidfSettings.load_from_yaml(settingsFolder / "tfidf.yaml")
        self.KFoldSettings = KFoldSettings.load_from_yaml(settingsFolder / "kFold.yaml")
        self.LogRegSettings = logregSettings.load_from_yaml(settingsFolder / "logisticRegressionCV.yaml")
        self.tfidf = TfidfVectorizer(
            analyzer=self.tfidfSettings.analyzer,
            ngram_range=self.tfidfSettings.ngram_range,
            min_df=self.tfidfSettings.min_df,
            max_features=self.tfidfSettings.max_features,
            sublinear_tf=self.tfidfSettings.sublinear_tf
        )
        self.model = LogisticRegressionCV(
            #Model Fit Params
            max_iter=self.LogRegSettings.max_iter,
            solver=self.LogRegSettings.solver,
            l1_ratios=self.LogRegSettings.l1_ratios, 
            tol=self.LogRegSettings.tol,
            class_weight=self.LogRegSettings.class_weight,
            #Cross Validation Params
            Cs=self.LogRegSettings.Cs, #100 in original experiment
            cv=StratifiedKFold(n_splits=self.KFoldSettings.n_splits, shuffle=self.KFoldSettings.shuffle, random_state=self.KFoldSettings.random_state),
            scoring=self.LogRegSettings.scoring,
            refit=self.LogRegSettings.refit,
            use_legacy_attributes=self.LogRegSettings.use_legacy_attributes, # type: ignore
        )
        self.scaler_debit = StandardScaler()
        self.scaler_credit = StandardScaler()
        self.dummy_cols = []

    
    def _preprocess(self, df:pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """General Pre-process transaction records."""
        df_clean = df.copy()
        df_clean['month'] = df_clean['Date'].dt.month
        df_clean['day'] = df_clean['Date'].dt.day
        df_clean['Credit'] = df_clean['Credit'].fillna(0).clip(lower=0)
        df_clean['Debit']  = df_clean['Debit'].fillna(0).clip(lower=0)

        if is_training and 'Tag1' in df_clean.columns:
            df_clean['Tag1'], self.budgetLabels = pd.factorize(df_clean['Tag1'])
        return df_clean

    def _split(self, df:pd.DataFrame) -> tuple:
        """Wrapper for sklearn train_test_split"""
        X_train, X_test, y_train, y_test = train_test_split(
            df, df['Tag1'], test_size=0.2, random_state=42
            )
        return X_train, X_test, y_train, y_test
    
    def _norm(self, df, mode='train'):
        """Normalize transaction data before catagorization"""
        df_norm = df.copy()
        if mode == 'train':
            df_tfidf = self.tfidf.fit_transform(df_norm['Location'])
            df_norm['Credit'] = self.scaler_credit.fit_transform(np.log1p(df_norm[['Credit']]))
            df_norm['Debit']  = self.scaler_debit.fit_transform(np.log1p(df_norm[['Debit']]))
            df_norm = pd.get_dummies(df_norm, columns=['Source', 'E_Transfer'], prefix=['src', 'etrf'])
            self.dummy_cols = [c for c in df_norm.columns if c.startswith('src_') or c.startswith('etrf_')]
        else:
            df_tfidf = self.tfidf.transform(df_norm['Location'])
            df_norm['Credit'] = self.scaler_credit.transform(np.log1p(df_norm[['Credit']]))
            df_norm['Debit']  = self.scaler_debit.transform(np.log1p(df_norm[['Debit']]))
            # encode test, then align to train's columns
            df_norm = pd.get_dummies(df_norm, columns=['Source', 'E_Transfer'], prefix=['src', 'etrf'])
            for col in self.dummy_cols:
                if col not in df_norm.columns:
                    df_norm[col] = False   

        df_norm['monthSin'] = np.sin(2 * np.pi * df_norm['Date'].dt.month / 12)
        df_norm['monthCos'] = np.cos(2 * np.pi * df_norm['Date'].dt.month / 12)
        df_norm['daySin']   = np.sin(2 * np.pi * df_norm['Date'].dt.day / 31)
        df_norm['dayCos']   = np.cos(2 * np.pi * df_norm['Date'].dt.day / 31)
        dense_cols = self.dummy_cols + ['Credit', 'Debit', 'monthSin', 'monthCos', 
                                        'daySin', 'dayCos']
        

        dense = df_norm.reindex(columns=dense_cols, fill_value=False)
        combined = sp.hstack([df_tfidf, sp.csr_matrix(dense.astype('float64'))])
        return combined
    
    def _fit(self, xTrainNormalized, trainingLabels):
        self.model.fit(xTrainNormalized, trainingLabels)

    def _predict(self, xTestNormalized):
        y_pred = self.model.predict(xTestNormalized)
        return y_pred
    
    def _accuracyReport(self, testLabels, testPrediction, save_plot_path):
        all_possible_labels = list(range(len(self.budgetLabels)))
        report = pd.DataFrame(
            data = classification_report(
            testLabels, 
            testPrediction, 
            target_names=self.budgetLabels, 
            labels=all_possible_labels,
            zero_division=np.nan,# type: ignore[argument-type]
            output_dict=True
            )
        )
        print("\n--- Model Classification Report ---")
        print(report.T[['precision', 'recall', 'f1-score']].round(2))
        today_str = pd.to_datetime('today').strftime('%Y-%m-%d')
        report.to_csv(save_plot_path / f"{today_str}.csv")
        fig, ax = plt.subplots(figsize=(12, 12))

        ConfusionMatrixDisplay.from_predictions(
            testLabels, 
            testPrediction, 
            labels=all_possible_labels,
            display_labels=self.budgetLabels,  
            xticks_rotation='vertical',   
            cmap='Blues',                 
            ax=ax,
            colorbar=True
        )
        plt.title("Budget Classification Confusion Matrix")
        if save_plot_path:
            plt.savefig(save_plot_path / f"{today_str}.png", bbox_inches='tight')
            print(f"[Info] Confusion Matrix graphic saved to: {save_plot_path}")
            plt.close()
        else:
            print("[Info] Close the pop-up window to return to the CLI menu.")
            plt.show()


    def fit_and_evaluate(self, gl, save_plot_path = None):
        gl_safe = gl
        glPreprocessed = self._preprocess(
            gl_safe, 
            is_training=True
        )
        glXTrain, glXTest, glyTrain, glyTest = self._split(
            df=glPreprocessed
        )
        glXTrainNormalized = self._norm(
            df=glXTrain,
            mode='train'
        )
        glXTestNormalized = self._norm(
            df=glXTest,
            mode='test'
        )
        self._fit(
            xTrainNormalized=glXTrainNormalized,
            trainingLabels=glyTrain
        )
        glPred = self._predict(
            xTestNormalized=glXTestNormalized
        )
        print(gl['Tag1'].value_counts())
        self._accuracyReport(
            testLabels=glyTest,
            testPrediction=glPred,
            save_plot_path=save_plot_path
        )

    def pred_new(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Processes fresh data using fitted states and returns original dataframe with predictions."""
        if len(self.budgetLabels) == 0:
            raise ValueError("Categorizer has not been trained yet. Run fit_and_evaluate first.")
            
        processed_new = self._preprocess(new_data, is_training=False)
        normalized_new = self._norm(processed_new, mode='test')
        
        numeric_predictions = self._predict(normalized_new)
        
        # Transform the numeric array tags back into strings (e.g., [0, 1] -> ['Groceries', 'Utilities'])
        string_predictions = self.budgetLabels[numeric_predictions]
        
        output_df = new_data.copy()
        output_df['Tag1'] = string_predictions
        return output_df