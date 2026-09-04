"""
Setup column specific preprocessing transformations. Use PartialColumnTransformer, a clone of ColumnTransformer with the same purpose as skpartial.pipeline PartialPipeline.
Contains a wrapper of FunctionTransformer with partial_fit(), satisfying PartialPipeline.
"""
#Basics
import numpy as np
#Transformers
from custom_objects.custom_transformers import CurrencyBasics, DateParsing, YearCycle, MonthCycle, Log1pTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import HashingVectorizer
#Pipelines
# from sklearn.compose import ColumnTransformer
from custom_objects.partial_column_transformer import PartialColumnTransformer
# from sklearn.pipeline import Pipeline
from skpartial.pipeline import (
    PartialPipeline,
)

class PartialFunctionTransformer(FunctionTransformer):
    def partial_fit(self, X, y=None, **kwargs):
        # FunctionTransformer is stateless, so calling fit/returning self satisfies the interface
        return self.fit(X, y)
    
date_pipeline = PartialPipeline([
    ('parsing', DateParsing()),
    ('month_in_year', YearCycle()),
    ('day_in_month', MonthCycle())
])

currency_pipeline = PartialPipeline([
    ('basics', CurrencyBasics()),
    ('log1p', Log1pTransformer()),
    ('standard_normalization', StandardScaler())
])

pos_pipeline = PartialPipeline([
    ('flatten', PartialFunctionTransformer(lambda x: np.asarray(x).ravel(), feature_names_out='one-to-one')),
    ('hash_POS_ID', HashingVectorizer(
        analyzer='char_wb',
        ngram_range=(5,6),
        n_features = 2**14 #Gemini: You should set n_features between 2**14 (16,384) and 2**18 (262,144) based on the size of your overall dataset.
    ))
])

# #this pipeline's memory contains the original budget categories
# label_pipeline = PartialPipeline([
#     ('factorize', FactorizeLabels())
# ])

# categorical_pipeline
#Column groups
DATE = ['Date']
CURRENCY = ['Credit', 'Debit']
POS = ['Location']
CATEGORICAL = ['Source', 'E_Transfer']

LABELS = ['Tag1']

preprocessor = PartialColumnTransformer(
    transformers=[
        ('date', date_pipeline, DATE),
        ('currency', currency_pipeline, CURRENCY),
        ('POS', pos_pipeline, POS),
        # ('lables', label_pipeline, LABELS)
    ],
    # verbose=True,
    # verbose_feature_names_out=True
)

