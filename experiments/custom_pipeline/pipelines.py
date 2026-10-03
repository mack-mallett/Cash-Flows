"""
Setup column specific preprocessing transformations. Use PartialColumnTransformer, a clone of ColumnTransformer with the same purpose as skpartial.pipeline PartialPipeline.
Contains a wrapper of FunctionTransformer with partial_fit(), satisfying PartialPipeline.
"""
#Basics
import numpy as np
#Transformers
from .custom_transformers import CurrencyBasics, DateParsing, Log1pTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import HashingVectorizer
#Pipelines
# from sklearn.compose import ColumnTransformer
from .partial_column_transformer import PartialColumnTransformer
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
])

sscaled_currency_pipeline = PartialPipeline([
    ('basics', CurrencyBasics()),
    ('log1p', Log1pTransformer()),
    ('standard_normalization', StandardScaler())
])

mmscaled_currency_pipeline = PartialPipeline([
    ('basics', CurrencyBasics()),
    ('log1p', Log1pTransformer()), #This should help with the heavy tail
    ('min_max_norm', MinMaxScaler()) #This will probably be sensitive to the heavy tail of financial transactions
])
unscaled_currency_pipeline = PartialPipeline([
    ('basics', CurrencyBasics()),
    ('log1p', Log1pTransformer()),
])


unscaled_pos_pipeline = PartialPipeline([
    ('flatten', PartialFunctionTransformer(lambda x: np.asarray(x).ravel(), feature_names_out='one-to-one')),
    ('hash_POS_ID', HashingVectorizer(
        analyzer='char',
        ngram_range=(5,6),
        n_features = 2**14, #Gemini: You should set n_features between 2**14 (16,384) and 2**18 (262,144) based on the size of your overall dataset.,
        alternate_sign=False,
        norm=None #type:ignore #Pylance error
    ))
])

l1scaled_pos_pipeline = PartialPipeline([
    ('flatten', PartialFunctionTransformer(lambda x: np.asarray(x).ravel(), feature_names_out='one-to-one')),
    ('hash_POS_ID', HashingVectorizer(
        analyzer='char',
        ngram_range=(5,6),
        n_features = 2**14, #Gemini: You should set n_features between 2**14 (16,384) and 2**18 (262,144) based on the size of your overall dataset.,
        alternate_sign=False,
        norm='l1'
    ))
])

l2scaled_pos_pipeline = PartialPipeline([
    ('flatten', PartialFunctionTransformer(lambda x: np.asarray(x).ravel(), feature_names_out='one-to-one')),
    ('hash_POS_ID', HashingVectorizer(
        analyzer='char',
        ngram_range=(5,6),
        n_features = 2**14, #Gemini: You should set n_features between 2**14 (16,384) and 2**18 (262,144) based on the size of your overall dataset.,
        alternate_sign=False,
        norm='l2'
    ))
])

# categorical_pipeline
#Column groups
DATE = ['Date']
CURRENCY = ['Credit', 'Debit']
POS = ['Location']
CATEGORICAL = ['Source', 'E_Transfer']

LABELS = ['Tag1']
#standard scaling to dollar
dollar_pipes = {'standard_scale_dollar': sscaled_currency_pipeline, 'minmax_dollar':mmscaled_currency_pipeline, 'unscaled_dollar': unscaled_currency_pipeline}
pos_pipes = {'l1_pos': l1scaled_pos_pipeline, 'l2_pos': l2scaled_pos_pipeline, 'unscaled_pos':unscaled_pos_pipeline}
date_pipes = {'sin_cos_date': date_pipeline}

def generate_pipeline_options(date_options:dict, curr_options:dict, pos_options:dict):
    options = {}
    for a, pa in date_options.items():
        for b, pb in curr_options.items():
            for c, pc in pos_options.items():
                preprocessor = PartialColumnTransformer(
                    transformers=[
                        ('date', pa, DATE),
                        ('currency', pb, CURRENCY),
                        ('POS', pc, POS),
                    ],
                )
                options[f"{a}_{b}_{c}"] = preprocessor
    return options

experiments = generate_pipeline_options(date_pipes, dollar_pipes, pos_pipes)