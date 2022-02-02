from feature_engine.creation import CombineWithReferenceFeature
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import CategoricalImputer
from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import CyclicalTransformerV2

preprocess_dict = {}

scaler = SklearnTransformerWrapper(StandardScaler())
cyc_1 = CyclicalTransformerV2(variables = ['day_of_year'], 
                            max_values = {'day_of_year':365}, 
                            suffix = '_1')
cyc_2 = CyclicalTransformerV2(variables = ['day_of_year'], 
                            max_values = {'day_of_year':365/2,}, 
                            suffix = '_2')
cyc_3 = CyclicalTransformerV2(variables = ['day_of_week'], 
                            max_values = {'day_of_week':7,}, 
                            suffix = '_week')
cyc_4 = CyclicalTransformerV2(variables = ['day_of_week'], 
                            max_values = {'day_of_week':7/2,}, 
                            suffix = '_semiweek')

prep = Pipeline(steps = [
    ('cat_imp', CategoricalImputer()),
    ('ohe', OneHotEncoder(drop_last=True)),
    ('cyc1', cyc_1),
    ('cyc2', cyc_2),
    ('cyc3', cyc_3),
    ('cyc4', cyc_4),
    ('combo', CombineWithReferenceFeature(variables_to_combine=['day_of_year_sin_1', 'day_of_year_cos_1',
                                                                'day_of_year_sin_2', 'day_of_year_cos_2'], 
                                        reference_variables=['product_Kaggle Mug', 'product_Kaggle Hat'],
                                        operations = ['mul'])),
    ('combo_week', CombineWithReferenceFeature(variables_to_combine=['day_of_week_sin_week', 'day_of_week_cos_week',
                                                                'day_of_week_sin_semiweek', 'day_of_week_cos_semiweek'], 
                                        reference_variables=['product_Kaggle Mug', 'product_Kaggle Hat'],
                                        operations = ['mul'])),
    ('drop', DropFeatures(features_to_drop=['year'],#'day_of_year','day_of_week','day_of_month',,]
                                            # 'store_KaggleMart','product_Kaggle Sticker', 'week_week','country_Sweden']
    )),
    ('sc', scaler)
    ])

preprocess_dict['linear_v1'] = prep


bcyc_1 = CyclicalTransformerV2(variables = ['day_of_year'], 
                            max_values = {'day_of_year':365}, 
                            suffix = '_1')
bcyc_2 = CyclicalTransformerV2(variables = ['day_of_year'], 
                            max_values = {'day_of_year':365/2,}, 
                            suffix = '_2')
bcyc_3 = CyclicalTransformerV2(variables = ['day_of_week'], 
                            max_values = {'day_of_week':7}, 
                            suffix = '_week')
bcyc_4 = CyclicalTransformerV2(variables = ['day_of_week'], 
                            max_values = {'day_of_week':7/2,}, 
                            suffix = '_semiweek')
prep = Pipeline(steps = [
    ('cat_imp', CategoricalImputer()),
    ('oe', OrdinalEncoder()),
    ('cyc1', bcyc_1),
    ('cyc2', bcyc_2),
    ('drop', DropFeatures(features_to_drop=['year']
                                            # 'store_KaggleMart','product_Kaggle Sticker', 'week_week','country_Sweden']
    )),
])

preprocess_dict['boosting_v1'] = prep




