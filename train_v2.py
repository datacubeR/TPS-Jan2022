import os

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import data as d
from models import HybridModel
from preprocessing import preprocess_dict
from utils import Preprocess, cv_trainer

#======================================================
# IMPORTING DATA
#======================================================


GDP_PATH = '../inputs/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv'
TRAIN_PATH = '../inputs/train.csv'
TEST_PATH = '../inputs/test.csv'
feature_engineer = Preprocess(GDP_PATH, d.EASTER_DICT, d.SPECIAL_DAYS_DICT, d.COUNTRY_HOLIDAYS_DICT, )
train_df = feature_engineer(TRAIN_PATH)
test_df = feature_engineer(TEST_PATH, name = 'Test')

#======================================================
# PREPARING FOR MODELLING
#======================================================

X = train_df.drop(columns = ['gdp','date','num_sold'])
y = train_df.num_sold
X_test = test_df.drop(columns = ['gdp','date'])

#======================================================
# MODEL DEFINITION
#======================================================z

ridge = Pipeline(steps = [
    ('prep', preprocess_dict['linear_v1']),
    ('model', LinearRegression())
])

xgb = Pipeline(steps = [
    ('prep', preprocess_dict['boosting_v1']),
    ('model', LGBMRegressor())
])

model = HybridModel(ridge, xgb)
folds = GroupKFold(n_splits=4).split(X = train_df, groups=train_df.year) # not sure if it's ok

#======================================================
# TRAINING
#======================================================

scores, S_train, S_test = cv_trainer(model, X,y, X_test, folds)

#======================================================
# REPORTING RESULTS AND EXPORTING
#======================================================

model_name = '_'.join([type(model).__name__, type(model.l_model.named_steps.model).__name__, type(model.b_model.named_steps.model).__name__])
print(f'Training results for {model_name}')
print(scores)
print('Mean Training Score: ', np.mean(scores['train']))
print('Mean Validaton Score: ', np.mean(scores['val']))

S_test.to_csv(f'../submissions/submission_{model_name}_v2.csv')

#======================================================
# Saving Predictions for Future
#======================================================


def save_predictions(preds, model_name, mode = 'train'):
    path = f'../submissions/Stacking_{mode}.csv'
    if os.path.exists(path):
        input = pd.read_csv(path, index_col=0)
        input[f'{model_name}'] = preds['num_sold'].sort_index()
        input.to_csv(path)
    else:
        out = preds['num_sold']
        out.name = model_name
        out.to_csv(path)

save_predictions(S_train, model_name)
save_predictions(S_test, model_name, mode = 'test')



