from typing_extensions import final
import pandas as pd
import joblib
import holidays
import numpy as np
from .utils import create_date_features

model_name = 'MAE_15000_cyc2_es_cv_l2_5_seed_123'

df_test = pd.read_csv('inputs/test.csv', index_col=0, parse_dates=['date'])
id = df_test.index
df_test = create_date_features(df_test)

finland = pd.DataFrame([dict(date = date, finland_holiday = event, country= 'Finland') for date, event in holidays.Finland(years=[2015, 2016, 2017, 2018, 2019]).items()])
finland['date'] = finland['date'].astype("datetime64")
norway = pd.DataFrame([dict(date = date, norway_holiday = event, country= 'Norway') for date, event in holidays.Norway(years=[2015, 2016, 2017, 2018, 2019]).items()])
norway['date'] = norway['date'].astype("datetime64")
sweden = pd.DataFrame([dict(date = date, sweden_holiday = event.replace(", Söndag", ""), country= 'Sweden') for date, event in holidays.Sweden(years=[2015, 2016, 2017, 2018, 2019]).items() if event != 'Söndag'])
sweden['date'] = sweden['date'].astype("datetime64")

df_test = df_test.merge(finland, on = ['date', 'country'], how = 'left').merge(norway, on = ['date', 'country'], how = 'left').merge(sweden, on = ['date', 'country'], how = 'left')
prep = joblib.load(f'models/prep_{model_name}.joblib')

X = prep.fit_transform(df_test)

preds_dict = {}
for fold in range(4):
    pipe = joblib.load(f'models/{model_name}_fold_{fold}.joblib')
    preds = pipe.predict(X)
    
    preds_dict[fold] = preds
    
final_preds = pd.DataFrame(preds_dict)#.mean(axis = 1).apply(np.round).astype("int")

final_preds = (0.1 * final_preds.iloc[:,0] + 0.1 * final_preds.iloc[:,1] + 0.3 * final_preds.iloc[:,2] + 0.5* final_preds.iloc[:,3]).apply(np.round).astype("int")
final_preds.index = id
final_preds.name = 'num_sold'
print(final_preds)

final_preds.to_csv(f'submissions/{model_name}_sub_weighting.csv')



