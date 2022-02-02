from feature_engine.creation import CyclicalTransformer
import numpy as np
import pandas as pd

#======================================================
# PREPROCESS CLASS
#======================================================
class Preprocess:
    def __init__(self, gdp_path, easter, special_days, country_holidays):
        self.gdp_path = gdp_path     
        self.easter = easter  
        self.special_days = special_days
        self.country_holidays = country_holidays
    
    def import_data(self, path):
        self.df = pd.read_csv(path, parse_dates=['date'], index_col=0)
        self.index = self.df.index
        
    def get_gdp(self, path):
        gdp = pd.read_csv(path)
        gdp.columns = gdp.columns.str.title().str.replace('Gdp_','')
        gdp = gdp.set_index('Year').stack(0).reset_index()
        gdp.columns = ['year','country','gdp']
        return gdp

    def create_date_features(self, df):
        df['day_of_year'] = df.date.dt.day_of_year
        df['day_of_month'] = df.date.dt.day
        df['day_of_week'] = df.date.dt.weekday
        df['week_of_year'] = df.date.dt.isocalendar().week.astype('int64')
        df['year'] = df.date.dt.year
        return df
    
    def add_country_holidays(self, df):
        years = [2015, 2016, 2017, 2018, 2019]
        
        finland = self.country_holidays['finland']
        norway = self.country_holidays['norway']
        sweden = self.country_holidays['sweden']
        
        return (df.merge(finland, on = ['date', 'country'], how = 'left')
                        .merge(norway, on = ['date', 'country'], how = 'left')
                        .merge(sweden, on = ['date', 'country'], how = 'left'))

    def get_specific_dates_features(self, df):
        
        return (df.assign(easter = lambda x: x.year.map(self.easter).astype('datetime64'),
                            moms_day = lambda x: x.year.map(self.special_days['moms_day']).astype('datetime64'),
                            wed_jun = lambda x: x.year.map(self.special_days['wed_june']).astype('datetime64'),
                            sun_nov = lambda x: x.year.map(self.special_days['sun_nov']).astype('datetime64'),
                            
                            days_from_easter = lambda x: (x.date - x.easter).dt.days.clip(-5, 65),
                            days_from_mom = lambda x: (x.date - x.moms_day).dt.days.clip(-1, 9),
                            days_from_wed = lambda x: (x.date - x.wed_jun).dt.days.clip(-5, 5),
                            days_from_sun = lambda x: (x.date - x.sun_nov).dt.days.clip(-1, 9),
            )).drop(columns = ['easter','moms_day','wed_jun','sun_nov'])
        
    def join_gdp(self, df):
        return df.merge(self.gdp, on = ['country','year'], how = 'left')

    def feature_engineering(self, df):
        df['log_gdp'] = np.log(df.gdp)
        days_cats = [df.day_of_week < 4, df.day_of_week == 4, df.day_of_week > 4]
        days = ['week','friday','weekends']
        df['week'] = np.select(days_cats, days)
        return df
    
    def __call__(self, path, name = 'Train'):
        self.import_data(path)
        self.gdp = self.get_gdp(self.gdp_path).set_index('year')
        out = (self.df.pipe(self.create_date_features)
                    .pipe(self.join_gdp)
                    .pipe(self.feature_engineering)
                    .pipe(self.add_country_holidays)
                    .pipe(self.get_specific_dates_features))
        out.index = self.index
        print(f'{name} Set created with {out.shape[1]} features')
        return out


#======================================================
# UTILS
#======================================================

class CyclicalTransformerV2(CyclicalTransformer):
    def __init__(self, suffix = None, **kwargs):
            super().__init__(**kwargs)
            self.suffix = suffix
        
    def transform(self, X):
        X = super().transform(X)
        if self.suffix is not None:
            transformed_names = X.filter(regex = r'sin$|cos$').columns
            new_names = {name: name + self.suffix for name in transformed_names}
            X.rename(columns=new_names, inplace=True)
        return X
    

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    return np.mean(numerator / denominator)*100

def cv_trainer(model, X,y, X_test = None, folds = None):
    
    S_train = pd.DataFrame(np.nan, index = range(len(X)), columns = ['fold','num_sold'])
    S_test = {}
    
    scores = dict(train = [],
                val = [])
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        
        X_train, y_train = X.iloc[train_idx],y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx],y.iloc[val_idx]
        
        id_train = X_train.index
        id_val = X_val.index
        id_test = X_test.index
        
        model.fit(X_train, y_train)
        y_pred_train = pd.Series(model.predict(X_train), index = id_train, name = 'num_sold')
        y_pred_val = pd.Series(model.predict(X_val), index = id_val, name = 'num_sold')
        y_pred_test = pd.Series(model.predict(X_test), index = id_test, name = 'num_sold')
        
        S_train.loc[val_idx, 'num_sold'] = y_pred_val
        S_train.loc[val_idx, 'fold'] = fold
        
        scores['train'].append(smape(y_train, y_pred_train))
        scores['val'].append(smape(y_val, y_pred_val))
        
        S_test[fold] = y_pred_test
        # print(f'Fold {fold}')
        # print('Train Score: ', smape(y_train, y_pred_train))
        # print('Val Score: ', smape(y_val, y_pred_val))
        
    S_test = pd.DataFrame(S_test).mean(axis = 1).to_frame()
    S_test.columns = ['num_sold']
    
    return scores, S_train, S_test

