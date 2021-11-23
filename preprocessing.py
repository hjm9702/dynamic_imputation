import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import poisson



def missing_value_generator(X, missing_rate, seed):
    row_num = X.shape[0]
    column_num = X.shape[1]
    missing_value_average_each_row = column_num * (missing_rate/100)

    np.random.seed(seed)
    poisson_dist = poisson.rvs(mu = missing_value_average_each_row, size = row_num, random_state = seed)
    poisson_dist = np.clip(poisson_dist, 0, X.shape[1]-1)
    
    column_idx = np.arange(column_num)
    X_missing = X.copy().astype(float)
    for i in range(row_num):
        missing_idx = np.random.choice(column_idx, poisson_dist[i], replace=False)
        for j in missing_idx:
            X_missing[i,j] = np.nan
    
    return X_missing


def preprocessing(x_trnval, x_tst, y_trnval, y_tst, missing_rate, seed):
    
    x_trnval = missing_value_generator(x_trnval, missing_rate, seed)
    x_tst_missing = missing_value_generator(x_tst, missing_rate, seed)
    
    scaler_x = StandardScaler()
    
    x_trnval = scaler_x.fit_transform(x_trnval)
    x_tst = scaler_x.transform(x_tst)
    x_tst_missing = scaler_x.transform(x_tst_missing)
    
    if len(np.unique(y_trnval)) > 2:
        enc = OneHotEncoder(sparse=False)
        y_trnval = enc.fit_transform(y_trnval.reshape(-1,1))
        y_tst = enc.fit_transform(y_tst.reshape(-1,1))
    
    else:
        y_trnval = y_trnval.reshape(-1,1)
        y_tst = y_tst.reshape(-1,1)
        
    return x_trnval, x_tst, x_tst_missing, y_trnval, y_tst
