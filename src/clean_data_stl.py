from statsmodels.tsa.seasonal import MSTL
import numpy as np
import pandas as pd

def clean_data_stl(df: pd.DataFrame,
                   DAY_LENGTH: int,
                   WEEK_LENGTH: int,
                   )-> pd.Series:
    '''
    Removing outliers that lie in 3x standard deviation range for each stock.
    '''
    for column in df.columns:
        stock = df[column]
        mstl = MSTL(stock, periods=(DAY_LENGTH*WEEK_LENGTH, DAY_LENGTH*WEEK_LENGTH*12))
        res = mstl.fit()

        cleaned_data = res.trend + (res.seasonal['seasonal_600'] - res.seasonal['seasonal_50'])

        resid = res.resid

        res_mean = resid.mean()
        resid_std = resid.std()

        lower_bound = res_mean - 3 * resid_std
        upper_bound = res_mean + 3 * resid_std

    # Replace the outlier with the mean of neighboring values
        for i in range(len(cleaned_data)):
            if resid[i] < lower_bound or resid[i] > upper_bound:
                
                resid[i] = 0

        df[column] = cleaned_data + resid
    return df