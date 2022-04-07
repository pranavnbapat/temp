import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import matplotlib.dates as mdates

import datetime

import statsmodels.tsa.api as smt

from statsmodels.tsa.stattools import adfuller, kpss

from sklearn.metrics import mean_squared_error


def time_plot(data, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(x_col, y_col, data=data, ax=ax, color='mediumblue', label='Total Sales')
    
    second = data.groupby(data.date.dt.year)[y_col].mean().reset_index()
    second.date = pd.to_datetime(second.date, format='%Y')
    sns.lineplot((second.date + datetime.timedelta(6*365/12)), y_col, data=second, ax=ax, color='red', label='Mean Sales')   
    
    ax.set(xlabel = "Date", ylabel = "Sales", title = title)
    
    sns.despine()

#Converts all dates of a month to a full month by addin all the sales
def monthly_sales(data):
    monthly_data = data.copy()
    monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:-3])
    monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
    monthly_data.date = pd.to_datetime(monthly_data.date)
    return monthly_data

#Converts non-stationary data to stationary data and saves by differencing
def get_diff(data, diff):
    data['sales_diff'] = data.sales.diff(diff)
    data = data.dropna()
    
    data.to_csv('/content/temp/stationary_df.csv')
    return data

def plots(data, lags=None):
    
    # Convert dataframe to datetime index
    dt_data = data.set_index('date').drop('sales', axis=1)
    dt_data.dropna(axis=0)
    
    layout = (1, 3)
    raw  = plt.subplot2grid(layout, (0, 0))
    acf  = plt.subplot2grid(layout, (0, 1))
    pacf = plt.subplot2grid(layout, (0, 2))
    
    dt_data.plot(ax=raw, figsize=(12, 5), color='mediumblue')
    smt.graphics.plot_acf(dt_data, lags=lags, ax=acf, color='mediumblue')
    smt.graphics.plot_pacf(dt_data, lags=lags, ax=pacf, color='mediumblue')
    sns.despine()
    plt.tight_layout()


#Draw boxplots
def draw_boxplots(monthly_df):
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
    sns.boxplot(x=monthly_df['date'].dt.year, y='sales', data=monthly_df, ax=axes[0])
    sns.boxplot(x=monthly_df['date'].dt.month, y='sales', data=monthly_df.loc[~monthly_df.date.isin([2013, 2018]), :])

    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(Trend)', fontsize=18)
    axes[1].set_title('Month-wise Box Plot\n(Seasonality)', fontsize=18)
    plt.show()

#For calculating missing values
def calc_percent_NAs(df):
    nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df), columns=['percent']) 
    idx = nans['percent'] > 0
    return nans[idx]


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value))
        
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis(Ho). Therefore, reject the null hypothesis and accept alternate hypothesis. Time series has no unit root and is stationary.")
    else:
        print("Weak evidence against null hypothesis. Therefore, accept null hypothesis and reject alternate hypothesis. Time series has a unit root, indicating it is non-stationary.")

def kpss_test(sales):
    result=kpss(sales, regression='c')
    labels = ['KPSS Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result,labels):
        print(label+' : '+str(value))
        
    if result[1] <= 0.05:
        print("Weak evidence against null hypothesis. Therefore, accept null hypothesis and reject alternate hypothesis. Time series has a unit root, indicating it is non-stationary.")
    else:
        print("Strong evidence against the null hypothesis(Ho). Therefore, reject the null hypothesis and accept alternate hypothesis. Data has no unit root and is stationary.")
       
  
def remove_outliers_iqr(data):
    df = data
    df.set_index('date', inplace=True)

    q25, q75 = np.percentile(df['sales'], 25), np.percentile(df['sales'], 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    outliers = [x for x in df['sales'] if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in df['sales'] if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    
    df = df[~df['sales'].isin(outliers)]
    df = df.reset_index()
    return df

def remove_outliers_quantile(data):
    df = data
    df.set_index('date', inplace=True)
    y = df['sales']
    removed_outliers = y.between(y.quantile(.05), y.quantile(.95))
    index_names = df[~removed_outliers].index # INVERT removed_outliers!!
    df.drop(index_names, inplace=True)
    df = df.reset_index()
    return df

def remove_outliers_std(data):
    df = data
    data_mean, data_std = np.mean(df['sales']), np.std(df['sales'])
    # identify outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    # identify outliers
    outliers = [x for x in df['sales'] if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in df['sales'] if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    
    df = df[~df['sales'].isin(outliers)]
    df = df.reset_index()
    return df
