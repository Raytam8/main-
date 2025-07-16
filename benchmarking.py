import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import matplotlib.pyplot as plt

dt_TZ = '2005-06-20' # commencement of 7.75-7.85 target zone (three refinements https://www.hkma.gov.hk/eng/news-and-media/press-releases/2005/05/20050518-4/)
dt_GFC_end = '2009-06-01' # NBER recession https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
dt_training = '2019-12-31' # data before that date is used for training
dt_end = '2025-06-30' # end date of data

def get_hkma_api(api_url, offset=0, **kwargs):
    result = requests.get(api_url, params={'pagesize': 1000, 'offset': offset, **kwargs}).json()
    records = result['result']['records']
    if result['result']['datasize'] == 1000:
        records += get_hkma_api(api_url, offset=offset+1000)
    return records

# from US Fed
df_EFFR = pd.DataFrame(requests.get('https://markets.newyorkfed.org/api/rates/unsecured/effr/last/0.json').json()['refRates']).eval('end_of_day = @pd.to_datetime(effectiveDate)', engine = 'python').set_index('end_of_day').drop(columns=['effectiveDate']).sort_index()

# daily from HKMA
df_HKDF = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/er-ir/hkd-fer-daily')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_HIBOR = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/er-ir/hk-interbank-ir-daily?segment=hibor.fixing')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_FX = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/er-ir/er-eeri-daily')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_MB = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/monetary-operation/monetary-base-daily')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_MO = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/monetary-operation/market-operation-daily')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_IL = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity')).eval('end_of_day = @pd.to_datetime(end_of_date)', engine = 'python').set_index('end_of_day').sort_index().drop(columns=['end_of_date'])

# monthly from HKMA
df_BS = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/banking/balance-sheet-ais')).eval('end_of_month = @pd.to_datetime(end_of_month).add(@pd.offsets.MonthEnd())', engine = 'python').set_index('end_of_month').sort_index()
df_MS = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/money/supply-components-hkd')).eval('end_of_month = @pd.to_datetime(end_of_month).add(@pd.offsets.MonthEnd())', engine = 'python').set_index('end_of_month').sort_index()

# from C&SD
df_NGDP = (
    pd.DataFrame(requests.post("https://www.censtatd.gov.hk/api/post.php", data={'query': json.dumps({
        "cv": { "GDP_COMPONENT": []},
        "sv": {"SA1": ["Raw_M_hkd_d"],"SA_DEF": ["Raw_1dp_idx_n"]},
        "period": {"start": "200001"},
        "id": "310-31007",
        "lang": "en"
        })}).json()['dataSet']
    ).query('freq == "Q"').eval('period = @pd.to_datetime(period, format="%Y%m").add(@pd.offsets.MonthEnd())', engine = 'python')
    .set_index(['period', 'sv'])['figure'].unstack().eval('SA1 * SA_DEF / 100')
    .rename('NGDP_SA')[dt_TZ:]
) #quarterly
df_BOP_R = (
    pd.DataFrame(requests.post("https://www.censtatd.gov.hk/api/post.php", data={'query': json.dumps({
        "cv": { "BOP_COMPONENT": [ "RA" ]},
        "sv": {"BOP": ["Raw_M_hkd_d"]},
        "period": {"start": "199901"},
        "id": "315-37001",
        "lang": "en"
        })}).json()['dataSet']
    ).query('freq == "Q"').eval('''
    period = @pd.to_datetime(period, format="%Y%m").add(@pd.offsets.MonthEnd())
    BOP_COMPONENT = BOP_COMPONENT.radd("BOP_")
    ''', engine = 'python')
    .set_index(['period', 'BOP_COMPONENT'])['figure'].unstack()
) #quarterly
df_IIP_R = (
    pd.DataFrame(requests.post("https://www.censtatd.gov.hk/api/post.php", data={'query': json.dumps({
        "cv": { "IIP_COMPONENT": [ "IIP_ATS_RA" ]},
        "sv": {"IIP": ["Raw_M_hkd_d"]},
        "period": {"start": "201001"},
        "id": "315-37021",
        "lang": "en"
        })}).json()['dataSet']
    ).query('freq == "Q"').eval('''
    period = @pd.to_datetime(period, format="%Y%m").add(@pd.offsets.MonthEnd())
    ''', engine = 'python')
    .set_index(['period', 'IIP_COMPONENT'])['figure'].unstack()
) #quarterly 

df_daily_raw = pd.concat([df_HKDF['hkd_fer_spot'], df_HIBOR['ir_overnight'].fillna(df_IL['hibor_overnight']), df_FX['usd'], df_MB['mb_bf_disc_win_total'], df_MO['closing_balance'], df_IL['disc_win_base_rate'], df_EFFR['percentRate']], axis=1)
df_monthly_raw = pd.concat([df_BS, df_MS], axis=1)
df_quarterly_raw = pd.concat([df_NGDP, df_BOP_R, df_IIP_R], axis=1)

#if dropna then asfreq --> missing data error: exog contains inf or nans
#if asfreq than dropna, or only dropna or only asfreq --> value warning: no associated frequency

''' check missing values
for date, row in df_daily_adjusted.iterrows():
    missing_info = []
    missing_columns = row.index[row.isnull()].tolist()  
    if missing_columns:  # If there are missing columns
        missing_info.append((date, missing_columns))
missing_df = pd.DataFrame(missing_info, columns=['Date', 'Missing_Columns'])
'''

df_daily_training = df_daily_raw[(df_daily_raw.index > dt_TZ) & (df_daily_raw.index < dt_training)].asfreq('B').bfill() #bfill() or fillna(avg of +1 and -1) #change this to working day hong kong
df_daily_forecasting = df_daily_raw[(df_daily_raw.index > dt_TZ) & (df_daily_raw.index <= dt_end)].asfreq('B').bfill()

#model AR1 and AR(p)

# -test stationary:

#forecast:
def find_p_AR(data, maxlag): #input column, maxlag
    results_p_AR = []
    for i in range(1, maxlag):
        results_p_AR.append(sm.tsa.AutoReg(data, lags=i).fit().bic)
    return results_p_AR.index(min(results_p_AR))+1

def model_AR_forecast_rolling(data): #input column, parameters can be removed
    rolling_forecast_AR = []
    parameters = []
    lag = 3 #find_p_AR(data)
    for i in range(0, 300): #not sure why needs to be 2*lag + 1, but if smaller than this it returns error not enough datapoints / cannot be divided by 0, #check kcross validation/constant width of time window?
        if i > 2*lag+1:
            rolling_forecast_AR.append((sm.tsa.AutoReg(data.iloc[:i], lags=lag).fit().predict(start=i, end=i)).iloc[0])
            parameters.append(sm.tsa.AutoReg(data.iloc[:i], lags=lag).fit().params)
        else:
            rolling_forecast_AR.append(data.iloc[i])
            parameters.append(0)
    forecast_df = pd.DataFrame(rolling_forecast_AR, columns=['Forecasted_Values'], index=data.index[:300])
    return forecast_df # parameters

def model_AR_forecast_static(data): #input column, change training set to forecast set if want to use data up till 2019 only
    lag = find_p_AR(data)
    forecast_value = sm.tsa.AutoReg(data, lags=lag).fit().model.predict(sm.tsa.AutoReg(data, lags=lag).fit().params, start=dt_training, end=dt_end)
    #forecast_value.index = df_daily_forecasting.index[len(df_daily_training):]
    return forecast_value

print(model_AR_forecast_static(df_daily_training['usd']))

#model VAR and VAR(p)

def model_VAR_forecast_static(data): 
    #Causality test
    #Stationary test
    results = pd.DataFrame(sm.tsa.VAR(data)
                            .fit(maxlags=20, ic='bic')
                            .forecast(y=data.values[-sm.tsa.VAR(data)
                            .fit(maxlags=20, ic='bic').k_ar:],
                            steps=len(df_daily_forecasting)-len(df_daily_training)), 
                            columns=list(map(lambda x: x + 'forecast', data.columns)), 
                            index=df_daily_forecasting.index[len(df_daily_training):])
    return results

def model_VAR_forecast_rolling(data):
    results = pd.DataFrame(columns=list(map(lambda x: x + 'forecast', data.columns)))
    lag = 1 #find_p_AR(data)
    for i in range(0, 60): #not sure why needs to be 2*lag + 1, but if smaller than this it returns error not enough datapoints / cannot be divided by 0, #check kcross validation/constant width of time window?
        if i > 2*lag+1:
            pd.concat([results, sm.tsa.VAR(data.iloc[:i]).fit(1).forecast(y=data[1:],steps=1)], axis=0) #'''.values[-sm.tsa.VAR(data.iloc[:i]).fit(maxlags=1, ic='bic').k_ar:], steps=1)]'''
            # parameters.append(sm.tsa.AutoReg(data.iloc[:i-1], lags=lag).fit().params)
        else:
            pd.concat([results, data.iloc[i]], axis=0)
            # parameters.append(0)
    return results

print(model_VAR_forecast_rolling(df_daily_forecasting))

#plot graph

def plot_graph(data_forecast, data_actual):
    plt.figure(figsize=(12, 6))
    plt.plot(data_forecast.index, data_forecast, label='Forecast Values', color='blue', marker='o')
    plt.plot(data_actual.index, data_actual, label='Actual Values', color='red', marker='x')
    plt.title('Rolling Forecast with AR(1) Model')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_graph(model_AR_forecast_rolling(df_daily_forecasting['usd']), df_daily_forecasting['usd'])

# Explaining performance: In-sample explaining power
# Forecasting performance: Out-of-sample forecast: Return four measures: 1) RMSE ratio of model to benchmark, 2) Ratio of direction of change, 3) t-statistics that test the null of forecast errors with 0 mean, 4) number of times ... 
# Train with 2005 to 2020, forecast 2020 to 2025 
