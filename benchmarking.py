import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import matplotlib.pyplot as plt
import holidays

hi

###########################data 

dt_TZ = '2005-06-20' # commencement of 7.75-7.85 target zone (three refinements https://www.hkma.gov.hk/eng/news-and-media/press-releases/2005/05/20050518-4/)
dt_GFC_end = '2009-06-01' # NBER recession https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
dt_training = '2020-01-01' # data before that date is used for training
dt_end = '2021-06-30' # end date of data

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

df_MS.to_csv('./df_MS.csv')

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

df_raw = pd.concat(
    [df_HIBOR['ir_overnight'].fillna(df_IL['hibor_overnight']), df_FX['usd']], axis=1
    ).assign(
        MZMe_t = df_MS['demand_deposits_with_lb'] + df_MS['savings_deposits_with_lb']
    ).pipe(
        lambda df: df[df.index.to_series().apply(lambda x: x.weekday() < 5 and x not in holidays.HongKong(years=range(2004,2026)))]
    ).dropna()


df_training_set = df_raw[(df_raw.index > dt_TZ) & (df_raw.index < dt_training)]
df_full_actual_set = df_raw[(df_raw.index > dt_TZ) & (df_raw.index <= dt_end)]


#### main.py
# from data_lers_beachmark import *
# [var.to_pickle(f'./data/dataframe/{name}.pkl') for name, var in globals().items() if name[:3] == 'df_']


for name in path.search('./data/dataframe/'):
    globals()[name <- remove .pkl suffix] = pd.read_pickle(... name)


class LERSBenchmark:
    file_path = ''
    model_list = []
    var_test = []
    in_result = []
    def __init__(self, model_list, var_to_test, freq_to_test, name='./output'):
        self.model_list = model_list


    def run_in_sample():
        for model in self.model_list():
            # for var in var_test
            #    for freq in freq_test:
            #      if var[0] == 'M' and freq == 'D': 
            #         continue
            #      model.fit()
            #      in_result[model.name] = model.implied_hist_value()
            # check whether model can handle this test,
            # if not,
            # continue
    
    def run_outsampele():
        ss

    def save_result(self, name):

    def generate_figure(self, name):
        .to_figure().jpg


# import argparse

if == 'main':
    bench = LERSBenchmark(...)
    bench.run_in_sample('')
    bench.save_result('')


class base_model:
    name = ''
    # some properties that tells what can this model handle and what can't
    def __init__(self, ...., data, name, ....):
        # what can this model handle and what can't
        return
    
    def fit():
        return
    #....

class ar1_model(base_model):
    def __init__():
        super().__init__(name='AR(1)')

####################main#########################

def forecasting_power(dataset):
    dfs = pd.DataFrame(
        {'ar_st': np.sqrt(np.mean((dataset['ar_st']-dataset['actual']).pow(0.5))),
        'ar_rl': np.sqrt(np.mean((dataset['ar_rl']-dataset['actual'])**2)),
        'var_st': np.sqrt(np.mean((dataset['var_st']-dataset['actual'])**2)),
        'var_rl': np.sqrt(np.mean((dataset['var_rl']-dataset['actual'])**2))}
    )
    return dfs
    
def combine_fc_powers(datasets):
    combined_list = pd.concat([forecasting_power(df) for df in datasets], axis=0)
    combined_list.to_csv('./forecasting_powers.csv')
    return combined_list 


####################################model AR1 and AR(p)

# test stationary:

# check autocorrelation

#forecast:

def find_p_AR(data): 
    results_p_AR = []
    for i in range(1, 10): #30 is maxlag assumed
        results_p_AR.append(sm.tsa.AutoReg(data, lags=i).fit().bic)
    return results_p_AR.index(min(results_p_AR))+1

def model_AR_forecast_rolling(t_data, e_data): #should input full time length #trained with 2005-2020, **expanding window** forecast 2021-2025, Y = ßX + residual, 
    lag = find_p_AR(t_data)
    dfs = []
    for i in range(len(t_data), len(e_data)):
        dfs.append(
            pd.DataFrame({
            'ar_rl': [sm.tsa.AutoReg(e_data[:i], lags=lag).fit().predict(start=i, end=i).iloc[0]],
        }, index=[e_data.index[i]])
        )
    forecast = pd.concat(dfs)
    return forecast

def model_AR_forecast_static(t_data, e_data): #trained with 2005-2020, forecast 2021-2025, Y = ßX + residual, ß=(X'X)X'Y (OLS)
    lag = find_p_AR(t_data)
    forecast = pd.DataFrame({'ar_st': sm.tsa.AutoReg(t_data, lags=lag).fit().predict(start=len(t_data), end=len(e_data)-1)}).set_index(e_data.index[len(t_data):])
    return forecast

####################################model VAR and VAR(p)

def model_VAR_forecast_rolling(t_data, e_data):
    dfs = []
    for i in range(len(t_data.index), len(e_data.index)): 
        fitted_model = sm.tsa.VAR(e_data[:i]).fit(maxlags=10, ic='bic')
        df = pd.DataFrame(
            {'var_rl': fitted_model.forecast(y=e_data.values[i-fitted_model.k_ar:i], steps=1)[0][0]}, 
            index = [e_data.index[i]]  
        ) 
        dfs.append(df)
    forecast = pd.concat(dfs)
    return forecast

###################################create dataset:

usd_dataset = pd.DataFrame({
    'actual': df_full_actual_set['usd'], 
    'ar_static': model_AR_forecast_static(df_training_set['usd'], df_full_actual_set['usd']),
    'ar_rolling': model_AR_forecast_rolling(df_training_set['usd'], df_full_actual_set['usd']),
    'var_static': model_VAR_forecast_static(df_training_set, df_full_actual_set),
    'var_rolling': model_VAR_forecast_rolling(df_training_set, df_full_actual_set)
    }).pipe(lambda df: df[df.index>=dt_training])

usd_dataset.to_csv("./usd_dataset.csv")

###################################ARIMA Model

ax = df.eval('forecast = forecast.shift(...)')[['actual', 'forecast']].plot()
plt.to_fig(...)
plt.show()

####################################Random Walk Model


#plot graph

def plot_graph(data_forecast, data_actual, data_forecast_2):
    plt.figure(figsize=(12, 6))
    plt.plot(data_forecast.index, data_forecast, label='Forecast Values', color='blue', marker='o')
    plt.plot(data_actual.index, data_actual, label='Actual Values', color='red', marker='x')
    plt.plot(data_forecast_2.index, data_forecast_2, label='VAR(P)', color='green', marker='v')
    plt.title('Rolling Forecast with AR(1) Model')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
plot_graph(model_AR_forecast_rolling(df_training_set['usd']), df_full_actual_set['usd'], model_VAR_forecast_rolling(df_training_set)['usd_forecast'])

# Explaining performance: In-sample explaining power
# Forecasting performance: Out-of-sample forecast: Return four measures: 1) RMSE ratio of model to benchmark, 2) Ratio of direction of change, 3) t-statistics that test the null of forecast errors with 0 mean, 4) number of times ... 
# Train with 2005 to 2020, forecast 2020 to 2025 




df_daily = pd.concat([
    df_FX, df_MB, df_IL, df_EFBN_yield, df_HIBOR, df_HKDF, 
    (process_df_EFB(df_EFB_91d, 91) + process_df_EFB(df_EFB_182d, 182) + process_df_EFB(df_EFB_28d, 28) + process_df_EFB(df_EFB_364d, 364))['yield_amount'].div(df_MB.eval('outstanding_efbn')).mul(365).rename('r_EFBN_t')
], axis=1).astype('float').join(df_USIR).join(
    df_MS.join(df_BS['liab_depcust_hkd']).eval('''
    ipo_money_adj = 0
    MZMd_t = savings_deposits_with_lb + demand_deposits_with_lb
    MZM_t = MZMd_t + legal_tender_notes_coins_in_pub
    MZMc_t = MZMd_t - ipo_money_adj
    M_t = MZMc_t
    HKDD_t = liab_depcust_hkd
    ''')
).eval('''
E_t = hkd_fer_spot
AB_t = aggr_balance_bf_disc_win
AB_t1 = forecast_aggregate_bal_t1
CI_t = cert_of_indebt
EFBN_t = outstanding_efbn.mask(outstanding_efbn.diff().sub(-interest_payment).abs().gt(4500)).interpolate(limit_area='inside')
r_EFBN_t = r_EFBN_t.interpolate(limit_area='inside')
EMB_t = AB_t + EFBN_t
ipo_short_rate_adj = 0
if_t = hibor_overnight - ipo_short_rate_adj
iU_t = DFEDTAR.combine_first(DFEDTARL + 0.1)
iDW_t = iU_t - 0.1 + 0.5
BaseRate_t = disc_win_base_rate
HIBORON_t = hibor_overnight
ΔFXO_t = market_activities
ΔA_t = -interest_payment
FXO_t = ΔFXO_t.cumsum()
A_t = ΔA_t.cumsum()
HKDD_t = HKDD_t.asfreq('D').interpolate()
MZMc_t = MZMc_t.asfreq('D').interpolate()
MZMd_t = MZMd_t.asfreq('D').interpolate()
MZM_t = MZM_t.asfreq('D').interpolate()
M_t = M_t.asfreq('D').interpolate()
ΔFXO_to_M_t = ΔFXO_t / M_t
ΔA_to_M_t = ΔA_t / M_t
AB_to_M_t = AB_t / M_t
EFBN_to_M_t = EFBN_t / M_t
EMB_to_M_t = EMB_t / M_t
MB_t = mb_bf_disc_win_total
epoch_t = index.astype("int").div(1e16)
epoch_exp_t = exp(epoch_t)
''')



