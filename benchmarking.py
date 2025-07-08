import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

dt_TZ = '2005-06-20' # commencement of 7.75-7.85 target zone (three refinements https://www.hkma.gov.hk/eng/news-and-media/press-releases/2005/05/20050518-4/)
dt_GFC_end = '2009-06-01' # NBER recession https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions

def get_hkma_api(api_url, offset=0, **kwargs):
    result = requests.get(api_url, params={'pagesize': 1000, 'offset': offset, **kwargs}).json()
    records = result['result']['records']
    if result['result']['datasize'] == 1000:
        records += get_hkma_api(api_url, offset=offset+1000)
    return records

# from US Fed
df_EFFR = pd.DataFrame(requests.get('https://markets.newyorkfed.org/api/rates/unsecured/effr/last/0.json').json()['refRates']).eval('effectiveDate = @pd.to_datetime(effectiveDate)', engine = 'python').set_index('effectiveDate')

# daily from HKMA
df_HKDF = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/er-ir/hkd-fer-daily')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_HIBOR = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/er-ir/hk-interbank-ir-daily?segment=hibor.fixing')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_FX = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/er-ir/er-eeri-daily')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_MB = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/monetary-operation/monetary-base-daily')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_MO = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/monetary-operation/market-operation-daily')).eval('end_of_day = @pd.to_datetime(end_of_day)', engine = 'python').set_index('end_of_day').sort_index()
df_IL = pd.DataFrame(get_hkma_api('https://api.hkma.gov.hk/public/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity')).eval('end_of_day = @pd.to_datetime(end_of_date)', engine = 'python').set_index('end_of_day').sort_index().drop(columns=['end_of_date'])

print(df_IL)
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

df_daily_raw = pd.concat([df_HKDF['hkd_fer_spot'], df_HIBOR['ir_overnight'], df_FX['usd'], df_MB, df_MO, df_IL], axis=1)
df_daily_adjusted = df_daily_raw.loc[df_daily_raw.index > pd.Timestamp(dt_TZ)].asfreq('B').dropna()  #or fill
df_monthly_raw = pd.concat([df_BS, df_MS], axis=1)
df_monthly_adjusted = df_monthly_raw.loc[df_monthly_raw.index > pd.Timestamp(dt_TZ)].asfreq('B').ffill().bfill()  #or drop
df_quarterly_raw = pd.concat([df_NGDP, df_BOP_R, df_IIP_R], axis=1)
df_quarterly_adjusted = df_quarterly_raw.loc[df_quarterly_raw.index > pd.Timestamp(dt_TZ)].asfreq('B').ffill().bfill()  #or drop

def model_VAR_1(data):
    sm.tsa.VAR(data).fit(1).summary()

def model_AR_1(data):
    print(sm.tsa.AutoReg(data, lags=1).fit().summary())

model_AR_1(df_daily_adjusted['hkd_fer_spot'])

def find_p_value(data):
    results_p_VAR = []
    for i in range(1,11):
        results_p_VAR.append(VAR(data).fit(i).bic)
    return results_p_VAR.index(min(results_p_VAR))+1

def model_VAR_p(data):
    #Causality test
    #Stationary test
    sm.tsa.VAR(data).fit(find_p_value(data)).summary()


'''
def model_AR_p(data):


def model_ARIMA(data):


def model_RW(data):
#df_daily.to_csv(os.path.join(os.getcwd(), 'daily_data.csv'), index=False)
'''

'''
def solve_OLS(Y, *args):
    T = len(args[0])
    N = len(args)
    X = np.empty((T, N))
    for ix in range(N):
        x = args[ix]
        X[:, ix] = x
    XX = X.T @ X
    XY = X.T @ Y
    coefs = np.linalg.inv(XX) @ XY
    return coefs
'''






