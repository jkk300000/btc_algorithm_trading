import pandas as pd
import numpy as np
import cupy as cp
from joblib import Parallel, delayed
import logging
from tqdm import tqdm

# logger 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def monte_carlo_var(prices, investment, confidence_level=0.99, days=1, num_simulations=100000, lookback=1000):
    
    arr = cp.array(prices)
    log_returns = cp.log(arr[1:] / arr[:-1])
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = arr[-1]
    # 시뮬레이션
    z = cp.random.normal(size=(num_simulations, days))
    random_returns = mu + sigma * z
    price_paths = last_price * cp.exp(cp.cumsum(random_returns, axis=1))
    simulated_end_prices = price_paths[:, -1]
    ending_returns = simulated_end_prices / last_price - 1
    var_percent = cp.percentile(ending_returns, (1 - confidence_level) * 100)
    var_dollar = investment * -var_percent
    # cupy에서 numpy로 변환
    
    var_percent = cp.asnumpy(var_percent)
    var_dollar = cp.asnumpy(var_dollar)
    return var_dollar, var_percent

def _calc_var_row(i, df, lookback, investment_factor, confidence_level, days, num_simulations):
    window = df.iloc[i-lookback:i]
    prices = window['close'].values
    investment = df.iloc[i]['close'] * investment_factor
    var_dollar, var_percent = monte_carlo_var(prices, investment, confidence_level, days, num_simulations, lookback)
    return i, var_percent, var_dollar

def calc_var(df, output_path=None, start_date='2022-09-01', investment_factor=10, confidence_level=0.99, days=1, num_simulations=100000, lookback=1000, n_jobs=7):
    df = df.copy()
    df['var'] = np.nan
    df['var_dollar'] = np.nan
    start_ts = pd.to_datetime(start_date)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(df.index.tz)
        else:
            start_ts = start_ts.tz_convert(df.index.tz)
    if start_ts in df.index:
        start_idx = df.index.get_loc(start_ts)
    else:
        # start_ts보다 크거나 같은 첫 번째 인덱스 위치 반환
        start_idx = df.index.searchsorted(start_ts, side='left')
    indices = [i for i in range(start_idx, len(df)) if i >= lookback]
    logger.info(f"calc_var 시작: 총 {len(indices)}개 대상, 병렬 작업(n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_calc_var_row)(i, df, lookback, investment_factor, confidence_level, days, num_simulations)
        for i in tqdm(indices, desc='calc_var 진행률')
    )
    logger.info("calc_var 병렬 처리 완료. 결과 반영 중...")
    for idx, (i, var_percent, var_dollar) in enumerate(results, 1):
        df.iloc[i, df.columns.get_loc('var')] = var_percent
        df.iloc[i, df.columns.get_loc('var_dollar')] = var_dollar
    if output_path:
        df.to_csv(output_path)
        logger.info(f'Saved to {output_path}')
    logger.info("calc_var 완료.")

    print("[train_and_predict] 데이터 진단 결과:")
    print(f"전체 행 개수: {len(df)}")
    print("피처별 결측치 개수:")
    print(df.isna().sum())
    print("피처별 결측치 비율(%):")
    print((df.isna().sum() / len(df) * 100).round(2))
    print("결측치가 없는 행 개수:", len(df.dropna()))


    return df

def _mean_var_row(i, df, lookback, investment_factor, confidence_level, days, num_simulations):
    window = df.iloc[i-lookback:i]
    prices = window['close'].values
    investment = df.iloc[i]['close'] * investment_factor
    _, var_percent = monte_carlo_var(prices, investment, confidence_level, days, num_simulations, lookback)
    return var_percent

def calc_mean_var_from_df(df, start_date=None, lookback=1000, investment_factor=10, confidence_level=0.99, days=1, num_simulations=100000, n_jobs=7):
    if start_date:
        start_ts = pd.to_datetime(start_date)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(df.index.tz)
            else:
                start_ts = start_ts.tz_convert(df.index.tz)
        df = df[df.index >= start_ts]
    if 'var' in df.columns and df['var'].notna().any():
        return df['var'].dropna().mean()
    indices = [i for i in range(lookback, len(df))]
    var_list = Parallel(n_jobs=n_jobs)(
        delayed(_mean_var_row)(i, df, lookback, investment_factor, confidence_level, days, num_simulations)
        for i in indices
    )
    return np.mean(var_list) if var_list else np.nan

if __name__ == '__main__':
    df = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_rf.csv', index_col=0, parse_dates=True)
    calc_var(df, 'C:/선물데이터/binance_btcusdt_1m_rf_var.csv', n_jobs=7)
    mean_var = calc_mean_var_from_df(df, start_date='2022-09-01', n_jobs=7)
    print('Mean VaR (2022-09-01~):', mean_var) 


    df = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_rf_var.csv', index_col=0, parse_dates=True)

    # val 컬럼에 양수 값이 하나라도 있는지 확인
    has_positive_val = (df['val'] > 0).any()

    if has_positive_val:
        print("val 컬럼에 양수 값이 있습니다.")
    else:
        print("val 컬럼에 양수 값이 없습니다.")