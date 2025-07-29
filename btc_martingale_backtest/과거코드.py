import backtrader as bt
import pandas as pd
import logging
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from backtrader.utils.date import date2num
import ccxt
from pathlib import Path
import os
import talib as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, StackingClassifier,GradientBoostingClassifier
from sklearn.linear_model import Ridge, RidgeClassifierCV
from xgboost import XGBClassifier
from backtrader.utils.py3 import string_types, integer_types
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from joblib import Parallel, delayed
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 바이낸스 API 설정
api_key = ""  # 바이낸스 API 키
api_secret = ""  # 바이낸스 API 시크릿 키
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# 거래 설정
trading_symbol = 'BTCUSDT'
timeframe = '1m'  # 1시간봉 (리샘플링하여 일봉으로 변환)
leverage = 10
limit = 1000
risk_per_trade = 0.02  # 계좌 잔고의 2% 리스크
stop_loss_pct = 0.02  # 2% 손절
take_profit_pct = 0.04  # 4% 익절
lookback_period = 10000  # 학습 데이터 기간
initial_capital = 1000  # 백테스팅 초기 자본 (USDT)
start = pd.Timestamp('2021-01-01')
end = pd.Timestamp('2025-07-17')
monte_start = pd.Timestamp('2019-09-08')
monte_end = pd.Timestamp('2020-12-31')

        
    
# 레버리지 및 격리 마진 설정 (CCXT 방식)
def set_leverage_and_margin(trading_symbol, leverage):
    try:
        binance.set_leverage(leverage, trading_symbol.replace('/', ''))  # CCXT set_leverage 메서드
        binance.set_margin_mode('isolated', trading_symbol.replace('/', ''))  # 격리 마진 설정
        logger.info(f"레버리지 {leverage}배, 격리 마진 설정 완료")
    except Exception as e:
        logger.error(f"레버리지/마진 설정 오류: {e}")

#차트 데이터 가져오기
def fetch_ohlcv(trading_symbol, timeframe, limit):
    ohlcv = binance.fetch_ohlcv(trading_symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def fetch_ohlcv_between(trading_symbol, timeframe, start_date, end_date, limit=1000, max_bars=1150000):
    """
    지정한 시작일~종료일까지의 OHLCV 데이터를 모두 가져옴.
    - start_date, end_date: 'YYYY-MM-DD' 또는 pandas.Timestamp 등으로 입력 가능
    """
    # 날짜를 ms 단위로 변환
    if isinstance(start_date, str):
        start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    else:
        start_ms = int(pd.to_datetime(start_date).timestamp() * 1000)
    if isinstance(end_date, str):
        end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)
    else:
        end_ms = int(pd.to_datetime(end_date).timestamp() * 1000)

    all_ohlcv = []
    since = start_ms
    while True:
        ohlcv = binance.fetch_ohlcv(trading_symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        # 기간 내 데이터만 필터링
        ohlcv = [row for row in ohlcv if row[0] <= end_ms]
        all_ohlcv += ohlcv
        if len(ohlcv) < limit or (all_ohlcv and all_ohlcv[-1][0] >= end_ms) or len(all_ohlcv) >= max_bars:
            break
        since = all_ohlcv[-1][0] + 1  # 마지막 캔들 이후부터

    # DataFrame 변환
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# 기관 (헤지펀트에서 사용한다고 하는 파라미터 값을 적용한)몬테카를로 시뮬레이션



def monte_carlo_var_parallel(close_series, investment, confidence_level=0.95, days=30, num_simulations=10000, n_jobs=-1):
    """
    병렬 연산 기반 Monte Carlo VaR 계산
    - close_series: 과거 종가 (pd.Series)
    - investment: 투자 금액 ($)
    - confidence_level: VaR 신뢰도 (예: 0.99)
    - days: 예측 기간 (예: 10)
    - num_simulations: 시뮬레이션 반복 횟수 (예: 100_000)
    - n_jobs: 동시에 사용할 CPU 코어 수, -1이면 전체 사용
    """

    log_returns = np.log(close_series / close_series.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = close_series.iloc[-1]

    # 단일 시뮬레이션 함수
    def simulate_mc():
        returns = np.random.normal(mu, sigma, days)
        return last_price * np.exp(np.cumsum(returns))[-1]  # 마지막 가격만 사용

    # 병렬 시뮬레이션 실행
    simulated_end_prices = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(simulate_mc)() for _ in range(num_simulations)
    )

    ending_returns = np.array(simulated_end_prices) / last_price - 1
    var_percent = np.percentile(ending_returns, (1 - confidence_level) * 100)
    var_dollar = investment * -var_percent  # 손실값이므로 -부호 처리

    return var_dollar, var_percent, simulated_end_prices
 



def calculate_indicators(df):
    # Copy DataFrame to avoid modifying original
    df = df.copy()
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open = df['open'].values
    
    # RSI
    df['rsi'] = ta.RSI(close, timeperiod=14)
    
    # SMA and EMA
    df['sma'] = ta.SMA(close, timeperiod=20)
    df['ema'] = ta.EMA(close, timeperiod=20)
    
    # Bollinger Bands
    upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_upper'] = upper
    df['bb_lower'] = lower
    df['bb_mid'] = middle

    # Squeeze Momentum Indicator
    length = 20
    mult = 2.0
    lengthKC = 20
    multKC = 1.0
    useTrueRange = True

    # Bollinger Bands
    basis = ta.SMA(close, timeperiod=length)
    dev = mult * ta.STDDEV(close, timeperiod=length)
    df['upperBB'] = basis + dev
    df['lowerBB'] = basis - dev

    # Keltner Channels
    ma = ta.SMA(close, timeperiod=lengthKC)
    range1 = ta.TRANGE(open, low, close) if useTrueRange else (high - low)
    rangema = ta.SMA(range1, timeperiod=lengthKC)
    df['upperKC'] = ma + rangema * multKC
    df['lowerKC'] = ma - rangema * multKC

    # Squeeze conditions
    df['sqzOn'] = (df['lowerBB'] > df['lowerKC']) & (df['upperBB'] < df['upperKC'])
    df['sqzOff'] = (df['lowerBB'] < df['lowerKC']) & (df['upperBB'] > df['upperKC'])
    df['noSqz'] = ~df['sqzOn'] & ~df['sqzOff']

    # Squeeze Momentum (val)
    highest_high = ta.MAX(high, timeperiod=lengthKC)
    lowest_low = ta.MIN(low, timeperiod=lengthKC)
    avg_hl = (highest_high + lowest_low) / 2
    avg_sma = ta.SMA(close, timeperiod=lengthKC)
    df['val'] = ta.LINEARREG(close - (avg_hl + avg_sma) / 2, timeperiod=lengthKC)

    # bcolor: Momentum direction
    df['bcolor'] = np.where(
        df['val'] > 0,
        np.where(df['val'] > df['val'].shift(1), 1, 2),  # 1: lime (increasing), 2: green (decreasing)
        np.where(df['val'] < df['val'].shift(1), 3, 4)   # 3: red (decreasing), 4: maroon (increasing)
    )

    # scolor: Squeeze state
    df['scolor'] = np.where(df['noSqz'], 1, np.where(df['sqzOn'], 2, 3))  # 1: blue, 2: black, 3: gray

    # Highest and lowest prices
    df['highest_price'] = high
    df['lowest_price'] = low
    
    
    
    
    # atr
    
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # Log NaN counts
    
    # logger.info(f"Indicator NaN counts:\n{df.isna().sum()}")
    
    
    
    return df


def train_ml_models(df):
    n = 240
    rolling_max = df['close'].shift(-1).rolling(window=n, min_periods=1).max()
    df['target'] = (rolling_max >= df['close'] * 1.004).astype(int)
    features = ['close', 'rsi', 'sma', 'ema', 'bb_upper', 'bb_lower', 'bb_mid', 
                'val', 'bcolor', 'scolor', 'volume', 'atr']
    
    # 분포 확인
    # logger.info(f"df['target'].value_counts() : {df['target'].value_counts()}")
    
    print(f"전체 데이터 길이: {len(df)}")
    
    
    df_clean = df.dropna()
    
    # print(f"NaN 제거 후 데이터 길이: {len(df_clean)}")
    # logger.info(f"Indicator NaN counts:\n{df_clean.isna().sum()}")
    
    
    # if df_clean.empty:
    #     raise ValueError("No valid data after removing NaN values.")
    X = df_clean[features]
    y = df_clean['target']
    
    # logger.info(f"y 값 : {y}")
    # if y.nunique() < 2:
    #     raise ValueError("Target must have at least two classes")

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # 클래스 1일 확률 예측
        y_prob = rf.predict_proba(X_test)[:, 1]  # 클래스 1의 확률
        
        score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        scores.append(score)
        print(f"Fold score: {score:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"클래스 1일 평균 확률: {np.mean(y_prob):.4f}")
        
    print(f"평균 교차검증 점수: {sum(scores)/len(scores):.4f}")
    # 전체 데이터로 최종 학습
    rf.fit(X, y)
    
  
    return features, rf



class MLFuturesStrategy(bt.Strategy):
    params = dict(
        lookback_period=lookback_period,
        inputTrade=10,          # 최대 진입 횟수
        additionalEntryPrice=1200,  # 물타기 한도 (가격 단위)
        profit=1.01,            # 익절 비율
        leverage=10,
        dividedLongCount=20,
        montecarlo_var_dollar = 0.0,
        montecarlo_var_percent = 0.0
    )

    def __init__(self):
        self.entryCount = 0
        self.entryPriceSum = 0.0
        self.avgPrice = 0.0
        self.is_trained = False
        self.order = None
        self.trades = []
        self.stopless = 0.0
        self.portfolio_values = []
        self._trade_id = 0
        
        self.entry_price_sum = 0.0
        self.total_entry_count = 0
        self.initial_entry_price = 0.0
        self.second_entry_price = 0.0
        self.is_trading_active = True
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        self.features = ['close', 'rsi', 'sma', 'ema', 'bb_upper', 'bb_lower', 'bb_mid', 
                'val', 'bcolor', 'scolor', 'volume']
        self.rf = None
        self.var_history = []  # ✅ 각 거래 때의 VaR 누적 저장소
        self.montecarlo_var_dollar = None
        self.montecarlo_var_percent = None
        
    def next(self):
        # 데이터프레임 준비
        df = pd.DataFrame({
            'open': self.datas[0].open.get(size=self.params.lookback_period),
            'high': self.datas[0].high.get(size=self.params.lookback_period),
            'low': self.datas[0].low.get(size=self.params.lookback_period),
            'close': self.datas[0].close.get(size=self.params.lookback_period),
            'volume': self.datas[0].volume.get(size=self.params.lookback_period)
        }, index=self.datas[0].datetime.get(size=self.params.lookback_period))
        df.index = pd.to_datetime([bt.num2date(x) for x in df.index])

        df = calculate_indicators(df)  # 지표 계산 함수

       # 모델 학습 (최초 1회만 수행)
        if not self.is_trained:
            try:
                features, rf = train_ml_models(df)
                self.is_trained = True
                self.features = features
                self.rf = rf
                self.log("Models trained successfully")
            except Exception as e:
                self.log(f"Model training failed: {e}")
                return

        val = df['val'].iloc[-1]
        atr = df['atr'].iloc[-1]
        latest_data = df[self.features].iloc[-1:]
        proba = self.rf.predict_proba(latest_data)[0][1]
        
        close_price = self.datas[0].close[0]
        open_price = self.datas[0].open[0]

        
        
        # if len(df) >= 100000:
        #     try:
                
        # 전략 변수에 저장 (최신)
        var_dollarForBacktest, var_pctForBacktest, paths = monte_carlo_var_parallel(
            close_series=df['close'][-1000:],   # 데이터 개수 1000~3000개 등
            investment=(self.broker.getvalue() * self.params.leverage),
            confidence_level=0.99,              # ← 권장 신뢰구간(99% 등)
            days=1,
            num_simulations=100000,             # ← 병렬이라 시간 부담 ↓ 가능
            n_jobs=-1                           # CPU 전체 사용 (8코어면 8개)
        )
        self.montecarlo_var_dollar = var_dollarForBacktest
        self.montecarlo_var_percent = var_pctForBacktest
                
                
                # log 확인
                # self.log(f"Monte Carlo VaR %: {var_dollar:.2f} ({var_pct:.2%})")

                # 조건 차단 예시: VaR이 현재 자본의 4% 이상이면 진입 금지
                # if var_dollar > self.broker.getvalue() * 0.04:
                    # self.log(f"VaR too high (${var_dollar:.2f}). Skipping entry.")
                    # return

            # except Exception as e:
            #     # self.log(f"Monte Carlo VaR calculation failed: {e}")
            #     return
        var_dollar = self.params.montecarlo_var_dollar
        var_pct = self.params.montecarlo_var_percent
        
        # 조건 차단 예시: VaR이 현재 자본의 4% 이상이면 진입 금지
        # if var_dollar > self.broker.getvalue() * 0.1:
        #     return
        # 진입 조건
        

        # position_size = max((self.params.dividedLongCount and (self.broker.getvalue() * self.params.leverage) / self.params.dividedLongCount) / close_price, 0.0001)
        position_size = ((self.broker.getvalue() * self.params.leverage) / self.params.dividedLongCount) / close_price
        # logger.info(f'initial_capital : {self.broker.getvalue()}')
        can_enter = self.entryCount < self.params.inputTrade and position_size > 0
        if can_enter:
            # 첫 진입 또는 추가 진입
            # logger.info(f'var_dollars : {var_dollar}')
            if self.entryCount == 0 and val > 0 and close_price > open_price and self.total_entry_count == 0 and var_dollar <= self.broker.getvalue() * 0.1 and var_dollarForBacktest <= self.broker.getvalue() * 0.1:
                
                self.var_history.append(var_dollar)
                # logger.info(f'self.var_history : {self.var_history}')
                self.montecarlo_var_dollar = var_dollar
                self.montecarlo_var_percent = var_pct
                self.entryPriceSum = close_price
                self.total_entry_count = 1
                self.avgPrice = self.entryPriceSum/self.total_entry_count
                self.buy(size=position_size)
                self.entryCount += 1
                self.log(f"Entry 1 at {close_price:.2f}, avgPrice: {self.avgPrice:.2f}, self.entryCount : {self.entryCount}, position : {position_size}, var_dollar : {var_dollarForBacktest}")
            
            # 물타기 조건: 현재가가 평균가에서 충분히 내려왔는지
            if self.entryCount >= 1  and self.entryCount < self.params.inputTrade :
                price_gap = self.avgPrice - close_price
                self.stopless = self.params.additionalEntryPrice - (2 * atr) 
                gap = price_gap >  self.stopless * self.entryCount
                # if gap :
                #     logger.info(f'gap : {gap}')
                
                if price_gap > self.stopless * self.entryCount :
                    
                    self.entryPriceSum = self.entryPriceSum + close_price
                    self.total_entry_count = self.total_entry_count + 1
                    self.avgPrice = self.entryPriceSum / self.total_entry_count
                    self.buy(size=position_size)
                    self.entryCount += 1
                    self.log(f"Entry {self.entryCount} at {close_price:.2f}, avgPrice: {self.avgPrice:.2f}")
        
        # 익절 / 청산 조건
        if self.entryCount >= 2 and (close_price > self.avgPrice):
            # 초기 투입 물량 제외하고 청산
            qty = (position_size * (self.entryCount - 1))
            qty_percent = qty / (position_size * self.entryCount) * 100
            qty_sell = (position_size * self.entryCount) * qty_percent
            self.sell(size=qty)
            self.log(f"Partial exit at {close_price:.2f}, avgPrice: {self.avgPrice:.2f}, entryCount : {self.entryCount}, , position : {position_size * self.entryCount}, qty : {qty}")
            self.entryCount = self.entryCount - (self.entryCount - 1)
            
        if self.entryCount == 1 and (close_price >= self.avgPrice * self.params.profit ):
            
            self.close()
            self.log(f"Exit all at {close_price:.2f}, avgPrice: {self.avgPrice:.2f}")
            self.entryCount = 0
            self.avgPrice = 0
            self.entryPriceSum = 0
            self.total_entry_count = 0
            
        if self.broker.getvalue() <= 0 :
            logger.warning(f'margin Call : {self.broker.getvalue()}, close : {close_price}')
            

        
    
    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        
        print(f'{dt.isoformat()} {txt}')
        
        


def prepare_backtrader_data(df):
    if df.empty:
        raise ValueError("No data fetched from Binance. Check API keys or network connection.")

    logger.info(f"Initial DataFrame: {len(df)} rows, from {df.index[0]} to {df.index[-1]}")

    df_backtrader = df[['open', 'high', 'low', 'close', 'volume']].resample('1min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logger.info(f"After resampling: {len(df_backtrader)} rows, from {df_backtrader.index[0]} to {df_backtrader.index[-1]}")

    df_backtrader.index = df_backtrader.index.tz_localize(None)
    df_backtrader.index.name = 'datetime'

    if not df_backtrader.index.is_monotonic_increasing:
        logger.warning("Index is not monotonic, sorting...")
        df_backtrader = df_backtrader.sort_index()

    df_backtrader = calculate_indicators(df_backtrader)
    logger.info(f"After indicators: {len(df_backtrader)} rows, NaN counts:\n{df_backtrader.isna().sum()}")

    df_backtrader = df_backtrader.loc[start:end]
    logger.info(f"After date filtering: {len(df_backtrader)} rows, from {df_backtrader.index[0]} to {df_backtrader.index[-1]}")

    if df_backtrader.empty:
        raise ValueError(f"No data available between {start} and {end}")

    for col in ['rsi', 'sma', 'ema', 'bb_upper', 'bb_lower', 'bb_mid', 'upperBB', 'lowerBB', 'upperKC', 'lowerKC']:
        if col in df_backtrader.columns:
            df_backtrader[col] = df_backtrader[col].bfill().fillna(0)
    df_backtrader['val'] = df_backtrader['val'].fillna(0)
    df_backtrader['bcolor'] = df_backtrader['bcolor'].fillna(0)
    df_backtrader['scolor'] = df_backtrader['scolor'].fillna(0)
    df_backtrader['sqzOn'] = df_backtrader['sqzOn'].fillna(0)
    df_backtrader['sqzOff'] = df_backtrader['sqzOff'].fillna(0)
    df_backtrader['noSqz'] = df_backtrader['noSqz'].fillna(0)
    df_backtrader['atr'] = df_backtrader['atr'].fillna(0)

    numeric_cols = [col for col in df_backtrader.columns if col not in ['sqzOn', 'sqzOff', 'noSqz']]
    for col in numeric_cols:
        df_backtrader[col] = pd.to_numeric(df_backtrader[col], errors='coerce')

    required_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'sma', 'ema',
                    'bb_upper', 'bb_lower', 'bb_mid', 'val', 'bcolor', 'scolor',
                    'highest_price', 'lowest_price', 'upperBB', 'lowerBB',
                    'upperKC', 'lowerKC', 'sqzOn', 'sqzOff', 'noSqz', 'atr']
    missing_cols = [col for col in required_cols if col not in df_backtrader.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df_backtrader = df_backtrader.dropna()
    if df_backtrader.empty:
        raise ValueError("DataFrame is empty after dropping NaNs")

    logger.info(f"DataFrame: {len(df_backtrader)} rows, from {df_backtrader.index[0]} to {df_backtrader.index[-1]}")
    return df_backtrader


def save_csvForOHLCV(df):
    current_path = Path(os.getcwd())
     # Save to CSV
    # csv_dir = Path('~/.알고리즘트레이딩btc').expanduser()
    result_dir = current_path / 'result_btc물타기'
    csv_path = result_dir / 'ohlcv.csv'
    result_dir.mkdir(parents=True, exist_ok=True)

    
    # ✅ timestamp 컬럼이 없고, 인덱스가 datetime이라면 컬럼으로 만들어주기
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    if csv_path.exists():
        os.remove(csv_path)
        logger.info(f"Deleted existing CSV: {csv_path}")

    if not df.empty:
        df.to_csv(csv_path, index=False)
        logger.info(f"Combined CSV saved: {csv_path}")
    else:
        logger.warning("No data to save to CSV.")
    
    
    
    

    
class ArithmeticReturns(bt.Analyzer): # ✅ 해당 클래스는 전략(bt.strategy를 파라미터로 받는 클래스)이 실행되기 전에 전략 값을 가져오므로 
                                      # 전략 실행 후의 값을 가져오고 싶으면 stop() 함수에서 값을 불러올 것
    def __init__(self):
        self.initial_cash = self.strategy.broker.getcash() # ✅ 초기 예산, 전략이 실행되기 전 이미 설정되어 있는 값이므로 init에서도 사용 가능
        self.net_profit_percent = 0.0
        self.net_profit = 0

    def notify_cashvalue(self, cash, value):
        self.final_value = value

    def stop(self):
        self.net_profit_percent = ((self.final_value - self.initial_cash) / self.initial_cash) * 100
        self.net_profit = self.final_value - self.initial_cash

    def get_analysis(self):
        return {'net_profit_percent': self.net_profit_percent,  'net_profit': self.net_profit}
                

class MonteCarloVaRAnalyzer(bt.Analyzer):
    def __init__(self):
        self.var_value = 0.0
        self.var_percent = 0.0
        self.var_history = []            # ✅ 거래별 VaR 저장 리스트
        self.avg_var_result = None       # ✅ 평균 저장 변수
        
    # def notify_strategy(self, strategy):
        
    #     self.var_history = strategy.var_history
        
    #     self.var_value = strategy.montecarlo_var_dollar
    #     self.var_percent = strategy.montecarlo_var_percent
    
        
    def stop(self):
        strategy = self.strategy  # 이제 완전히 attach 됨
        
        
        self.var_history = strategy.var_history
        self.var_value = strategy.montecarlo_var_dollar
        self.var_percent = strategy.montecarlo_var_percent
        
        
        if self.var_history:
            self.avg_var_result = np.mean(self.var_history)
            print(f"✅ 평균 VaR (총 {len(self.var_history)} 거래): ${self.avg_var_result:.2f}")
        else:
            print("❌ 거래 중 저장된 VaR 정보가 없습니다.")
            print(f'self.var_history : {self.var_history}')
            
    def get_analysis(self):
        return {
            'var_value': self.var_value,
            'var_percent': self.var_percent,
            'avg_var': self.avg_var_result,
            'var_history': self.var_history,
            
        }
        
        


def run_backtest(df):
    logger.info("run_backtest 시작")
    df_backtrader = prepare_backtrader_data(df)
    logger.info(f"df_backtrader 준비 완료: {len(df_backtrader)} 행, {df_backtrader.index[0]} ~ {df_backtrader.index[-1]}")


    mc_df = df.loc[monte_start:monte_end]
    
    
    if len(df_backtrader) < lookback_period:
        raise ValueError(f"데이터 부족: {len(df_backtrader)} 행, 필요: {lookback_period}")
    
    
    
    # logger.info(f'mc_df  : {mc_df}')
    
    bt_df = df_backtrader.loc[start:end]
    
    # logger.info(f'bt_df  : {bt_df}')
    
    # (헤지펀드에서 사용한다고하는 파라미터 값을 적용한 몬테카를로 시뮬레이션)
                
    if mc_df is None or mc_df.empty or len(mc_df['close']) < 2:
        raise ValueError("Monte Carlo 시뮬레이션용 close 데이터가 충분하지 않습니다.")
    else:
        var_dollar, var_pct, paths = monte_carlo_var_parallel(
            close_series=mc_df['close'],
            investment=(initial_capital * 10),
            confidence_level=0.99,
            days=10,
            num_simulations=100000,
            n_jobs=-1
        )
        
        
        
    logger.info(f'var_dollar : {var_dollar}')
    cerebro = bt.Cerebro()  # exactbars 제거
    logger.info("Cerebro 초기화 완료")

    data_feed = bt.feeds.PandasData(
        dataname=bt_df,
        fromdate=bt_df.index[0].to_pydatetime(),
        todate=bt_df.index[-1].to_pydatetime(),
        timeframe=bt.TimeFrame.Minutes,
        compression=5
    )
    
    
    
    
    logger.info("데이터 피드 생성 완료")

    cerebro.adddata(data_feed)
    logger.info("데이터 피드 추가 완료")

    
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.0005, leverage=leverage, margin=0 / leverage )
    cerebro.broker.set_slippage_perc(perc = 0.00015)
    logger.info("브로커 설정 완료")

    cerebro.addanalyzer(ArithmeticReturns, _name='arithmetic_returns')
    cerebro.addanalyzer(MonteCarloVaRAnalyzer, _name='mc_var')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    logger.info("분석기 추가 완료")

    try:
        cerebro.addstrategy(MLFuturesStrategy, lookback_period=lookback_period, montecarlo_var_dollar = var_dollar,
        montecarlo_var_percent = var_pct)  # lookback_period 축소
        logger.info("MLFuturesStrategy 전략 등록 완료")
    except Exception as e:
        logger.error(f"전략 등록 실패: {e}")
        raise

    logger.info(f"데이터 피드 범위: {bt_df.index[0]} ~ {bt_df.index[-1]}, 길이={len(bt_df)}")
    try:
        logger.info("백테스트 실행 시작")
        results = cerebro.run()
        logger.info("백테스트 실행 완료")
    except Exception as e:
        logger.error(f"백테스트 실행 실패: {e}")
        raise

    strat = results[0]
    mc_result = strat.analyzers.mc_var.get_analysis()
    avg_var = mc_result.get('avg_var', None)
    last_var = mc_result.get('var_value', None)
    var_per = mc_result.get('var_percent', None)
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0) or 0.0
    drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
    drawdown_money = strat.analyzers.drawdown.get_analysis().get('max', {}).get('moneydown')
    total_return = strat.analyzers.returns.get_analysis().get('rtot', 0.0) * 100
    arithmetic_profit_percent = strat.analyzers.arithmetic_returns.get_analysis().get('net_profit_percent', 0.0)
    arithmetic_profit = strat.analyzers.arithmetic_returns.get_analysis().get('net_profit', 0.0)
    trades = strat.analyzers.trades.get_analysis().get('total', {}).get('total', 0)

    logger.info(f"샤프 비율: {sharpe:.2f}")
    logger.info(f"최대 낙폭: {drawdown:.2f}%, {drawdown_money}")
    logger.info(f"총 수익률(로그): {total_return:.2f}%")
    logger.info(f"총 수익률(산술): {arithmetic_profit_percent:.2f}%")
    logger.info(f"총 수익: {arithmetic_profit:.2f}")
    logger.info(f"총 거래 횟수: {trades}")
    if avg_var is not None:
        logger.info(f"📊 평균 VaR (누적 전체 거래): ${avg_var:.2f}")
    
    if last_var is not None:
        logger.info(f"📌 마지막 거래의 VaR: ${last_var:.2f}")
  
    if var_per is not None:
        logger.info(f"📊 평균 VaR (누적 전체 거래) %: ${var_per:.2f}%")
   
    
    
    if trades == 0:
        logger.warning("백테스트 중 거래 실행되지 않음")

    return strat

def main():
    try:
        set_leverage_and_margin(trading_symbol, leverage)
        
        # 몬테카를로 모델 참고용 데이터를 함수에서 직접 불러올 경우
        # mc_df = fetch_ohlcv_between(
        #     symbol='BTC/USDT',
        #     timeframe='1m',
        #     start='2019-09-08',
        #     end='2020-12-31',
        #     max_bars=500_000   # 충분히 큰 값으로 설정
        # )
        
        
        # df = fetch_ohlcv_between(trading_symbol, timeframe, monte_start, end, max_bars=3985920)
        # save_csvForOHLCV(df)
        df = pd.read_csv('result_btc물타기/ohlcv.csv',
                        index_col='timestamp',      # 날짜 컬럼이름에 맞게 지정
                        parse_dates=['timestamp'],  # 날짜 컬럼을 실제 날짜타입으로 변환
                        encoding='utf-8'           # 필요시 인코딩 조정
                        )
        
        # save_csvForOHLCV(df)
        logger.info(f"Fetched data: {len(df)} rows, from {df.index[0]} to {df.index[-1]}")
        
        
        
        run_backtest(df)
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()