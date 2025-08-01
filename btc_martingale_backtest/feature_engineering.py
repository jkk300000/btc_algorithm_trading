import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
from squeeze_momentum_core import squeeze_momentum_core
import talib
import matplotlib.pyplot as plt







def add_features(input_path, output_path=None, diagnose=True):
    """
    ta_lib 라이브러리를 활용해 진입 조건 및 ml 가격 상승 및 하락 예측에 사용할 특정 지표를 계산.
    
    """
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    # 기존 ta 패키지 기반 코드 (주석 처리)
    # df['rsi_7'] = RSIIndicator(df['close'], window=7).rsi()
    # df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
    # df['rsi_21'] = RSIIndicator(df['close'], window=21).rsi()
    # df['ema_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    # df['ema_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    # df['atr_14'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    # talib 기반으로 대체
    df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
    df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['val'] = squeeze_momentum_core(df)
    print(df['val'])
    # Bollinger Bands (basis, upperBB, lowerBB) 추가
    basis, bb_upper, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_basis'] = basis
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    # SMA(20)
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    # bcolor, scolor (val 기준)
    df['bcolor'] = (df['val'] > 0).astype(int)
    df['scolor'] = (df['val'] < 0).astype(int)
    # volume (없으면 0으로 채움)
    if 'volume' not in df.columns:
        df['volume'] = 0
    if diagnose:
        print("[add_features] 데이터 진단 결과:")
        print(f"전체 행 개수: {len(df)}")
        print("피처별 결측치 개수:")
        print(df.isna().sum())
        print("피처별 결측치 비율(%):")
        print((df.isna().sum() / len(df) * 100).round(2))
        print("결측치가 없는 행 개수:", len(df.dropna()))
    if output_path:
        df.to_csv(output_path)
        print(f'Saved to {output_path}')
    return df

if __name__ == '__main__':
    add_features('C:/선물데이터/binance_btcusdt_1m.csv', 'C:/선물데이터/binance_btcusdt_1m_features.csv')

    df = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_features.csv', index_col=0, parse_dates=True)

    # val 컬럼에 양수 값이 하나라도 있는지 확인
    has_positive_val = (df['val'] > 0).any()

    if has_positive_val:
        print("val 컬럼에 양수 값이 있습니다.")
    else:
        print("val 컬럼에 양수 값이 없습니다.")

   