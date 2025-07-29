import numpy as np
import pandas as pd
import talib

def squeeze_momentum_core(df, length=20, multKC=1, lengthKC=20, useTrueRange=True):
    close = df['close']
    high = df['high']
    low = df['low']
    open = df['open']
    # Bollinger Bands (not used in val, but for reference)
    # upperBB, middleBB, lowerBB = talib.BBANDS(close, timeperiod=length, nbdevup=2, nbdevdn=2, matype=0)
    # Keltner Channel (not used in val, but for reference)
    if useTrueRange:
        tr = talib.TRANGE(high, low, close)
    else:
        tr = high - low
    rangema = talib.SMA(tr, timeperiod=lengthKC)
    ma = talib.SMA(close, timeperiod=lengthKC)
    upperKC = ma + rangema * multKC
    lowerKC = ma - rangema * multKC
    # Squeeze Momentum (val)
    highest = talib.MAX(high, timeperiod=lengthKC)
    lowest = talib.MIN(low, timeperiod=lengthKC)
    sma = talib.SMA(close, timeperiod=lengthKC)
    avg = (highest + lowest) / 2
    diff = close - ((avg + sma) / 2)
    val = talib.LINEARREG(diff, timeperiod=lengthKC)
    return val 