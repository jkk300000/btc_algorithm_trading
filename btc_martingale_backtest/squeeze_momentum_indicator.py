import backtrader as bt
import numpy as np
import pandas as pd
from squeeze_momentum_core import squeeze_momentum_core

class SqueezeMomentum(bt.Indicator):
    lines = ('val',)
    params = (('length', 20), ('mult', 2.0), ('lengthKC', 20), ('multKC', 1), ('useTrueRange', True))

    def next(self):
        window = self.p.lengthKC
        if len(self.data) < window:
            self.lines.val[0] = float('nan')
            return
        close = np.array([self.data.close[-i] for i in reversed(range(window))])
        high = np.array([self.data.high[-i] for i in reversed(range(window))])
        low = np.array([self.data.low[-i] for i in reversed(range(window))])
        open_ = np.array([self.data.open[-i] for i in reversed(range(window))])
        df = pd.DataFrame({
            'close': close,
            'high': high,
            'low': low,
            'open': open_
        })
        val_series = squeeze_momentum_core(
            df,
            length=self.p.length,
            multKC=self.p.multKC,
            lengthKC=self.p.lengthKC,
            useTrueRange=self.p.useTrueRange
        )
        self.lines.val[0] = val_series.iloc[-1] 