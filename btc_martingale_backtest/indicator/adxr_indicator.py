import backtrader as bt
import numpy as np

class ADXR(bt.Indicator):
    """
    ADXR (Average Directional Index Rating) 지표 계산
    """
    lines = ('adxr', 'adx', 'diplus', 'diminus')
    params = (
        ('len', 14),
        ('lenx', 14),
    )
    
    def __init__(self):
        self.addminperiod(self.p.len + self.p.lenx)
        
    def next(self):
        if len(self) < self.p.len + self.p.lenx:
            return
            
        # TrueRange 계산
        tr = max(
            self.data.high[0] - self.data.low[0],
            abs(self.data.high[0] - self.data.close[-1]),
            abs(self.data.low[0] - self.data.close[-1])
        )
        
        # DMP, DMM 계산
        dmp = 0
        dmm = 0
        
        if self.data.high[0] - self.data.high[-1] > self.data.low[-1] - self.data.low[0]:
            dmp = max(self.data.high[0] - self.data.high[-1], 0)
        else:
            dmp = 0
            
        if self.data.low[-1] - self.data.low[0] > self.data.high[0] - self.data.high[-1]:
            dmm = max(self.data.low[-1] - self.data.low[0], 0)
        else:
            dmm = 0
        
        # Smoothed 값들 계산 (Wilder's smoothing)
        if len(self) == self.p.len + self.p.lenx:
            # 초기값 설정
            self.smoothed_tr = tr
            self.smoothed_dmp = dmp
            self.smoothed_dmm = dmm
        else:
            # Wilder's smoothing
            self.smoothed_tr = self.smoothed_tr - (self.smoothed_tr / self.p.len) + tr
            self.smoothed_dmp = self.smoothed_dmp - (self.smoothed_dmp / self.p.len) + dmp
            self.smoothed_dmm = self.smoothed_dmm - (self.smoothed_dmm / self.p.len) + dmm
        
        # DI+ and DI- 계산
        if self.smoothed_tr > 0:
            diplus = (self.smoothed_dmp / self.smoothed_tr) * 100
            diminus = (self.smoothed_dmm / self.smoothed_tr) * 100
        else:
            diplus = 0
            diminus = 0
        
        # DX 계산
        if diplus + diminus > 0:
            dx = abs(diplus - diminus) / (diplus + diminus) * 100
        else:
            dx = 0
        
        # ADX 계산 (SMA of DX)
        if len(self) >= self.p.len + self.p.lenx + self.p.len:
            dx_sum = sum([abs(self.lines.diplus[-i] - self.lines.diminus[-i]) / 
                         (self.lines.diplus[-i] + self.lines.diminus[-i]) * 100 
                         for i in range(1, self.p.len + 1) 
                         if self.lines.diplus[-i] + self.lines.diminus[-i] > 0])
            adx = dx_sum / self.p.len
        else:
            adx = dx
        
        # ADXR 계산
        if len(self) >= self.p.len + self.p.lenx + self.p.len:
            adxr = (adx + self.lines.adx[-self.p.lenx]) / 2
        else:
            adxr = adx
        
        # 라인에 값 할당
        self.lines.adxr[0] = adxr
        self.lines.adx[0] = adx
        self.lines.diplus[0] = diplus
        self.lines.diminus[0] = diminus














