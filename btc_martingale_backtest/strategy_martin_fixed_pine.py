import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from squeeze_momentum_core import squeeze_momentum_core
from calc_var import calc_mean_var_from_df
from binance_calculator import calculate_liquidation_price, calculate_martingale_liquidation_price, validate_liquidation_risk, BinanceAveragePriceCalculator
import os

# logger 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class StrategyMartinFixedPine(bt.Strategy):
    """
    strategy_martin_fixed_pinescript.pine의 거래 로직을 기반으로 한 파이썬 버전
    """
    
    params = dict(
        inputTrade=10,  # 거래 투입 횟수/최대 값은 시드 분할 값과 같음
        profit=1.0098,  # 익절%
        profit_partial=1.005,  # 0.4%에서 1.5%로 상향 조정
        leverage=0,  # 포지션 크기 계산용 10배
        dividedLongCount=20,  # 시드 분할
        additionalEntryPrice=1500,  # 물타기 한도
        max_var=0.05,  # 12% (균형잡힌 설정)
        rf_threshold=0.7, # RandomForest 확률 임계값
        rf_threshold_partial=0.6, # RandomForest 확률 임계값
        mean_var=None,     # 전체 백테스팅 구간 평균 VaR
        max_var_dollar=1000,  # VaR 달러 기준 예시
        
        save_trade_log=True,  # 거래 로그 CSV 저장 여부
        trade_log_dir='trade_logs'  # 거래 로그 저장 디렉토리
    )
    
    def __init__(self):
        # 기본 변수 초기화
        self.entryCount = 0
        self.totalEntryCount = 0
        self.initialPositionSize = 0
        self.last_entry_bar = -100
        self.mean_var = self.p.mean_var
        self.var_history = []  # VaR 히스토리 저장용 리스트
        self.montecarlo_var_dollar = None
        self.montecarlo_var_percent = None
        self.margin_called = False  # 마진콜 상태 추적
        self.last_log_time = 0
        self.log_interval = 1000
        self.tick_count = 0
        self.enable_logging = False
        
        # 백테스팅 결과 저장용
        self.trade_history = []
        self.margin_history = []
        
        # 초기 자본 저장
        self.initial_capital = None
        
        # 포지션 크기 추적
        self.tracked_position_size = 0.0  # BTC 단위
        self.tracked_position_value = 0.0  # USD 단위
        
        # 초기 진입 수량 고정 저장
        self.initial_entry_size = None
        
        # 🆕 자본 분할 관리
        self.first_half_capital = None  # 첫 번째 절반 자본 (1-9번 진입용)
        self.second_half_capital = None  # 두 번째 절반 자본 (10번째 긴급 진입용)
        self.emergency_position_size = None  # 긴급 진입용 포지션 크기
        
        # 🆕 긴급 진입 플래그
        self.emergency_entry_executed = False  # 긴급 진입 실행 여부
        
        # 거래 로그 저장용
        self.trade_logs = []
        self.current_trade_id = 0
        self.trade_start_time = None
        self.trade_start_price = None
        self.trade_rf_pred = None
        self.trade_threshold = None
        
        # 🆕 바이낸스 평균가격 계산기 초기화
        self.binance_calculator = BinanceAveragePriceCalculator()
        
        # take_profit 변수 초기화
        self.take_profit = self.params.profit
        
        # 거래 로그 디렉토리 생성
        if self.p.save_trade_log:
            os.makedirs(self.p.trade_log_dir, exist_ok=True)
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'[{dt}] {txt}')

    def save_trade_log(self, action_type, **kwargs):
        """
        거래 로그를 CSV로 저장 (핵심 정보만)
        """
        if not self.p.save_trade_log:
            return
            
        dt = self.data.datetime.datetime(0)
        
        log_entry = {
            'timestamp': dt,
            'trade_id': self.current_trade_id,
            'action_type': action_type,  # 'entry', 'martingale', 'partial_exit', 'final_exit', 'margin_call'
            'price': self.data.close[0],
            'rf_pred': self.data.rf_pred[0],
            'rf_pred_down': self.data.rf_pred_down[0] if hasattr(self.data, 'rf_pred_down') else None,
            'entry_count': self.entryCount,
            'avg_price_binance': self.binance_calculator.get_average_price(),
            'position_size': self.tracked_position_size,
            'broker_value': self.broker.getvalue(),
            'profit_loss': None,
            'profit_ratio': None,
            'threshold': self.trade_threshold if self.trade_threshold else self.p.rf_threshold,
            'take_profit': self.take_profit if self.take_profit else self.p.profit,
            'threshold_partial': self.p.rf_threshold_partial,
        }
        
        # 수익률 계산 (청산 시에만)
        if action_type in ['final_exit', 'partial_exit'] and self.binance_calculator.get_average_price():
            binance_avg = self.binance_calculator.get_average_price()
            log_entry['profit_loss'] = self.data.close[0] - binance_avg
            log_entry['profit_ratio'] = (self.data.close[0] / binance_avg - 1) * 100
        
        # 추가 키워드 인자들 추가
        for key, value in kwargs.items():
            log_entry[key] = value
        
        self.trade_logs.append(log_entry)

    def save_trade_logs_to_csv(self):
        """
        거래 로그를 CSV 파일로 저장
        """
        if not self.p.save_trade_log or not self.trade_logs:
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'trade_logs_{timestamp}.csv'
        filepath = os.path.join(self.p.trade_log_dir, filename)
        
        df = pd.DataFrame(self.trade_logs)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"거래 로그 저장 완료: {filepath}")
        print(f"총 거래 로그 수: {len(self.trade_logs)}")
        
        return filepath
        
    def next(self):
        # 마진콜 상태면 모든 거래 중단
        if self.margin_called:
            return
            
        # 외부 피드: rf_pred(랜덤포레스트 예측값), rf_pred_down(하락 예측값), var(몬테카를로 VaR)
        rf_pred = self.data.rf_pred[0]
        rf_pred_down = self.data.rf_pred_down[0]  # 5% 하락 후 10% 하락할 확률
        var = self.data.var[0] 
        var_dollar = self.data.var_dollar[0] 
        atr = self.data.atr_14[0] 
        val = self.data.val[0] 
        
        close = self.data.close[0]
        open_price = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        
        # 동적 자본 분할 계산 (자본 증가 시 재계산)
        current_capital = self.broker.getvalue() * self.p.leverage
        
        if (self.first_half_capital is None or 
            abs(current_capital - (self.first_half_capital + self.second_half_capital)) > 1):
            
            self.first_half_capital = current_capital / 2  # 첫 번째 절반
            self.second_half_capital = current_capital / 2  # 두 번째 절반 (긴급 진입용)
        
        # 일반 진입용 포지션 크기 (첫 번째 절반 자본 사용)
        capital_per_once = self.first_half_capital / self.p.dividedLongCount
        position_size = round(capital_per_once / close * 1000) / 1000
        
        # 일반 진입 조건 (1-10번째 진입)
        if self.entryCount < self.p.inputTrade and position_size > 0:
            # First Entry
            if (val > 0 and self.entryCount == 0 and close > open_price):
                self.initialEntryPrice = close
                self.initialPositionSize = position_size  # 초기 진입 시 positionSize 저장
                
                # 바이낸스 평균가 계산
                self.binance_calculator.add_position(close, position_size)
                
                self.buy(size=position_size)
                self.entryCount = 1
                
                self.log(f"Entry 1 at {close}, avgPrice: {self.binance_calculator.get_average_price()}, initialPositionSize: {self.initialPositionSize}, 자본: {self.broker.getvalue():.2f}")
                
            # 일반 물타기 조건 (1-9번째 진입) - 긴급 진입 이후에는 실행하지 않음
            if (self.entryCount >= 1 and self.entryCount < self.p.inputTrade and not self.emergency_entry_executed):
                stoploss = self.p.additionalEntryPrice - (2 * atr)
                price_gap = self.binance_calculator.get_average_price() - close  # 바이낸스 평균가격 사용
                
                if price_gap > stoploss * self.entryCount:
                    self.secondEntryPrice = close
                    
                    # 바이낸스 평균가 계산 (증분 방식)
                    self.binance_calculator.add_position(close, position_size)
                    
                    self.buy(size=position_size)
                    self.entryCount += 1
                    
                    self.log(f"Entry {self.entryCount} at {close}, avgPrice: {self.binance_calculator.get_average_price()}, 자본: {self.broker.getvalue():.2f}")
        
        # 🆕 긴급 진입 조건 (별도 블록으로 분리)
        if (self.entryCount == self.p.inputTrade and not self.emergency_entry_executed):
            # 평균가에서 5% 이상 하락 시 현재까지 투입한 전체 포지션과 같은 수량을 추가 진입
            drop_percentage = ((self.binance_calculator.get_average_price() - close) / self.binance_calculator.get_average_price()) * 100
            
            if drop_percentage >= 20.0:
                # 🆕 현재까지 투입한 전체 포지션과 같은 수량을 긴급 진입
                total_position_size = self.binance_calculator.get_total_quantity()  # 현재까지 투입한 전체 포지션 크기
                emergency_position_size = total_position_size  # 전체 포지션과 같은 수량
                
                # 바이낸스 평균가 계산 (증분 방식)
                self.binance_calculator.add_position(close, emergency_position_size)
                
                self.buy(size=emergency_position_size)
                self.entryCount += 1
                
                # 🆕 긴급 진입 플래그 설정
                self.emergency_entry_executed = True
                
                self.log(f"🚨 긴급물타기 - Entry {self.entryCount} at {close}, avgPrice: {self.binance_calculator.get_average_price()}, 하락률: {drop_percentage}%, 진입수량: {emergency_position_size} (전체 포지션과 같은 수량)")
                self.log(f"🚨 긴급물타기 - 현재까지 투입한 총 포지션: {total_position_size}")
        
        # Partial Exit Logic (바이낸스 방식)
        if (self.entryCount >= 2 and close > self.binance_calculator.get_average_price() * 1.003 and self.binance_calculator.get_average_price() > 0):
           
            
            if self.initial_entry_size is not None:
                qty = self.tracked_position_size - self.initial_entry_size
            else:
                # fallback: 현재 포지션에서 새 진입 수량 제외
                qty = self.tracked_position_size - position_size


            # 바이낸스 방식: 매도 시 평균가 유지, 수량만 감소
            self.binance_calculator.remove_position(qty)
            
            self.close(size=qty)
            self.log(f"초기 투입 물량 빼고 청산 at {close}, avgPrice: {self.binance_calculator.get_average_price()}, qty: {qty}, position_size: {self.tracked_position_size}, 자본: {self.broker.getvalue():.2f}")
            self.entryCount = 1
        
        # 전체 청산
        if (self.entryCount == 1 and close >= (self.binance_calculator.get_average_price() * self.p.profit)):
            self.close()
            self.log(f"exit all at {close}, avgPrice: {self.binance_calculator.get_average_price()}, 자본: {self.broker.getvalue():.2f}")
            
            # 바이낸스 방식: 모든 포지션 청산 시 변수 초기화
            self.binance_calculator.reset()
            self.initialEntryPrice = 0
            self.secondEntryPrice = 0
            self.entryCount = 0
            
            # 🆕 자본 분할 초기화 (다음 거래를 위해)
            self.first_half_capital = None
            self.second_half_capital = None
            
            # 🆕 긴급 진입 플래그 초기화
            self.emergency_entry_executed = False 