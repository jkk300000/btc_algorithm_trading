import backtrader as bt
import pandas as pd
import numpy as np
from indicator.squeeze_momentum_core import squeeze_momentum_core
from indicator.calc_var import calc_mean_var_from_df
import logging
from binance.binance_calculator import calculate_liquidation_price, calculate_martingale_liquidation_price, validate_liquidation_risk, BinanceAveragePriceCalculator
import os
from datetime import datetime

# logger 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)




class ModifiedMartingaleStrategy(bt.Strategy):
    """
    🆕 올바른 바이낸스 청산가 공식이 적용된 물타기 전략
    
    초기 투자 자본을 7배 레버리지로 투자한다. 해당 투자 자본(레버리지를 적용한)을 20번 나누고 그 중 10개를 진입한다.
    초기 진입 조건은 엄격하게 설정하며, 물타기 조건은 초기 진입 조건 보다는 관대하게 적용한다.
    
    🆕 바이낸스 청산가 공식 적용:
    - Liquidation Price = Entry Price - ((Entry Price / Leverage) * (1 + Maintenance Margin))
    - Maintenance Margin = 0.5% (0.005)
    - 🆕 실제 사용 레버리지 = 총 포지션 가치 / 총 자산 (정확한 바이낸스 방식)
    
    랜덤 포레스트 상승 및 하락 예측 값을 활용하여 상승장 및 하락장 예측
    특정 지표를 활용하여 진입 조건을 설정한다.

    1. 초기 진입 수량 복리 계산 : 초기 투자 자본 * 레버리지 / 20 /
    2. 물타기 진입 조건
    3. 부분 청산 조건
    4. 전체 청산 조건
    5. 🆕 정확한 바이낸스 기준 마진콜 처리 (총포지션가치/총자산 레버리지)
    6. 거래 로그 저장   
    
    """
    
    params = dict(
        inputTrade=10,
        profit=1.008 ,
        profit_partial=1.0055,  # 0.4%에서 1.5%로 상향 조정
        leverage=0,  # 설정 레버리지 7배
        dividedLongCount=20,
        additionalEntryPrice=1500, 
        max_var=0.05,  # 12% (균형잡힌 설정)
        rf_threshold=0.9, # RandomForest 확률 임계값
        rf_threshold_partial=0.6, # RandomForest 확률 임계값
        # rf_threshold_down=0.9, # 하락 예측 임계값 (70% 이상이면 거래 회피)
        # rf_threshold_down_martingale=0.9, # 물타기 시 하락 예측 임계값 (80% 이상이면 물타기 회피) 
        mean_var=None,     # 전체 백테스팅 구간 평균 VaR
        max_var_dollar=1000,  # VaR 달러 기준 예시
        drop_percentage=20.0,
        partial_exit_profit=1.003,
        save_trade_log=True,  # 거래 로그 CSV 저장 여부
        trade_log_dir='trade_logs'  # 거래 로그 저장 디렉토리
    )

    def __init__(self):
        
        
        
        
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
        
        # 🆕 초기화 완료 로그
        print(f"🚀 ModifiedMartingaleStrategy 초기화 완료")
        print(f"🚀 설정 레버리지: {self.p.leverage}")
        print(f"🚀 설정 파라미터: inputTrade={self.p.inputTrade}, dividedLongCount={self.p.dividedLongCount}")

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
            'action_type': action_type,  # 'entry', 'martingale', 'partial_exit', 'final_exit', 'margin_call', 'liquidation_risk_warning'
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
            # 'threshold_down': self.p.rf_threshold_down,
            # 'threshold_down_martingale': self.p.rf_threshold_down_martingale
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
        
        # 🆕 액션 타입별 로그 수 통계
        action_counts = {}
        for log in self.trade_logs:
            action_type = log.get('action_type', 'unknown')
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        print(f"액션 타입별 로그 수:")
        for action_type, count in action_counts.items():
            print(f"  - {action_type}: {count}건")
        
        return filepath

    def next(self):
        # 마진콜 상태면 모든 거래 중단
        if self.margin_called:
            return
            
        # self.log(f'거래 시작')
        # 외부 피드: rf_pred(랜덤포레스트 예측값), rf_pred_down(하락 예측값), var(몬테카를로 VaR)
        rf_pred = self.data.rf_pred[0]
        rf_pred_down = self.data.rf_pred_down[0]  # 5% 하락 후 10% 하락할 확률
        var = self.data.var[0] 
        var_dollar = self.data.var_dollar[0] 
        atr = self.data.atr_14[0] 
        val = self.data.val[0] 
        close = self.data.close[0]
        open = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        
        highest_price = self.data.high[-1] if len(self.data) > 1 else high
        # 진입 조건 등에서 ema9, ema21을 활용할 수 있음
        # mean_var 예시 활용
        # if self.mean_var is not None and var < self.mean_var:
        #     ...
        


        # 🆕 동적 자본 분할 계산 (방법 1 적용)
        # 백테스터 현재 자본을 기준으로 일관성 있게 계산
        self.initial_capital = self.broker.getvalue()  # 백테스터 현재 자본
        
        # 🆕 자본 분할 디버깅 로그 (매 틱마다)
        # if self.tick_count % 100 == 0:  # 100틱마다 출력
        #     self.log(f"💰 자본분할 디버깅 - 현재자본: {self.initial_capital:.2f}, 초기자본: {self.initial_capital:.2f}")
        #     if self.first_half_capital is not None:
        #         self.log(f"💰 자본분할 디버깅 - first_half: {self.first_half_capital:.2f}, second_half: {self.second_half_capital:.2f}")
        
        # 자본이 변경되었거나 초기 설정인 경우 재계산
        if (self.first_half_capital is None or 
            abs(self.initial_capital - (self.first_half_capital + self.second_half_capital)) > 1):
            
            self.first_half_capital = self.initial_capital / 2  # 첫 번째 절반
            self.second_half_capital = self.initial_capital / 2  # 두 번째 절반 (긴급 진입용)
            self.emergency_position_size = None  # 긴급 진입 크기 재계산 필요
            
            # self.log(f"💰 자본 분할 재설정 - 현재자본: {self.initial_capital:.2f}")
            # self.log(f"💰 자본 분할 재설정 - 첫 번째 절반: {self.first_half_capital:.2f}, 두 번째 절반: {self.second_half_capital:.2f}")
        
        # 일반 진입용 포지션 크기 (레버리지 반영)
        # 레버리지를 적용한 실제 투자 가능 자본으로 포지션 크기 계산
        leveraged_capital_per_once = (self.first_half_capital * self.p.leverage) / self.p.dividedLongCount
        positionSize = leveraged_capital_per_once / close
        positionSize = np.round(positionSize, 3)
        
        # 긴급 진입용 포지션 크기 (레버리지 반영)
        if self.emergency_position_size is None:
            emergency_leveraged_capital = (self.second_half_capital * self.p.leverage) / self.p.dividedLongCount
            self.emergency_position_size = emergency_leveraged_capital / close
            self.emergency_position_size = np.round(self.emergency_position_size, 3)
            # self.log(f"🚨 긴급 진입용 포지션 크기 재설정: {self.emergency_position_size}")


        # 틱 카운트 증가
        self.tick_count += 1
            # 진입 조건별 값 로그 출력
        # self.log(f"val: {val}, entryCount: {self.entryCount}, totalEntryCount: {self.totalEntryCount}, close: {close}, open: {open_}, rf_pred: {rf_pred}, var: {var}, mean_var: {self.mean_var}, initial_capital: {initial_capital}, var_dollar: {var_dollar}")
        # self.log(f'initial_capital: {initial_capital}, capitalPerOnce: {capitalPerOnce}, positionSize: {self.positionSize}')
        self.montecarlo_var_dollar = var_dollar
        self.montecarlo_var_percent = var


        can_enter = self.entryCount < self.params.inputTrade and positionSize > 0
        entry_condition = (
            can_enter and
            val > 0 and
            self.entryCount == 0 and
            self.totalEntryCount == 0 and
            close > open 
            # and
            # abs(var) <= self.p.max_var 
            # rf_pred_down < self.p.rf_threshold_down
        )

        if entry_condition:
            # 익절 목표 분기
            if rf_pred >= self.p.rf_threshold:
                self.take_profit = self.p.profit  # 0.8% 익절
            else:
                self.take_profit = self.p.profit_partial  # 0.5% 익절

            # 거래 ID 생성
            self.current_trade_id += 1
            self.trade_start_time = self.data.datetime.datetime(0)
            self.trade_start_price = close
            self.trade_rf_pred = rf_pred
            self.trade_threshold = self.p.rf_threshold

            # 진입 로직 (공통)
            if self.initial_capital is None:
                #백테스터 현재 자본을 초기 자본으로 설정 (방법 1 적용)
                self.initial_capital = self.broker.getvalue()
                # self.log(f"💰 초기 자본 설정: {self.initial_capital:.2f}")
                # self.log(f"💰 백테스터 현재 자본: {self.broker.getvalue():.2f}")
                # self.log(f"💰 설정 레버리지: {self.p.leverage}")
                # self.log(f"💰 자본 일관성 적용: 백테스터 현재 자본 기준")
            
            # 🆕 초기 진입 디버깅 로그 추가
            # self.log(f"[진입 디버깅] 진입 전 상태:")
            # self.log(f"[진입 디버깅] - entryCount: {self.entryCount}")
            # self.log(f"[진입 디버깅] - positionSize: {positionSize:.6f}")
            # self.log(f"[진입 디버깅] - leveraged_capital_per_once: ${(self.first_half_capital * self.p.leverage) / self.p.dividedLongCount:.2f}")
            # self.log(f"[진입 디버깅] - first_half_capital: ${self.first_half_capital:.2f}")
            
            # 🆕 바이낸스 평균가격 계산기 사용
            
            self.binance_calculator.add_position(close, positionSize)
            self.buy(size=positionSize)
            self.entryCount += 1
            self.var_history.append(var_dollar)
            self.tracked_position_size += positionSize
            self.tracked_position_value = self.tracked_position_size * self.binance_calculator.get_average_price()
            if self.initial_entry_size is None:
                self.initial_entry_size = positionSize
            
            # 🆕 초기 진입 후 디버깅 로그
            # self.log(f"[진입 디버깅] 진입 후 상태:")
            # self.log(f"[진입 디버깅] - entryCount: {self.entryCount}")
            # self.log(f"[진입 디버깅] - tracked_position_size: {self.tracked_position_size:.6f}")
            # self.log(f"[진입 디버깅] - tracked_position_value: {self.tracked_position_value:.2f}")
            # self.log(f"[진입 디버깅] - 바이낸스 총수량: {self.binance_calculator.get_total_quantity():.6f}")
            
            # 로그 출력 (바이낸스 평균가격 사용)
            binance_avg = self.binance_calculator.get_average_price()
            self.log(f"[진입] 진입가: {close}, 바이낸스평균가: {binance_avg:.2f}, entryCount: {self.entryCount}, 자본: {self.broker.getvalue():.2f}, positionSize: {positionSize}")
            self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
            
            # 거래 로그 저장
            self.save_trade_log('entry', 
                               entry_price=close, 
                               position_size=positionSize,
                               rf_pred=rf_pred,
                            #    rf_pred_down=rf_pred_down,
                               threshold=self.p.rf_threshold,
                               take_profit=self.take_profit,
                               capital_used='first_half')  # 첫 번째 절반 자본 사용 표시

        # 일반 물타기 조건 (1-9번째 진입) - 긴급 진입 이후에는 실행하지 않음
        if self.entryCount >= 1 and self.entryCount < self.p.inputTrade and not self.emergency_entry_executed:
            # self.log(f'물타기 조건')
            stoploss = self.p.additionalEntryPrice - (2 * atr)
            price_gap = self.binance_calculator.get_average_price() - close  # 🆕 바이낸스 평균가격 사용
            
            if price_gap > stoploss * self.entryCount and rf_pred >= self.p.rf_threshold_partial:
                # # 🆕 물타기 진입 디버깅 로그 추가
                # self.log(f"[물타기 디버깅] 진입 전 상태:")
                # self.log(f"[물타기 디버깅] - entryCount: {self.entryCount}")
                # self.log(f"[물타기 디버깅] - positionSize: {positionSize:.6f}")
                # self.log(f"[물타기 디버깅] - tracked_position_size: {self.tracked_position_size:.6f}")
                # self.log(f"[물타기 디버깅] - tracked_position_value: {self.tracked_position_value:.2f}")
                
                # 🆕 바이낸스 평균가격 계산기 사용
                self.binance_calculator.add_position(close, positionSize)
                
                self.buy(size=positionSize)
                self.entryCount += 1
                self.var_history.append(var_dollar)  # VaR 히스토리에 저장
                
                # 포지션 크기 추적
                self.tracked_position_size += positionSize
                self.tracked_position_value = self.tracked_position_size * self.binance_calculator.get_average_price()
                
                # 🆕 물타기 진입 후 디버깅 로그
                # self.log(f"[물타기 디버깅] 진입 후 상태:")
                # self.log(f"[물타기 디버깅] - entryCount: {self.entryCount}")
                # self.log(f"[물타기 디버깅] - tracked_position_size: {self.tracked_position_size:.6f}")
                # self.log(f"[물타기 디버깅] - tracked_position_value: {self.tracked_position_value:.2f}")
                # self.log(f"[물타기 디버깅] - 바이낸스 총수량: {self.binance_calculator.get_total_quantity():.6f}")
                
                # 로그 출력 (바이낸스 평균가격 사용)
                binance_avg = self.binance_calculator.get_average_price()
                self.log(f"[물타기] 진입가: {close}, 바이낸스평균가: {binance_avg:.2f}, entryCount: {self.entryCount}")
                self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
                
                # 거래 로그 저장
                self.save_trade_log('martingale', 
                                    entry_price=close, 
                                    position_size=positionSize,
                                    avg_price_binance=binance_avg,
                                    entry_count=self.entryCount,
                                    rf_pred_partial=self.p.rf_threshold_partial,
                                    capital_used='first_half'  # 첫 번째 절반 자본 사용 표시
                                    # rf_pred_down=rf_pred_down
                                    )
        
        # 🆕 10개 포지션이 모두 진입된 상태에서 평균가 대비 -10%일 때 추가 긴급 진입
        if self.entryCount == self.p.inputTrade and not self.emergency_entry_executed:  # 10개 포지션이 모두 진입된 상태이고 긴급 진입이 아직 실행되지 않았을 때
            # 평균가에서 10% 이상 하락 시 현재까지 투입한 전체 포지션과 같은 수량을 추가 진입
            drop_percentage = ((self.binance_calculator.get_average_price() - close) / self.binance_calculator.get_average_price()) * 100
            
            if drop_percentage >= self.p.drop_percentage and rf_pred >= self.p.rf_threshold_partial:
                # 🆕 현재까지 투입한 전체 포지션과 같은 수량을 긴급 진입
                total_position_size = self.tracked_position_size  # 현재까지 투입한 전체 포지션 크기
                emergency_position_size = total_position_size  # 전체 포지션과 같은 수량
                
                # 🆕 긴급 물타기 디버깅 로그 추가
                # self.log(f"[🚨 긴급물타기 디버깅] 진입 전 상태:")
                # self.log(f"[🚨 긴급물타기 디버깅] - entryCount: {self.entryCount}")
                # self.log(f"[🚨 긴급물타기 디버깅] - tracked_position_size: {self.tracked_position_size:.6f}")
                # self.log(f"[🚨 긴급물타기 디버깅] - tracked_position_value: {self.tracked_position_value:.2f}")
                # self.log(f"[🚨 긴급물타기 디버깅] - 바이낸스 총수량: {self.binance_calculator.get_total_quantity():.6f}")
                # self.log(f"[🚨 긴급물타기 디버깅] - 바이낸스 평균가: {self.binance_calculator.get_average_price():.2f}")
                # self.log(f"[🚨 긴급물타기 디버깅] - 긴급물타기 수량: {emergency_position_size:.6f}")
                
                # 🆕 바이낸스 평균가격 계산기 사용
                self.binance_calculator.add_position(close, emergency_position_size)
                
                # 🆕 긴급 물타기 포지션 크기 설정 (개선됨)
                self.binance_calculator.set_emergency_position_size(emergency_position_size)
                
                self.buy(size=emergency_position_size)
                self.entryCount += 1  # 11번째 진입으로 카운트
                self.var_history.append(var_dollar)  # VaR 히스토리에 저장
                
                # 🆕 긴급 진입 플래그 설정
                self.emergency_entry_executed = True
                
                # 포지션 크기 추적
                self.tracked_position_size += emergency_position_size
                self.tracked_position_value = self.tracked_position_size * self.binance_calculator.get_average_price()
                
                # 🆕 긴급 물타기 실행 후 디버깅 로그
                # self.log(f"[🚨 긴급물타기 실행 후] 진입 후 상태:")
                # self.log(f"[🚨 긴급물타기 실행 후] - entryCount: {self.entryCount}")
                # self.log(f"[🚨 긴급물타기 실행 후] - tracked_position_size: {self.tracked_position_size:.6f}")
                # self.log(f"[🚨 긴급물타기 실행 후] - tracked_position_value: {self.tracked_position_value:.2f}")
                # self.log(f"[🚨 긴급물타기 실행 후] - 바이낸스 총수량: {self.binance_calculator.get_total_quantity():.6f}")
                # self.log(f"[🚨 긴급물타기 실행 후] - 바이낸스 평균가: {self.binance_calculator.get_average_price():.2f}")
                
                # 로그 출력 (바이낸스 평균가격 사용)
                binance_avg = self.binance_calculator.get_average_price()
                self.log(f"[🚨 긴급물타기] 진입가: {close}, 바이낸스평균가: {binance_avg:.2f}, entryCount: {self.entryCount}")
                self.log(f"[🚨 긴급물타기] 하락률: {drop_percentage:.1f}%, 진입수량: {emergency_position_size:.6f} (전체 포지션과 같은 수량)")
                self.log(f"[🚨 긴급물타기] 현재까지 투입한 총 포지션: {total_position_size:.6f}")
                self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
                
                # 거래 로그 저장
                self.save_trade_log('emergency_martingale', 
                                    entry_price=close, 
                                    position_size=emergency_position_size,
                                    avg_price_binance=binance_avg,
                                    entry_count=self.entryCount,
                                    drop_percentage=drop_percentage,
                                    rf_pred_partial=self.p.rf_threshold_partial,
                                    total_position_before=total_position_size,
                                    capital_used='total_position_100%')  # 전체 포지션과 같은 수량 사용 표시
        # 부분 청산
        if self.entryCount >= 2 and (close > self.binance_calculator.get_average_price() * self.p.partial_exit_profit):  # 🆕 바이낸스 평균가격 사용
            # 🆕 긴급 물타기 이후 청산 로직 분기
            if self.emergency_entry_executed:
                # 긴급 물타기 수량은 평균가에 청산 (개선됨)
                emergency_qty = self.binance_calculator.get_emergency_position_size()
                if emergency_qty > 0:
                    # 긴급 물타기 수량만 먼저 청산
                    self.binance_calculator.remove_position(emergency_qty)
                    self.sell(size=emergency_qty)
                    
                    # 포지션 크기 추적 (긴급 물타기 수량 청산)
                    self.tracked_position_size -= emergency_qty
                    self.tracked_position_value = self.tracked_position_size * self.binance_calculator.get_average_price()
                    
                    binance_avg = self.binance_calculator.get_average_price()
                    self.log(f"[🚨 긴급물타기 청산] 청산가: {close}, 바이낸스평균가: {binance_avg:.2f}, 긴급물타기 청산 수량: {emergency_qty}")
                    self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
                    
                    # 거래 로그 저장
                    if binance_avg > 0:
                        profit_ratio = (close / binance_avg - 1) * 100
                    else:
                        profit_ratio = 0.0
                        
                    self.save_trade_log('emergency_exit', 
                                       exit_price=close, 
                                       exit_qty=emergency_qty,
                                       remaining_avg_price=binance_avg,
                                       profit_ratio=profit_ratio,
                                       exit_type='emergency_at_avg')
                    
                    # 긴급 진입 플래그 해제 (긴급 물타기 수량 청산 완료)
                    self.emergency_entry_executed = False
                    
                    # entryCount 조정
                    self.entryCount = self.p.inputTrade  # 10개로 되돌림
                    
                    # 긴급 물타기 수량이 청산된 후, 나머지 포지션은 기존 로직으로 처리
                    return
            
            # 🆕 기존 부분 청산 로직 (긴급 물타기 이후가 아닌 경우)
            # 초기 진입 수량을 제외한 나머지 부분청산
            if self.initial_entry_size is not None:
                qty = self.tracked_position_size - self.initial_entry_size
            else:
                # fallback: 현재 포지션에서 새 진입 수량 제외
                qty = self.tracked_position_size - positionSize
            
            # 🆕 바이낸스 평균가격 계산기에서 포지션 제거
            self.binance_calculator.remove_position(qty)
            
            self.sell(size=qty)
            
            # 포지션 크기 추적 (부분 청산)
            self.tracked_position_size -= qty
            self.tracked_position_value = self.tracked_position_size * self.binance_calculator.get_average_price()
                
            binance_avg = self.binance_calculator.get_average_price()
            self.log(f"[부분청산] 청산가: {close}, 바이낸스평균가: {binance_avg:.2f}, entryCount: {self.entryCount}, 자본: {self.broker.getvalue():.2f}, 부분 청산 수량: {qty}")
            self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
            
            # 거래 로그 저장 (0으로 나누기 방지)
            if binance_avg > 0:
                profit_ratio = (close / binance_avg - 1) * 100
            else:
                profit_ratio = 0.0
                
            self.save_trade_log('partial_exit', 
                               exit_price=close, 
                               exit_qty=qty,
                               remaining_avg_price=binance_avg,
                               profit_ratio=profit_ratio,
                               exit_type='normal_partial')
            
            # 부분 청산 후 entryCount 조정 (바이낸스 계산기와 일치)
            remaining_quantity = self.binance_calculator.get_total_quantity()
            if remaining_quantity > 0:
                # 남은 수량이 있으면 entryCount를 1로 설정
                self.entryCount = 1
            else:
                # 모든 포지션이 청산되면 초기화
                self.entryCount = 0
                self.initial_entry_size = None
        
        # 전체 청산
        if self.entryCount == 1 and (close >= self.binance_calculator.get_average_price() * self.take_profit):  # 🆕 바이낸스 평균가격 사용
            # 청산 전 평균가 저장
            final_avg_price = self.binance_calculator.get_average_price()
            
            # 전체 청산 실행
            self.close()
            
            # 🆕 바이낸스 평균가격 계산기에서 모든 포지션 제거
            self.binance_calculator.close_all_positions()
            
            # 🆕 긴급 물타기 포지션 크기 초기화
            self.binance_calculator.clear_emergency_position_size()
            
            # 포지션 크기 추적 (전체 청산)
            self.tracked_position_size = 0.0
            self.tracked_position_value = 0.0

            self.initial_capital = None
            
            # 거래 로그 저장 (0으로 나누기 방지)
            if final_avg_price > 0:
                profit_ratio = (close / final_avg_price - 1) * 100
            else:
                profit_ratio = 0.0
                
            self.save_trade_log('final_exit', 
                               exit_price=close, 
                               profit_ratio=profit_ratio,
                               trade_duration=(self.data.datetime.datetime(0) - self.trade_start_time).total_seconds() / 3600 if self.trade_start_time else None,
                               final_avg_price=final_avg_price)
            
            # 모든 변수 초기화
            self.entryCount = 0
            
            # 초기 진입 수량 초기화
            self.initial_entry_size = None
            self.take_profit = self.params.profit  # 초기화
            
            # 🆕 자본 분할 초기화 (다음 거래를 위해)
            self.first_half_capital = None
            self.second_half_capital = None
            self.emergency_position_size = None
            
            # 🆕 긴급 진입 플래그 초기화
            self.emergency_entry_executed = False
            
            self.log(f"[최종청산] 청산가: {close}, 바이낸스평균가: {final_avg_price:.2f}, entryCount: {self.entryCount}, 자본: {self.broker.getvalue():.2f}")
            self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")

       
            
           

        # self.log(f"[상태체크] 자본: {self.broker.getvalue():.2f}, 포지션: {self.position.size}, entryCount: {self.entryCount}")


        # 바이낸스 기준 마진콜 처리
        if (self.entryCount > 0 and 
            self.binance_calculator.get_average_price() is not None and 
            self.binance_calculator.get_average_price() > 0 and 
            not self.margin_called):
            # 추적된 포지션 크기 사용
            position_size_btc = self.tracked_position_size
            current_position_value = self.tracked_position_value
            
            # 🆕 올바른 바이낸스 청산가 공식 기반 동적 레버리지 계산
            # 설정 레버리지가 0이면 기본값 7배 사용
            base_leverage = self.p.leverage if self.p.leverage > 0 else 7
            
            # 🆕 실제 레버리지 = 총 포지션 가치 / 총 자산 (정확한 바이낸스 공식)
            # 이는 바이낸스 청산가 공식에서 사용하는 실제 사용 레버리지와 동일
            # 공식: actual_leverage = total_position_value / total_assets
            # 현재 총 자산을 사용하여 실시간 레버리지 계산
            actual_leverage = current_position_value / self.initial_capital
            
            # 마진콜 가능성 판단
            margin_call_possible = actual_leverage > 1.0
            theoretical_liquidation_drop = 100 / actual_leverage if actual_leverage > 0 else float('inf')
            
            # 🆕 레버리지가 너무 낮으면 청산가 계산 건너뛰기
            min_leverage_for_liquidation = 1.0  # 청산가 계산을 위한 최소 레버리지
            if actual_leverage < min_leverage_for_liquidation:
                # 레버리지가 낮으면 청산가 계산하지 않음
                # if self.tick_count % 200 == 0:  # 200틱마다 출력
                #     self.log(f"🔍 레버리지 부족 - entryCount: {self.entryCount}, 실제레버리지: {actual_leverage:.3f}배, 청산가 계산 생략")
                return
            
            # 🆕 레버리지 상세 정보 로그 (디버깅용)
            # if self.tick_count % 200 == 0:  # 200틱마다 출력
            #     self.log(f"🔍 레버리지 상세 - entryCount: {self.entryCount}, 포지션가치: ${current_position_value:.2f}, 초기자본: ${self.initial_capital:.2f}")
            #     self.log(f"🔍 레버리지 계산 - 실제레버리지: {actual_leverage:.2f}배, 청산가 계산 가능")
            #     self.log(f"🔍 청산가 정보 - 평균가: ${self.binance_calculator.get_average_price():.2f}, 현재가: ${close:.2f}")
                
            #     # 🆕 상세 디버깅 정보 추가
            #     self.log(f"🔍 디버깅 - tracked_position_size: {self.tracked_position_size:.6f}, close: {close:.2f}")
            #     self.log(f"🔍 디버깅 - 바이낸스 총수량: {self.binance_calculator.get_total_quantity():.6f}")
            #     self.log(f"🔍 디버깅 - 긴급물타기수량: {self.binance_calculator.get_emergency_position_size():.6f}")
            #     self.log(f"🔍 디버깅 - 레버리지 계산식: {current_position_value:.2f} / {self.initial_capital:.2f} = {actual_leverage:.2f}")
                
            #     # 🆕 자본 분할 디버깅 정보 추가
            #     self.log(f"🔍 자본분할 - first_half: ${self.first_half_capital:.2f}, second_half: ${self.second_half_capital:.2f}")
            #     self.log(f"🔍 자본분할 - 레버리지적용 capitalPerOnce: ${(self.first_half_capital * self.p.leverage) / self.p.dividedLongCount:.2f}")
            #     self.log(f"🔍 자본분할 - positionSize: {(self.first_half_capital * self.p.leverage) / self.p.dividedLongCount / close:.6f}")
            
            # 🆕 올바른 바이낸스 청산가 계산 (레버리지가 충분할 때만)
            # 공식: Entry Price - ((Entry Price / Leverage) * (1 + Maintenance Margin))
            # Maintenance Margin = 0.5% (0.005)
            liquidation_price = calculate_martingale_liquidation_price(
                self.binance_calculator.get_average_price(), 
                current_position_value, 
                actual_leverage,  # ✅ 실제 레버리지 사용
                self.initial_capital
            )
            
            # 청산가가 유효하지 않으면 건너뛰기
            if liquidation_price <= 0 or liquidation_price == float('inf'):
                return
            
            # 청산가 근처 경고 (청산가의 10% 이내)
            risk_info = validate_liquidation_risk(close, liquidation_price, warning_threshold=0.1)
            if risk_info['warning'] and self.tick_count % 100 == 0:
                self.log(f"⚠️ 청산가 근처 - 현재가: {close:.2f}, 청산가: {liquidation_price:.2f}, 거리: {risk_info['distance_percentage']:.1f}%, 위험도: {risk_info['risk_level']}")
                self.log(f"⚠️ 레버리지 정보 - 진입횟수: {self.entryCount}, 실제레버리지: {actual_leverage:.1f}배, 이론상청산하락률: {theoretical_liquidation_drop:.1f}%")
                
                # 현재 하락률 계산
                current_drop_percentage = ((self.binance_calculator.get_average_price() - close) / self.binance_calculator.get_average_price()) * 100
                self.log(f"⚠️ 현재 하락률: {current_drop_percentage:.1f}% (청산까지 {theoretical_liquidation_drop - current_drop_percentage:.1f}% 여유)")
                
                # 🆕 risk_info 발생 시 trade_logs CSV 파일에 저장 (개선됨)
                self.save_trade_log('liquidation_risk_warning', 
                                   liquidation_price=liquidation_price,
                                   distance_percentage=risk_info['distance_percentage'],
                                   risk_level=risk_info['risk_level'],
                                   current_drop_percentage=current_drop_percentage,
                                   theoretical_liquidation_drop=theoretical_liquidation_drop,
                                   actual_leverage=actual_leverage,
                                   warning_threshold=0.1)
            
            # 바이낸스 기준 마진콜 조건
            if close <= liquidation_price:
                current_drop_percentage = ((self.binance_calculator.get_average_price() - close) / self.binance_calculator.get_average_price()) * 100
                
                self.log(f'🚨 🆕 올바른 바이낸스 기준 마진콜! 현재가: {close:.2f}, 청산가: {liquidation_price:.2f}')
                self.log(f'🚨 마진콜 상세 - 평단가: {self.binance_calculator.get_average_price():.2f}, 포지션가치: {current_position_value:.2f}, 진입횟수: {self.entryCount}')
                self.log(f'🚨 레버리지 정보 - 실제레버리지: {actual_leverage:.1f}배, 이론상청산하락률: {theoretical_liquidation_drop:.1f}%')
                self.log(f'🚨 손실률: {current_drop_percentage:.2f}% (이론상 청산률: {theoretical_liquidation_drop:.1f}%)')
                
                # 🆕 긴급 물타기 상태 정보 추가
                if self.emergency_entry_executed:
                    emergency_qty = self.binance_calculator.get_emergency_position_size()
                    self.log(f'🚨 긴급 물타기 상태 - 긴급물타기수량: {emergency_qty:.6f}, 긴급물타기실행여부: {self.emergency_entry_executed}')
                
                # 거래 로그 저장 (개선됨)
                self.save_trade_log('margin_call', 
                                   liquidation_price=liquidation_price,
                                   current_drop_percentage=current_drop_percentage,
                                   actual_leverage=actual_leverage,
                                   emergency_entry_executed=self.emergency_entry_executed,
                                   emergency_position_size=self.binance_calculator.get_emergency_position_size())
                
                self.margin_called = True
                if self.position.size != 0:
                    self.close()
                    self.log(f'🚨 마진콜로 인한 강제 청산 완료')
                
                # 🆕 바이낸스 계산기 초기화 추가
                self.binance_calculator.close_all_positions()
                
                # 🆕 긴급 물타기 포지션 크기 초기화
                self.binance_calculator.clear_emergency_position_size()
                
                # 포지션 크기 초기화
                self.tracked_position_size = 0.0
                self.tracked_position_value = 0.0
                
                self.entryCount = 0
                
                # 초기 진입 수량 초기화
                self.initial_entry_size = None
                
                # 🆕 자본 분할 초기화 (마진콜 후 다음 거래를 위해)
                self.first_half_capital = None
                self.second_half_capital = None
                self.emergency_position_size = None
                
                # 🆕 긴급 진입 플래그 초기화
                self.emergency_entry_executed = False
                return

    def stop(self):
        """
        백테스팅 종료 시 거래 로그 저장
        """
        if self.p.save_trade_log:
            log_filepath = self.save_trade_logs_to_csv()
            if log_filepath:
                print(f"거래 로그가 저장되었습니다: {log_filepath}")
        
        # 🆕 바이낸스 평균가격 계산기 최종 통계 출력
        print(f"\n=== 바이낸스 평균가격 계산기 최종 통계 ===")
        print(f"총 거래 횟수: {len(self.trade_logs)}")
        print(f"최종 평균가격: ${self.binance_calculator.get_average_price():.2f}")
        print(f"최종 포지션 수량: {self.binance_calculator.get_total_quantity():.6f}")
        print(f"최종 포지션 가치: ${self.binance_calculator.get_total_value():.2f}")
        
        # 🆕 자본 분할 정보 출력
        if self.first_half_capital is not None:
            print(f"\n=== 자본 분할 정보 ===")
            print(f"첫 번째 절반 자본: ${self.first_half_capital:.2f} (1-9번 진입용)")
            print(f"두 번째 절반 자본: ${self.second_half_capital:.2f} (10번째 긴급 진입용)")
            if self.emergency_position_size is not None:
                print(f"긴급 진입용 포지션 크기: {self.emergency_position_size}")
        
        # 🆕 긴급 물타기 상태 정보 출력
        if self.emergency_entry_executed:
            emergency_qty = self.binance_calculator.get_emergency_position_size()
            print(f"\n=== 긴급 물타기 상태 ===")
            print(f"긴급 물타기 실행 여부: {self.emergency_entry_executed}")
            print(f"긴급 물타기 수량: {emergency_qty:.6f}")
        
        # 현재가 기준 손익 계산
        if self.binance_calculator.get_total_quantity() > 0:
            current_price = self.data.close[0]
            pnl = self.binance_calculator.calculate_pnl(current_price)
            print(f"현재가 ${current_price} 기준 손익: ${pnl['unrealized_pnl']:.2f} ({pnl['unrealized_pnl_percent']:.2f}%)")
        
        print("=" * 50)


# 데이터 수집 및 피드 생성, ML/VAR 계산 등은 별도 모듈로 분리해 작성할 수 있습니다.
# 예시: fetch_binance_data.py, feature_engineering.py, train_rf_model.py 등 