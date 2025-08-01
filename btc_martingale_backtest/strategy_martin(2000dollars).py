"""
바이낸스 평균가격 계산기를 사용하는 수정된 물타기 전략

기존 코드의 문제점:
- self.avgPrice = self.entryPriceSum / self.total_entry_count
- 수량을 고려하지 않고 단순히 가격만 평균을 계산

수정된 코드:
- BinanceAveragePriceCalculator 사용
- 가격과 수량을 모두 고려한 가중평균 계산
"""

import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime
import os
from binance_calculator import BinanceAveragePriceCalculator

class MartingaleStrategyFixed2000dollars(bt.Strategy):
    """
    바이낸스 평균가격 계산기를 사용하는 물타기 전략
    
    기존 코드의 문제점:
    - self.avgPrice = self.entryPriceSum / self.total_entry_count
    - 수량을 고려하지 않고 단순히 가격만 평균을 계산
    
    수정된 코드:
    - BinanceAveragePriceCalculator 사용
    - 가격과 수량을 모두 고려한 가중평균 계산
    """

    params = dict(
        inputTrade=10,
        profit=1.01,
        profit_partial=1.005,  # 0.4%에서 1.5%로 상향 조정
        leverage=10,  # 포지션 크기 계산용 10배 (0에서 10으로 수정)
        dividedLongCount=20,
        additionalEntryPrice=1500,
        max_var=0.05,  # 12% (균형잡힌 설정)
        rf_threshold=0.5, # RandomForest 확률 임계값
        rf_threshold_partial=0.6, # RandomForest 확률 임계값
        # rf_threshold_down=0.9, # 하락 예측 임계값 (70% 이상이면 거래 회피)
        # rf_threshold_down_martingale=0.9, # 물타기 시 하락 예측 임계값 (80% 이상이면 물타기 회피)
        mean_var=None,     # 전체 백테스팅 구간 평균 VaR
        max_var_dollar=1000,  # VaR 달러 기준 예시
        
        save_trade_log=True,  # 거래 로그 CSV 저장 여부
        trade_log_dir='trade_logs'  # 거래 로그 저장 디렉토리
    )

    def __init__(self):
        # 전략 상태 변수
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
            'take_profit': self.take_profit,
            'threshold_partial': self.p.rf_threshold_partial,
            # 'threshold_down': self.p.rf_threshold_down,
            # 'threshold_down_martingale': self.p.rf_threshold_down_martingale
        }
        
        # 수익률은 broker.getvalue()로 충분히 추적 가능하므로 제거
        
        # 추가 키워드 인자들 추가
        for key, value in kwargs.items():
            log_entry[key] = value
        
        self.trade_logs.append(log_entry)

    def save_trade_logs_to_csv(self):
        """거래 로그를 CSV 파일로 저장"""
        if not self.trade_logs:
            return
        
        df = pd.DataFrame(self.trade_logs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.p.trade_log_dir, f"trade_logs_binance_fixed_{timestamp}.csv")
        df.to_csv(filename, index=False)
        print(f"거래 로그가 {filename}에 저장되었습니다.")

    def next(self):
        # 마진콜 상태면 모든 거래 중단
        if self.margin_called:
            return
            
        # 외부 피드: rf_pred(랜덤포레스트 예측값), rf_pred_down(하락 예측값), var(몬테카를로 VaR)
        rf_pred = self.data.rf_pred[0]
        rf_pred_down = self.data.rf_pred_down[0] if hasattr(self.data, 'rf_pred_down') else 0.5  # 5% 하락 후 10% 하락할 확률
        var = self.data.var[0] if hasattr(self.data, 'var') else 0.05
        var_dollar = self.data.var_dollar[0] if hasattr(self.data, 'var_dollar') else 100
        atr = self.data.atr_14[0] if hasattr(self.data, 'atr_14') else 1000
        val = self.data.val[0] if hasattr(self.data, 'val') else 1
        close = self.data.close[0]
        open_price = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        
        highest_price = self.data.high[-1] if len(self.data) > 1 else high
        
        # 🆕 마진콜 체크 로직 추가
        if self.tracked_position_size > 0:
            avg_price = self.binance_calculator.get_average_price()
            liquidation_price = avg_price * (1 - 100 / self.p.leverage / 100)  # 10배 레버리지 기준 청산가
            
            # 청산 위험도 체크 (현재가가 청산가의 5% 이내로 접근)
            risk_distance = (close - liquidation_price) / liquidation_price
            if risk_distance <= 0.05:  # 5% 이내
                self.log(f"⚠️ 청산 위험 감지! 현재가: {close}, 청산가: {liquidation_price:.2f}, 거리: {risk_distance*100:.2f}%")
                if risk_distance <= 0.01:  # 1% 이내면 강제 청산
                    self.log(f"🚨 강제 청산 실행! 청산 위험도가 너무 높습니다.")
                    self.close()
                    self.binance_calculator.close_all_positions()
                    self.entryCount = 0
                    self.tracked_position_size = 0.0
                    self.tracked_position_value = 0.0
                    self.initial_entry_size = None
                    self.margin_called = True
                    self.save_trade_log('margin_call', 
                                       current_price=close,
                                       liquidation_price=liquidation_price,
                                       risk_distance=risk_distance)
                    return
        
        # 동적 자본 계산 (누적 투자 고려)
        initial_capital = self.broker.getvalue() * self.p.leverage
        capitalPerOnce = initial_capital / self.p.dividedLongCount
        positionSize = capitalPerOnce / close
        positionSize = np.round(positionSize, 3)
        
        # 틱 카운트 증가
        self.tick_count += 1
        self.montecarlo_var_dollar = var_dollar
        self.montecarlo_var_percent = var
        
        can_enter = self.entryCount < self.params.inputTrade and positionSize > 0
        entry_condition = (
            can_enter and
            val > 0 and
            self.entryCount == 0 and
            self.totalEntryCount == 0 and
            close > open_price and
            abs(var) <= self.p.max_var 
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

            # 진입 로직
            if self.initial_capital is None:
                self.initial_capital = self.broker.getvalue()
                self.log(f"💰 초기 자본 설정: {self.initial_capital:.2f}")
            
            # 🆕 바이낸스 평균가격 계산기 사용
            self.binance_calculator.add_position(close, positionSize)
            
            self.buy(size=positionSize)
            self.entryCount += 1
            self.var_history.append(var_dollar)
            self.tracked_position_size += positionSize
            self.tracked_position_value = self.tracked_position_size * close
            if self.initial_entry_size is None:
                self.initial_entry_size = positionSize
            
            # 로그 출력 (바이낸스 평균가격만 사용)
            binance_avg = self.binance_calculator.get_average_price()
            self.log(f"[진입] 진입가: {close}, 바이낸스평균가: {binance_avg:.2f}, entryCount: {self.entryCount}, 자본: {self.broker.getvalue():.2f}, positionSize: {positionSize}")
            self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
            
            # 거래 로그 저장
            self.save_trade_log('entry', 
                               entry_price=close, 
                               position_size=positionSize,
                               rf_pred=rf_pred,
                               threshold=self.p.rf_threshold,
                               take_profit=self.take_profit)

        if self.entryCount >= 1 and self.entryCount <= self.p.inputTrade:
            # self.log(f'물타기 조건')
            stoploss = self.p.additionalEntryPrice - (2 * atr)
            price_gap = self.binance_calculator.get_average_price() - close  # 🆕 바이낸스 평균가격 사용
           
            # 일반 물타기 조건 (1~9번째 진입)
            if (self.entryCount < self.p.inputTrade and 
                price_gap > stoploss * self.entryCount and 
                rf_pred >= self.p.rf_threshold_partial and 
                abs(var) <= self.p.max_var):
                
                # 🆕 바이낸스 평균가격 계산기 사용
                self.binance_calculator.add_position(close, positionSize)
                
                self.buy(size=positionSize)
                self.entryCount += 1
                self.var_history.append(var_dollar)
                
                # 포지션 크기 추적
                self.tracked_position_size += positionSize
                self.tracked_position_value = self.tracked_position_size * close
                
                # 로그 출력 (바이낸스 평균가격만 사용)
                binance_avg = self.binance_calculator.get_average_price()
                self.log(f"[물타기] 진입가: {close}, 바이낸스평균가: {binance_avg:.2f}, entryCount: {self.entryCount}")
                self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
                
                # 거래 로그 저장
                self.save_trade_log('martingale', 
                                    entry_price=close, 
                                    position_size=positionSize,
                                    avg_price_binance=binance_avg,
                                    entry_count=self.entryCount,
                                    rf_pred_partial=self.p.rf_threshold_partial)

            # 🆕 10번째 진입 조건 (최대 물타기)
            if (self.entryCount == self.p.inputTrade and 
                  price_gap > stoploss * self.entryCount and 
                  rf_pred >= self.p.rf_threshold_partial 
                  ):
                
                # 10번째 진입: 현재 포지션 크기의 100%만큼 추가 매수 (최대 물타기)
                max_martingale_size = self.tracked_position_size
                
                # 🆕 바이낸스 평균가격 계산기 사용
                self.binance_calculator.add_position(close, max_martingale_size)
                
                self.buy(size=max_martingale_size)
                self.entryCount += 1
                self.var_history.append(var_dollar)
                
                # 포지션 크기 추적
                self.tracked_position_size += max_martingale_size
                self.tracked_position_value = self.tracked_position_size * close
                
                # 로그 출력 (바이낸스 평균가격만 사용)
                binance_avg = self.binance_calculator.get_average_price()
                self.log(f"[최대물타기] 진입가: {close}, 바이낸스평균가: {binance_avg:.2f}, entryCount: {self.entryCount}")
                self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
                self.log(f"[최대물타기] 10번째 진입! 최대 물타기 실행")

                # 거래 로그 저장
                self.save_trade_log('max_martingale', 
                                    entry_price=close, 
                                    position_size=max_martingale_size,
                                    avg_price_binance=binance_avg,
                                    entry_count=self.entryCount,
                                    rf_pred_partial=self.p.rf_threshold_partial)

            


        # 부분 청산
        if self.entryCount >= 2 and (close > self.binance_calculator.get_average_price() * 1.003):  # 🆕 바이낸스 평균가격 사용
            # 초기 진입 수량을 제외한 나머지 부분청산
            if self.initial_entry_size is not None:
                qty = self.tracked_position_size - self.initial_entry_size
            else:
                qty = self.tracked_position_size - positionSize
            
            # 부분 청산 실행
            if qty > 0:
                # 🆕 바이낸스 평균가격 계산기에서 포지션 제거
                self.binance_calculator.remove_position(qty)
                
                self.sell(size=qty)
                
                # 포지션 크기 추적 (부분 청산)
                self.tracked_position_size -= qty
                self.tracked_position_value = self.tracked_position_size * close
                    
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
                                   profit_ratio=profit_ratio)
                
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
            
            # 포지션 크기 추적 (전체 청산)
            self.tracked_position_size = 0.0
            self.tracked_position_value = 0.0
            
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
            
            self.log(f"[최종청산] 청산가: {close}, 바이낸스평균가: {final_avg_price:.2f}, entryCount: {self.entryCount}, 자본: {self.broker.getvalue():.2f}")
            self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")

    def stop(self):
        """전략 종료 시 실행"""
        self.save_trade_logs_to_csv()
        
        # 최종 통계 출력
        print(f"\n=== 바이낸스 평균가격 계산기 최종 통계 ===")
        print(f"총 거래 횟수: {len(self.trade_logs)}")
        print(f"최종 평균가격: ${self.binance_calculator.get_average_price():.2f}")
        print(f"최종 포지션 수량: {self.binance_calculator.get_total_quantity():.6f}")
        print(f"최종 포지션 가치: ${self.binance_calculator.get_total_value():.2f}")
        
        # 현재가 기준 손익 계산
        if self.binance_calculator.get_total_quantity() > 0:
            current_price = self.data.close[0]
            pnl = self.binance_calculator.calculate_pnl(current_price)
            print(f"현재가 ${current_price} 기준 손익: ${pnl['unrealized_pnl']:.2f} ({pnl['unrealized_pnl_percent']:.2f}%)")
        
        print("=" * 50) 