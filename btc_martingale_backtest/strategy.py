import backtrader as bt
import pandas as pd
import numpy as np
from squeeze_momentum_core import squeeze_momentum_core
from calc_var import calc_mean_var_from_df
import logging
from binance_calculator import calculate_liquidation_price, calculate_martingale_liquidation_price, validate_liquidation_risk
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




class MartingaleStrategy(bt.Strategy):
    """
    물타기 전략
    1. 초기 진입 수량 고정
    2. 물타기 진입 조건
    3. 부분 청산 조건
    4. 전체 청산 조건
    5. 마진콜 처리
    6. 거래 로그 저장   
    하락장은 기가 막히게 잘 피하는 전략이다. 다만 그러다보니 진입 기회가 적어서 수익률을 높이기 위한 변경이 필요하다.
    """
    
    params = dict(
        inputTrade=10,
        profit=1.01,  # 0.4%에서 1.5%로 상향 조정
        leverage=12,  # 포지션 크기 계산용 10배
        dividedLongCount=20,
        additionalEntryPrice=1200,
        max_var=0.2,  # 10%
        rf_threshold=0.3,  # RandomForest 확률 임계값
        mean_var=None,     # 전체 백테스팅 구간 평균 VaR
        max_var_dollar=1000,  # VaR 달러 기준 예시
        max_martingale_count=3,  # 최대 물타기 횟수 제한
        save_trade_log=True,  # 거래 로그 CSV 저장 여부
        trade_log_dir='trade_logs'  # 거래 로그 저장 디렉토리
    )

    def __init__(self):
        
        
        self.avgPrice = None
        self.entryCount = 0
        self.entryPriceSum = 0
        self.totalEntryCount = 0
        self.total_entry_count = 0
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
            'entry_count': self.entryCount,
            'avg_price': self.avgPrice if self.avgPrice else None,
            'position_size': self.tracked_position_size,
            'broker_value': self.broker.getvalue(),
            'profit_loss': None,
            'profit_ratio': None,
            'threshold': self.trade_threshold if self.trade_threshold else self.p.rf_threshold
        }
        
        # 수익률 계산 (청산 시에만)
        if action_type in ['final_exit', 'partial_exit'] and self.avgPrice:
            log_entry['profit_loss'] = self.data.close[0] - self.avgPrice
            log_entry['profit_ratio'] = (self.data.close[0] / self.avgPrice - 1) * 100
        
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
            
        # self.log(f'거래 시작')
        # 외부 피드: rf_pred(랜덤포레스트 예측값), var(몬테카를로 VaR)
        rf_pred = self.data.rf_pred[0]
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
        
        # 동적 자본 계산 (누적 투자 고려)
        initial_capital = self.broker.getvalue() * self.p.leverage
        capitalPerOnce = initial_capital / self.p.dividedLongCount
        positionSize = capitalPerOnce / close
        positionSize = np.round(positionSize, 3)


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
            close > open and
            abs(var) <= self.p.max_var
        )

        if entry_condition:
            # 익절 목표 분기
            if rf_pred >= self.p.rf_threshold:
                self.take_profit = 1.01  # 1% 익절
            else:
                self.take_profit = 1.004  # 0.4% 익절

            # 거래 ID 생성
            self.current_trade_id += 1
            self.trade_start_time = self.data.datetime.datetime(0)
            self.trade_start_price = close
            self.trade_rf_pred = rf_pred
            self.trade_threshold = self.p.rf_threshold

            # 진입 로직 (공통)
            if self.initial_capital is None:
                self.initial_capital = self.broker.getvalue()
                self.log(f"💰 초기 자본 설정: {self.initial_capital:.2f}")
            self.entryPriceSum = close
            self.total_entry_count = 1
            self.avgPrice = self.entryPriceSum / self.total_entry_count
            self.buy(size=positionSize)
            self.entryCount += 1
            self.var_history.append(var_dollar)
            self.tracked_position_size += positionSize
            self.tracked_position_value = self.tracked_position_size * close
            if self.initial_entry_size is None:
                self.initial_entry_size = positionSize
            self.log(f"[진입] 진입가: {close}, 평균가: {self.avgPrice}, entryCount: {self.entryCount}, 자본: {self.broker.getvalue():.2f}, positionSize: {positionSize}")
            self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
            
            # 거래 로그 저장
            self.save_trade_log('entry', 
                               entry_price=close, 
                               position_size=positionSize,
                               rf_pred=rf_pred,
                               threshold=self.p.rf_threshold)

            if self.entryCount >= 1 and self.entryCount < self.p.inputTrade:
                # self.log(f'물타기 조건')
                stoploss = self.p.additionalEntryPrice - (2 * atr)
                price_gap = self.avgPrice - close 
                # self.log(f'stoploss: {stoploss}')
                # self.log(f'물타기 조건 분석: avgPrice={self.avgPrice:.2f}, close={close:.2f}, avgPrice-close={self.avgPrice-close:.2f}, stoploss={stoploss:.2f}, entryCount={self.entryCount}, stoploss*entryCount={stoploss*self.entryCount:.2f}')
                # self.log(f'(self.avgPrice - close) > stoploss * self.entryCount : {(self.avgPrice - close) > stoploss * self.entryCount}')
                if price_gap > stoploss * self.entryCount  :
                    self.entryPriceSum = self.entryPriceSum + close
                    self.total_entry_count = self.total_entry_count + 1
                    self.avgPrice = self.entryPriceSum / self.total_entry_count
                    self.buy(size=positionSize)
                    self.entryCount += 1
                    self.var_history.append(var_dollar)  # VaR 히스토리에 저장
                    
                    # 포지션 크기 추적
                    self.tracked_position_size += positionSize
                    self.tracked_position_value = self.tracked_position_size * close
                    
                    # self.log(f"[물타기] 진입가: {close}, 평균가: {self.avgPrice}, entryCount: {self.entryCount}, var_dollar: {var_dollar}")
                    self.log(f"[물타기] 진입가: {close}, 평균가: {self.avgPrice}, entryCount: {self.entryCount}")
                    self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
                    
                    # 거래 로그 저장
                    self.save_trade_log('martingale', 
                                       entry_price=close, 
                                       position_size=positionSize,
                                       avg_price=self.avgPrice,
                                       entry_count=self.entryCount)
        # 부분 청산
        if self.entryCount >= 2 and (close > self.avgPrice * 1.003) :
            # 초기 진입 수량을 제외한 나머지 부분청산
            if self.initial_entry_size is not None:
                qty = self.tracked_position_size - self.initial_entry_size
            else:
                # fallback: 현재 포지션에서 새 진입 수량 제외
                qty = self.tracked_position_size - positionSize
            
            self.sell(size=qty)
            
            # 포지션 크기 추적 (부분 청산)
            self.tracked_position_size -= qty
            self.tracked_position_value = self.tracked_position_size * close
                
            self.log(f"[부분청산] 청산가: {close}, 평균가: {self.avgPrice}, entryCount: {self.entryCount}, 자본: {self.broker.getvalue():.2f}, 부분 청산 수량: {qty}")
            self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
            
            # 거래 로그 저장
            self.save_trade_log('partial_exit', 
                               exit_price=close, 
                               exit_qty=qty,
                               profit_ratio=(close / self.avgPrice - 1) * 100)
            
            self.entryCount = 1  # 부분청산 후 entryCount를 1로 설정
        
        # 전체 청산
        if self.entryCount == 1 and (close >= self.avgPrice * self.take_profit):
            self.close()
            
            # 포지션 크기 추적 (전체 청산)
            self.tracked_position_size = 0.0
            self.tracked_position_value = 0.0
            
            self.log(f"[최종청산] 청산가: {close}, 평균가: {self.avgPrice}, entryCount: {self.entryCount}, 자본: {self.broker.getvalue():.2f}")
            self.log(f"[포지션추적] 누적크기: {self.tracked_position_size:.6f}, 현재가치: {self.tracked_position_value:.2f}")
            
            # 거래 로그 저장
            self.save_trade_log('final_exit', 
                               exit_price=close, 
                               profit_ratio=(close / self.avgPrice - 1) * 100,
                               trade_duration=(self.data.datetime.datetime(0) - self.trade_start_time).total_seconds() / 3600 if self.trade_start_time else None)
            
            self.entryCount = 0
            self.avgPrice = 0
            self.entryPriceSum = 0
            self.total_entry_count = 0
            
            # 초기 진입 수량 초기화
            self.initial_entry_size = None
            self.take_profit = self.params.profit  # 초기화

       
            
           

        # self.log(f"[상태체크] 자본: {self.broker.getvalue():.2f}, 포지션: {self.position.size}, entryCount: {self.entryCount}")


        # 바이낸스 기준 마진콜 처리
        if self.entryCount > 0 and self.avgPrice is not None and not self.margin_called:
            # 추적된 포지션 크기 사용
            position_size_btc = self.tracked_position_size
            current_position_value = self.tracked_position_value
            
            # 동적 레버리지 계산
            actual_leverage = (self.p.leverage * self.entryCount) / self.p.dividedLongCount
            
            # 마진콜 가능성 판단
            margin_call_possible = actual_leverage > 1.0
            theoretical_liquidation_drop = 100 / actual_leverage if actual_leverage > 0 else float('inf')
            
            # 디버깅 로그
            # if self.tick_count % 1000 == 0:
            #     # self.log(f"🔍 디버깅 - 추적포지션크기(BTC): {position_size_btc:.6f}, 추적포지션가치(USD): {current_position_value:.2f}, 현재가: {close:.2f}")
            #     # self.log(f"🔍 비교 - Backtrader포지션크기: {self.position.size:.6f}, 차이: {abs(position_size_btc - self.position.size):.6f}")
            #     # self.log(f"🔍 레버리지 - 진입횟수: {self.entryCount}, 실제레버리지: {actual_leverage:.1f}배")
                
            #     # 마진콜 가능성 표시
            #     # if margin_call_possible:
            #     #     self.log(f"⚠️ 마진콜 위험 - 이론상청산하락률: {theoretical_liquidation_drop:.1f}%")
            #     # else:
            #     #     self.log(f"✅ 마진콜 안전 - 레버리지 {actual_leverage:.1f}배로 마진콜 불가능")
            
            # # 포지션가치가 0이면 계산 건너뛰기
            # if current_position_value <= 0:
            #     return
            
            # # 레버리지가 1배 이하이면 마진콜 불가능하므로 계산 건너뛰기
            # if not margin_call_possible:
            #     return
            
            # # 3번째 진입부터 마진콜 위험 경고
            # if self.entryCount >= 3 and self.tick_count % 500 == 0:
            #     self.log(f"🚨 마진콜 위험 구간 - 진입횟수: {self.entryCount}, 레버리지: {actual_leverage:.1f}배, 청산하락률: {theoretical_liquidation_drop:.1f}%")
            
            # # 바이낸스 청산가 계산 (동적 레버리지 사용)
            liquidation_price = calculate_martingale_liquidation_price(
                self.avgPrice, 
                current_position_value, 
                actual_leverage,  # ✅ 동적 레버리지 사용
                self.initial_capital if self.initial_capital is not None else 1000
            )
            
            # # 청산가가 유효하지 않으면 건너뛰기
            # if liquidation_price <= 0 or liquidation_price == float('inf'):
            #     return
            
            # 청산가 근처 경고 (청산가의 5% 이내)
            risk_info = validate_liquidation_risk(close, liquidation_price, warning_threshold=0.05)
            if risk_info['warning'] and self.tick_count % 100 == 0:
                self.log(f"⚠️ 청산가 근처 - 현재가: {close:.2f}, 청산가: {liquidation_price:.2f}, 거리: {risk_info['distance_percentage']:.1f}%, 위험도: {risk_info['risk_level']}")
                self.log(f"⚠️ 레버리지 정보 - 진입횟수: {self.entryCount}, 실제레버리지: {actual_leverage:.1f}배, 이론상청산하락률: {theoretical_liquidation_drop:.1f}%")
                
            #     # 현재 하락률 계산
            #     current_drop_percentage = ((self.avgPrice - close) / self.avgPrice) * 100
            #     self.log(f"⚠️ 현재 하락률: {current_drop_percentage:.1f}% (청산까지 {theoretical_liquidation_drop - current_drop_percentage:.1f}% 여유)")
            
            # # 바이낸스 기준 마진콜 조건
            if close <= liquidation_price:
                current_drop_percentage = ((self.avgPrice - close) / self.avgPrice) * 100
                
                self.log(f'🚨 바이낸스 기준 마진콜! 현재가: {close:.2f}, 청산가: {liquidation_price:.2f}')
                self.log(f'🚨 마진콜 상세 - 평단가: {self.avgPrice:.2f}, 포지션가치: {current_position_value:.2f}, 진입횟수: {self.entryCount}')
                self.log(f'🚨 레버리지 정보 - 실제레버리지: {actual_leverage:.1f}배, 이론상청산하락률: {theoretical_liquidation_drop:.1f}%')
                self.log(f'🚨 손실률: {current_drop_percentage:.2f}% (이론상 청산률: {theoretical_liquidation_drop:.1f}%)')
                
                # 거래 로그 저장
                self.save_trade_log('margin_call', 
                                   liquidation_price=liquidation_price,
                                   current_drop_percentage=current_drop_percentage,
                                   actual_leverage=actual_leverage)
                
            #     # 진입 횟수별 분석
            #     if self.entryCount <= 2:
            #         self.log(f'🚨 예상치 못한 마진콜! 진입횟수 {self.entryCount}회는 마진콜이 발생하면 안됨')
            #     elif self.entryCount <= 5:
            #         self.log(f'🚨 중간 위험 구간 마진콜 - 진입횟수 {self.entryCount}회 (레버리지 {actual_leverage:.1f}배)')
            #     else:
            #         self.log(f'🚨 고위험 구간 마진콜 - 진입횟수 {self.entryCount}회 (레버리지 {actual_leverage:.1f}배)')
                
            #     self.margin_called = True
            #     if self.position.size != 0:
            #         self.close()
            #         self.log(f'🚨 마진콜로 인한 강제 청산 완료')
                
                # 포지션 크기 초기화
                self.tracked_position_size = 0.0
                self.tracked_position_value = 0.0
                
                self.entryCount = 0
                self.avgPrice = 0
                self.entryPriceSum = 0
                self.total_entry_count = 0
                
                # 초기 진입 수량 초기화
                self.initial_entry_size = None
                return

    def stop(self):
        """
        백테스팅 종료 시 거래 로그 저장
        """
        if self.p.save_trade_log:
            log_filepath = self.save_trade_logs_to_csv()
            if log_filepath:
                print(f"거래 로그가 저장되었습니다: {log_filepath}")


# 데이터 수집 및 피드 생성, ML/VAR 계산 등은 별도 모듈로 분리해 작성할 수 있습니다.
# 예시: fetch_binance_data.py, feature_engineering.py, train_rf_model.py 등 