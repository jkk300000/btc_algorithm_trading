import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PineScriptConvertedStrategy(bt.Strategy):
    """
    Pine Script 코드를 Python으로 변환한 전략
    - 스퀴즈 모멘텀 지표 기반 진입
    - 바이낸스 평균가 계산
    - 동적 레버리지 기반 청산가 계산
    - 부분청산 및 전체청산 로직
    """
    
    params = (
        ('initial_capital', 1000),
        ('leverage', 6),
        ('input_trade', 10),  # 거래 투입 횟수
        ('profit', 1.013),    # 익절%
        ('divided_long_count', 20),  # 시드 분할
        ('additional_entry_price', 1500),  # 물타기 한도
        ('bb_length', 20),    # BB Length
        ('bb_mult', 2.0),     # BB MultFactor
        ('kc_length', 20),    # KC Length
        ('kc_mult', 1),       # KC MultFactor
        ('use_true_range', True),  # Use TrueRange (KC)
        ('atr_period', 14),   # ATR 기간
        ('partial_profit', 1.0035),  # 부분청산 익절%
        ('start_time', '2022-09-01 00:00:00'),  # 자동매매 시작
        ('end_time', '2024-12-31 23:59:59'),    # 자동매매 종료
    )
    
    def __init__(self):
        # 시간 설정
        self.start_time = pd.to_datetime(self.params.start_time)
        self.end_time = pd.to_datetime(self.params.end_time)
        
        # 바이낸스 평균가 계산 변수들
        self.total_cost = 0.0
        self.total_quantity = 0.0
        self.avg_price = 0.0
        
        # 청산가 계산 변수들
        self.liquidation_price = None
        self.total_position_value = 0.0
        self.cycle_start_capital = None
        
        # 거래 관련 변수들
        self.entry_count = 0
        self.initial_position_size = None
        self.initial_entry_price = None
        self.second_entry_price = None
        self.position_size = None
        self.capital_per_once = None
        self.stoploss = None
        
        # 지표 계산용 변수들
        self.bb_basis = None
        self.bb_upper = None
        self.bb_lower = None
        self.kc_upper = None
        self.kc_lower = None
        self.val = None
        self.atr = None
        self.vwap = None
        
        # 거래 로그
        self.trade_logs = []
        
        # 마진콜 관련
        self.margin_called = False
        
    def start(self):
        """전략 시작 시 초기화"""
        self.cycle_start_capital = self.broker.getvalue()
        self.capital_per_once = (self.broker.getvalue() * self.params.leverage) / self.params.divided_long_count
        
    def next(self):
        """각 바에서 실행되는 메인 로직"""
        current_time = self.data.datetime.datetime(0)
        
        # 거래 기간 확인
        if not (self.start_time <= current_time <= self.end_time):
            return
            
        # 지표 계산
        self._calculate_indicators()
        
        # 포지션 크기 계산
        self.position_size = round(self.capital_per_once / self.data.close[0] * 1000) / 1000
        self.stoploss = self.params.additional_entry_price - (2.5 * self.atr)
        
        # entryCount가 1, 2일 때는 청산가를 강제로 None으로 설정
        if self.entry_count <= 2:
            self.liquidation_price = None
            
        # 진입 로직
        if self.broker.getposition().size < self.params.input_trade and self.position_size > 0:
            self._handle_entries()
            
        # 청산가 모니터링
        if (self.broker.getposition().size > 0 and 
            self.entry_count >= 3 and 
            self.liquidation_price is not None):
            self._monitor_liquidation()
            
        # 부분청산 로직
        if (self.entry_count >= 2 and 
            self.data.close[0] > self.avg_price * self.params.partial_profit and 
            self.avg_price != 0):
            self._handle_partial_exit()
            
        # 전체 청산
        if (self.entry_count == 1 and 
            self.data.close[0] >= self.avg_price * self.params.profit):
            self._handle_full_exit()
    
    def _calculate_indicators(self):
        """스퀴즈 모멘텀 지표 계산"""
        # BB 계산
        if len(self.data) >= self.params.bb_length:
            self.bb_basis = np.mean(self.data.close.get(size=self.params.bb_length))
            bb_std = np.std(self.data.close.get(size=self.params.bb_length))
            self.bb_upper = self.bb_basis + self.params.bb_mult * bb_std
            self.bb_lower = self.bb_basis - self.params.bb_mult * bb_std
            
        # KC 계산
        if len(self.data) >= self.params.kc_length:
            kc_ma = np.mean(self.data.close.get(size=self.params.kc_length))
            
            if self.params.use_true_range:
                # True Range 계산
                high = self.data.high.get(size=self.params.kc_length)
                low = self.data.low.get(size=self.params.kc_length)
                close_prev = self.data.close.get(size=self.params.kc_length+1)[1:]
                
                tr_values = []
                for i in range(len(high)):
                    tr1 = high[i] - low[i]
                    tr2 = abs(high[i] - close_prev[i])
                    tr3 = abs(low[i] - close_prev[i])
                    tr_values.append(max(tr1, tr2, tr3))
                
                range_ma = np.mean(tr_values)
            else:
                high = self.data.high.get(size=self.params.kc_length)
                low = self.data.low.get(size=self.params.kc_length)
                range_ma = np.mean(high - low)
                
            self.kc_upper = kc_ma + range_ma * self.params.kc_mult
            self.kc_lower = kc_ma - range_ma * self.params.kc_mult
            
        # val 계산 (스퀴즈 모멘텀)
        if (len(self.data) >= self.params.kc_length and 
            self.bb_upper is not None and self.bb_lower is not None and
            self.kc_upper is not None and self.kc_lower is not None):
            
            source = self.data.close[0]
            highest_high = np.max(self.data.high.get(size=self.params.kc_length))
            lowest_low = np.min(self.data.low.get(size=self.params.kc_length))
            sma_close = np.mean(self.data.close.get(size=self.params.kc_length))
            
            # Linear regression 계산 (간단한 버전)
            x = np.arange(self.params.kc_length)
            y = self.data.close.get(size=self.params.kc_length) - np.mean([highest_high, lowest_low, sma_close])
            
            if len(x) == len(y):
                slope = np.polyfit(x, y, 1)[0]
                self.val = slope
            else:
                self.val = 0
        else:
            self.val = 0
            
        # ATR 계산
        if len(self.data) >= self.params.atr_period:
            high = self.data.high.get(size=self.params.atr_period)
            low = self.data.low.get(size=self.params.atr_period)
            close_prev = self.data.close.get(size=self.params.atr_period+1)[1:]
            
            tr_values = []
            for i in range(len(high)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close_prev[i])
                tr3 = abs(low[i] - close_prev[i])
                tr_values.append(max(tr1, tr2, tr3))
            
            self.atr = np.mean(tr_values)
        else:
            self.atr = 0
            
        # VWAP 계산 (간단한 버전)
        if len(self.data) >= 20:
            typical_price = (self.data.high.get(size=20) + 
                           self.data.low.get(size=20) + 
                           self.data.close.get(size=20)) / 3
            volume = self.data.volume.get(size=20)
            self.vwap = np.sum(typical_price * volume) / np.sum(volume)
        else:
            self.vwap = self.data.close[0]
    
    def _handle_entries(self):
        """진입 로직 처리"""
        # 첫 번째 진입
        if (self.val > 0 and self.entry_count == 0 and 
            self.data.close[0] > self.data.open[0]):
            
            self.initial_entry_price = self.data.close[0]
            self.initial_position_size = self.position_size
            
            # 사이클 시작 시 자본 고정
            self.cycle_start_capital = self.broker.getvalue()
            
            # 바이낸스 평균가 계산
            self.total_cost = self.data.close[0] * self.position_size
            self.total_quantity = self.position_size
            self.avg_price = self.total_cost / self.total_quantity
            
            # 청산가 계산 (첫 진입 시에는 청산가 없음)
            self.total_position_value = self.total_quantity * self.avg_price
            self.liquidation_price = None
            
            # 주문 실행
            self.buy(size=self.position_size)
            
            self.entry_count = 1
            
            # 로그 기록
            self._log_trade("Entry 1", self.data.close[0], self.avg_price, 
                          self.initial_position_size, None, 0)
            
        # 추가 진입 (2-10번째)
        elif (self.entry_count >= 1 and self.entry_count < self.params.input_trade):
            if (self.avg_price - self.data.close[0]) > self.stoploss * self.entry_count:
                
                self.second_entry_price = self.data.close[0]
                
                # 바이낸스 평균가 계산 (증분 방식)
                new_cost = self.data.close[0] * self.position_size
                self.total_cost += new_cost
                self.total_quantity += self.position_size
                self.avg_price = self.total_cost / self.total_quantity
                
                # 청산가 재계산
                self.total_position_value = self.total_quantity * self.avg_price
                self.entry_count += 1
                
                # 청산가 계산
                self.liquidation_price = self._calculate_liquidation_price()
                
                # 주문 실행
                self.buy(size=self.position_size)
                
                # 로그 기록
                actual_leverage = self._calculate_actual_leverage()
                self._log_trade(f"Entry {self.entry_count}", self.data.close[0], 
                              self.avg_price, self.liquidation_price, actual_leverage)
    
    def _monitor_liquidation(self):
        """청산가 모니터링"""
        if self.liquidation_price is None:
            return
            
        actual_leverage = self._calculate_actual_leverage()
        distance_to_liquidation = self.data.close[0] - self.liquidation_price
        distance_percentage = (distance_to_liquidation / self.liquidation_price) * 100
        
        theoretical_liquidation_drop = 100 / actual_leverage if actual_leverage > 0 else 0
        current_drop_percentage = ((self.avg_price - self.data.close[0]) / self.avg_price) * 100
        
        # 청산 위험 경고
        if self.data.close[0] <= self.liquidation_price * 1.05:
            logger.warning(f"🚨 청산 위험: 현재가={self.data.close[0]:.2f}, "
                         f"청산가={self.liquidation_price:.2f}, 거리={distance_percentage:.2f}%")
            logger.warning(f"🚨 레버리지 정보: 진입횟수={self.entry_count}, "
                         f"실제레버리지={actual_leverage:.2f}배, "
                         f"이론상청산하락률={theoretical_liquidation_drop:.2f}%")
            logger.warning(f"🚨 현재 하락률: {current_drop_percentage:.2f}% "
                         f"(청산까지 {theoretical_liquidation_drop - current_drop_percentage:.2f}% 여유)")
        
        # 실제 청산 체크
        if self.data.close[0] <= self.liquidation_price * 1.01:
            self.margin_called = True
            self.close()
            
            logger.error(f"🚨 바이낸스 청산 발생: 현재가={self.data.close[0]:.2f}, "
                        f"청산가={self.liquidation_price:.2f}")
            logger.error(f"🚨 마진콜 상세: 평단가={self.avg_price:.2f}, "
                        f"진입횟수={self.entry_count}, 실제레버리지={actual_leverage:.2f}배")
            logger.error(f"🚨 손실률: {current_drop_percentage:.2f}% "
                        f"(이론상 청산률: {theoretical_liquidation_drop:.2f}%)")
            
            # 마진콜 로그 기록
            self._log_trade("Margin Call", self.data.close[0], self.avg_price, 
                          self.liquidation_price, actual_leverage, 
                          action_type="margin_call", 
                          current_drop_percentage=current_drop_percentage)
    
    def _handle_partial_exit(self):
        """부분청산 처리"""
        qty = self.broker.getposition().size - self.initial_position_size
        
        if qty > 0:
            # 바이낸스 방식: 매도 시 평균가 유지, 수량만 감소
            removed_cost = self.avg_price * qty
            self.total_cost -= removed_cost
            self.total_quantity -= qty
            
            # 주문 실행
            self.sell(size=qty)
            
            # 상태 업데이트
            self.entry_count = 1
            self.liquidation_price = None
            self.cycle_start_capital = None
            
            # 로그 기록
            self._log_trade("Partial Exit", self.data.close[0], self.avg_price, 
                          qty, self.broker.getposition().size, action_type="partial_exit")
    
    def _handle_full_exit(self):
        """전체 청산 처리"""
        self.close()
        
        # 로그 기록
        self._log_trade("Full Exit", self.data.close[0], self.avg_price, 
                      action_type="full_exit")
        
        # 변수 초기화
        self._reset_variables()
    
    def _calculate_actual_leverage(self):
        """실제 레버리지 계산"""
        if (self.entry_count <= 0 or self.params.leverage <= 0 or 
            self.params.divided_long_count <= 0):
            return 0.0
        
        return self.total_position_value / self.cycle_start_capital if self.cycle_start_capital else 0.0
    
    def _calculate_liquidation_price(self):
        """청산가 계산"""
        if self.entry_count <= 2:
            return None
            
        actual_leverage = self._calculate_actual_leverage()
        
        if actual_leverage < 1.0:
            # 이론상 청산가
            return self.avg_price * (1 - 1/actual_leverage)
        else:
            # 바이낸스 청산가 공식
            maintenance_margin = 0.005
            return self.avg_price - ((self.avg_price / actual_leverage) * (1 + maintenance_margin))
    
    def _reset_variables(self):
        """변수 초기화"""
        self.avg_price = 0
        self.total_cost = 0
        self.total_quantity = 0
        self.initial_entry_price = 0
        self.second_entry_price = 0
        self.liquidation_price = None
        self.total_position_value = 0.0
        self.cycle_start_capital = None
        self.entry_count = 0
    
    def _log_trade(self, action, price, avg_price, *args, **kwargs):
        """거래 로그 기록"""
        log_entry = {
            'timestamp': self.data.datetime.datetime(0),
            'action': action,
            'price': price,
            'avg_price': avg_price,
            'entry_count': self.entry_count,
            'position_size': self.broker.getposition().size,
            'equity': self.broker.getvalue(),
            **kwargs
        }
        
        # 추가 인자들 처리
        if len(args) >= 1:
            log_entry['arg1'] = args[0]
        if len(args) >= 2:
            log_entry['arg2'] = args[1]
        if len(args) >= 3:
            log_entry['arg3'] = args[2]
        if len(args) >= 4:
            log_entry['arg4'] = args[3]
            
        self.trade_logs.append(log_entry)
        
        # 콘솔 로그
        if action.startswith("Entry"):
            logger.info(f"{action} at {price:.2f}, avgPrice: {avg_price:.2f}, "
                       f"initialPositionSize: {args[0] if len(args) > 0 else 'N/A'}, "
                       f"청산가: {args[1] if len(args) > 1 else '없음'}, "
                       f"실제레버리지: {args[2] if len(args) > 2 else 'N/A'}배")
        elif action == "Partial Exit":
            logger.info(f"초기 투입 물량 빼고 청산 at {price:.2f}, avgPrice: {avg_price:.2f}, "
                       f"qty: {args[0] if len(args) > 0 else 'N/A'}, "
                       f"strategy.position_size: {args[1] if len(args) > 1 else 'N/A'}, "
                       f"청산가: 없음 (Partial Exit 후), entryCount: {self.entry_count}")
        elif action == "Full Exit":
            logger.info(f"exit all at {price:.2f}, avgPrice: {avg_price:.2f}")
    
    def stop(self):
        """전략 종료 시"""
        if self.margin_called:
            logger.error("🚨 마진콜로 인해 전략이 종료되었습니다.")
        else:
            logger.info("✅ 전략이 정상적으로 종료되었습니다.")
            
        # 최종 통계
        logger.info(f"📊 최종 통계:")
        logger.info(f"   - 총 거래 횟수: {len(self.trade_logs)}")
        logger.info(f"   - 최종 자본: {self.broker.getvalue():.2f}")
        logger.info(f"   - 최종 진입 횟수: {self.entry_count}")
        logger.info(f"   - 최종 평균가: {self.avg_price:.2f}")
