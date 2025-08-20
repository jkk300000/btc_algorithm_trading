import backtrader as bt
import pandas as pd
from strategy.strategy_martin_fixed import ModifiedMartingaleStrategy
from strategy.strategy_martin import MartingaleStrategy
from strategy.strategy_new import NewModifiedMartingaleStrategy
from strategy.strategy_martin_fixed_pine import StrategyMartinFixedPine

from ml_model.train_rf_model import train_and_predict
from ml_model.train_rf_model_down import train_and_predict_10pct_after_5pct
from indicator.calc_var import calc_var, calc_mean_var_from_df 
import logging
import numpy as np
import os
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArithmeticReturns(bt.Analyzer): # ✅ 해당 클래스는 전략(bt.strategy를 파라미터로 받는 클래스)이 실행되기 전에 전략 값을 가져오므로 
                                      # 전략 실행 후의 값을 가져오고 싶으면 stop() 함수에서 값을 불러올 것
        def __init__(self):
            self.initial_cash = None
            self.net_profit_percent = 0.0
            self.net_profit = 0
            self.final_value = None

        def start(self):
            # 전략 시작 시 초기 자본 설정
            self.initial_cash = self.strategy.broker.getcash()

        def notify_cashvalue(self, cash, value):
            self.final_value = value

        def stop(self):
            # final_value가 설정되지 않았으면 현재 broker value를 사용
            if self.final_value is None: 
                self.final_value = self.strategy.broker.getvalue()
            
            # initial_cash가 설정되지 않았으면 현재 broker cash를 사용
            if self.initial_cash is None:
                self.initial_cash = self.strategy.broker.getcash()
            
            self.net_profit_percent = ((self.final_value - self.initial_cash) / self.initial_cash) * 100
            self.net_profit = self.final_value - self.initial_cash

        def get_analysis(self):
            return {'net_profit_percent': self.net_profit_percent,  'net_profit': self.net_profit}


class CommissionAnalyzer(bt.Analyzer):
    def __init__(self):
        self.total_commission = 0.0
        self.total_fees = 0.0
        self.commission_by_trade = []
        
    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_commission += trade.commission
            self.total_fees += trade.commission
            self.commission_by_trade.append(trade.commission)
    
    def get_analysis(self):
        return {
            'total_commission': self.total_commission,
            'total_fees': self.total_fees,
            'avg_commission_per_trade': np.mean(self.commission_by_trade) if self.commission_by_trade else 0.0,
            'commission_by_trade': self.commission_by_trade
        }


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


class ArithmeticReturnAnalyzer(bt.Analyzer):
    def __init__(self):
        self.start_value = None
        self.end_value = None
        self.returns = []

    def start(self):
        self.start_value = self.strategy.broker.getvalue()

    def stop(self):
        self.end_value = self.strategy.broker.getvalue()
        self.returns = (self.end_value - self.start_value) / self.start_value

    def get_analysis(self):
        return {'arithmetic_return': self.returns}

class CustomDrawDownAnalyzer(bt.Analyzer):
    def __init__(self):
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.peak = None

    def next(self):
        value = self.strategy.broker.getvalue()
        if self.peak is None or value > self.peak:
            self.peak = value
        drawdown = self.peak - value
        drawdown_pct = drawdown / self.peak * 100 if self.peak else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        if drawdown_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown_pct

    def get_analysis(self):
        return {
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct
        }

def run_backtest(df, start_date='2022-09-01', cash=1000, commission=0.0005, leverage=8):
    print("="*50)
    print(f"Backtest Start: {start_date}")
    print(f"Initial Cash: {cash}, Commission: {commission}, Leverage: {leverage}")
    print(f"Data rows: {len(df)}")
    print("="*50)
    train_end = pd.Timestamp('2022-08-31 23:59:00+00:00')
    start_ts = pd.to_datetime(start_date)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(df.index.tz)
        else:
            start_ts = start_ts.tz_convert(df.index.tz)
    if start_ts in df.index:
        start_idx = df.index.get_loc(start_ts)
    else:
        start_idx = df.index.searchsorted(start_ts, side='left')
    mean_var = calc_mean_var_from_df(df, start_date=start_date)
    df = df[df.index >= start_ts]
    df = df.dropna(subset=['rf_pred'])
    
    print(f"Mean VaR (start_date={start_date}): {mean_var:.6f}")
    
    # 데이터 확장
    class PandasDataExt(bt.feeds.PandasData):
        lines = ('rf_pred', 'rf_pred_down', 'var', 'var_dollar', 'val', 'atr_14')
        params = (('rf_pred', -1), ('rf_pred_down', -1), ('var', -1), ('var_dollar', -1), ('val', -1), ('atr_14', -1))
    

    # 먼저 기본 데이터 생성
    data = PandasDataExt(dataname=df)
    
    # 타임프레임 설정 (예: 5분봉)
    # data = bt.feeds.PandasData.resample(data, timeframe=bt.TimeFrame.Minutes, compression=5)
    
    cerebro = bt.Cerebro()
    
    # 🚀 멀티 전략 실행 (3개 전략 비교)
    # cerebro.addstrategy(ModifiedMartingaleStrategy, mean_var=mean_var, leverage=leverage)
    # cerebro.addstrategy(NewModifiedMartingaleStrategy, mean_var=mean_var, leverage=leverage)
    cerebro.addstrategy(MartingaleStrategy, mean_var=mean_var, leverage=leverage)
    # cerebro.addstrategy(AdaptiveMartingaleStrategy, mean_var=mean_var, leverage=leverage)
    # cerebro.addstrategy(StrategyMartinFixedPine, mean_var=mean_var, leverage=leverage)
    cerebro.adddata(data) 
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission, leverage=leverage, margin=0)
    cerebro.broker.set_slippage_perc(0.00015)
     
    
    cerebro.addanalyzer(ArithmeticReturns, _name='arithmetic_returns')
    cerebro.addanalyzer(MonteCarloVaRAnalyzer, _name='mc_var')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(ArithmeticReturnAnalyzer, _name='arithret')
    cerebro.addanalyzer(CustomDrawDownAnalyzer, _name='customdd')
    # cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')  # ZeroDivisionError 방지를 위해 제거
    cerebro.addanalyzer(CommissionAnalyzer, _name='commission')
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]

    mc_result = strat.analyzers.mc_var.get_analysis()
    avg_var = mc_result.get('avg_var', None)
    last_var = mc_result.get('var_value', None)
    var_per = mc_result.get('var_percent', None)
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0) or 0.0
    drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
    drawdown_money = strat.analyzers.drawdown.get_analysis().get('max', {}).get('moneydown')
    # total_return = strat.analyzers.returns.get_analysis().get('rtot', 0.0) * 100  # returns 분석기 제거로 인해 주석 처리
    total_return = 0.0  # returns 분석기 제거로 인해 0으로 설정
    arithmetic_profit_percent = strat.analyzers.arithmetic_returns.get_analysis().get('net_profit_percent', 0.0)
    arithmetic_profit = strat.analyzers.arithmetic_returns.get_analysis().get('net_profit', 0.0)
    trades = strat.analyzers.trades.get_analysis().get('total', {}).get('total', 0)
    
    # 수수료 정보 추출 (TradeAnalyzer에서)
    trades_analysis = strat.analyzers.trades.get_analysis()
    total_commission = trades_analysis.get('total', {}).get('commission', 0.0)
    total_slippage = trades_analysis.get('total', {}).get('slippage', 0.0)
    total_fees = total_commission + total_slippage
    
    # 커스텀 CommissionAnalyzer에서 더 자세한 정보
    commission_analysis = strat.analyzers.commission.get_analysis()
    custom_total_commission = commission_analysis.get('total_commission', 0.0)
    custom_total_fees = commission_analysis.get('total_fees', 0.0) 
    avg_commission_per_trade = commission_analysis.get('avg_commission_per_trade', 0.0)
    
    

    
    logger.info(f"샤프 비율: {sharpe:.2f}")
    logger.info(f"최대 낙폭: {drawdown:.2f}%, {drawdown_money}")
    # logger.info(f"총 수익률(로그): {total_return:.2f}%")  # returns 분석기 제거로 인해 주석 처리
    logger.info(f"총 수익률(산술): {arithmetic_profit_percent:.2f}%")
    logger.info(f"총 수익: {arithmetic_profit:.2f}")
    logger.info(f"총 거래 횟수: {trades}")
    
  
   
    logger.info(f"💰 총 수수료 (TradeAnalyzer): ${total_commission:.2f}")
    logger.info(f"💰 총 슬리피지 (TradeAnalyzer): ${total_slippage:.2f}")
    logger.info(f"💰 총 비용 (TradeAnalyzer): ${total_fees:.2f}")
    logger.info(f"💰 총 수수료 (CommissionAnalyzer): ${custom_total_commission:.2f}")
    logger.info(f"💰 총 비용 (CommissionAnalyzer): ${custom_total_fees:.2f}")
    logger.info(f"💰 거래당 평균 수수료: ${avg_commission_per_trade:.4f}")
    logger.info(f"💰 수수료 비율 (총 수익 대비): {(custom_total_fees/arithmetic_profit*100):.2f}%" if arithmetic_profit != 0 else "💰 수수료 비율: 계산 불가 (수익이 0)")
    if avg_var is not None:
        logger.info(f"📊 평균 VaR (누적 전체 거래): ${avg_var:.2f}")
    
    if last_var is not None:
        logger.info(f"📌 마지막 거래의 VaR: ${last_var:.2f}")
  
    if var_per is not None:
        logger.info(f"📊 평균 VaR (누적 전체 거래) %: ${var_per:.2f}%")
    
    # 🚨 마진콜 분석 추가
    if hasattr(strat, 'margin_called') and strat.margin_called:
        logger.info("🚨 마진콜 발생!")
        logger.info(f"🚨 마진콜 발생 시점의 자본: ${strat.broker.getvalue():.2f}")
        
        # 마진콜 관련 추가 정보 출력
        if hasattr(strat, 'consecutive_losses'):
            logger.info(f"🚨 연속 손실 횟수: {strat.consecutive_losses}")
        
        if hasattr(strat, 'total_profit'):
            logger.info(f"🚨 총 손실률: {strat.total_profit:.2f}%")
        
        # 마진콜 발생 시 거래 로그에서 마진콜 정보 추출
        if hasattr(strat, 'trade_logs') and strat.trade_logs:
            margin_call_logs = [log for log in strat.trade_logs if log.get('action_type') == 'margin_call']
            if margin_call_logs:
                latest_margin_call = margin_call_logs[-1]
                logger.info(f"🚨 마지막 마진콜 정보:")
                logger.info(f"   - 청산가: ${latest_margin_call.get('liquidation_price', 0):.2f}")
                logger.info(f"   - 현재 하락률: {latest_margin_call.get('current_drop_percentage', 0):.2f}%")
                logger.info(f"   - 실제 레버리지: {latest_margin_call.get('actual_leverage', 0):.1f}배")
        
        logger.info("🚨 마진콜로 인해 모든 포지션이 강제 청산되었습니다.")
    else:
        logger.info("✅ 마진콜 미발생 - 안전한 거래 완료")
    
    # 마진콜 정보 추출
    margin_called = hasattr(strat, 'margin_called') and strat.margin_called
    consecutive_losses = getattr(strat, 'consecutive_losses', 0)
    total_profit = getattr(strat, 'total_profit', 0.0)
    
    # 마진콜 상세 정보
    margin_call_info = {}
    if margin_called and hasattr(strat, 'trade_logs') and strat.trade_logs:
        margin_call_logs = [log for log in strat.trade_logs if log.get('action_type') == 'margin_call']
        if margin_call_logs:
            latest_margin_call = margin_call_logs[-1]
            margin_call_info = {
                'liquidation_price': latest_margin_call.get('liquidation_price', 0),
                'current_drop_percentage': latest_margin_call.get('current_drop_percentage', 0),
                'actual_leverage': latest_margin_call.get('actual_leverage', 0)
            }
    
    # 백테스팅 결과를 CSV로 저장
    save_backtest_results(
        sharpe=sharpe,
        drawdown=drawdown,
        drawdown_money=drawdown_money,
        total_return=total_return,
        arithmetic_profit_percent=arithmetic_profit_percent,
        arithmetic_profit=arithmetic_profit,
        trades=trades,
        total_commission=total_commission,
        total_slippage=total_slippage,
        total_fees=total_fees,
        custom_total_commission=custom_total_commission,
        custom_total_fees=custom_total_fees,
        avg_commission_per_trade=avg_commission_per_trade,
        avg_var=avg_var,
        last_var=last_var,
        var_per=var_per,
        start_date=start_date,
        cash=cash,
        commission=commission,
        leverage=leverage,
        margin_called=margin_called,
        consecutive_losses=consecutive_losses,
        total_profit=total_profit,
        margin_call_info=margin_call_info
    )
    
    return {
        'sharpe': sharpe,
        'drawdown': drawdown,
        'total_return': total_return,
        'arithmetic_profit_percent': arithmetic_profit_percent,
        'arithmetic_profit': arithmetic_profit,
        'trades': trades,
        'total_commission': total_commission,
        'total_fees': total_fees,
        'avg_var': avg_var,
        'last_var': last_var,
        'var_per': var_per,
        'leverage': leverage,
        # 🚨 마진콜 관련 정보 추가
        'margin_called': margin_called,
        'consecutive_losses': consecutive_losses,
        'total_profit': total_profit,
        'margin_call_info': margin_call_info
    }


def save_backtest_results(**kwargs):
    """
    백테스팅 결과를 CSV 파일로 저장
    
    Args:
        **kwargs: 백테스팅 결과 데이터
    """
    # 타임스탬프 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 결과 저장 디렉토리 생성
    results_dir = 'btc_martingale_backtest/backtest_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 파일명 생성
    filename = f'backtest_results_{timestamp}.csv'
    filepath = os.path.join(results_dir, filename)
    
    # 백테스팅 결과 데이터 준비
    results_data = {
        'timestamp': timestamp,
        'start_date': kwargs.get('start_date', ''),
        'initial_cash': kwargs.get('cash', 0),
        'commission': kwargs.get('commission', 0),
        'leverage': kwargs.get('leverage', 0),
        'sharpe_ratio': kwargs.get('sharpe', 0.0),
        'max_drawdown_pct': kwargs.get('drawdown', 0.0),
        'max_drawdown_money': kwargs.get('drawdown_money', 0.0),
        'total_return_log_pct': kwargs.get('total_return', 0.0),
        'arithmetic_return_pct': kwargs.get('arithmetic_profit_percent', 0.0),
        'total_profit': kwargs.get('arithmetic_profit', 0.0),
        'total_trades': kwargs.get('trades', 0),
        'total_commission_tradeanalyzer': kwargs.get('total_commission', 0.0),
        'total_slippage_tradeanalyzer': kwargs.get('total_slippage', 0.0),
        'total_fees_tradeanalyzer': kwargs.get('total_fees', 0.0),
        'total_commission_commissionanalyzer': kwargs.get('custom_total_commission', 0.0),
        'total_fees_commissionanalyzer': kwargs.get('custom_total_fees', 0.0),
        'avg_commission_per_trade': kwargs.get('avg_commission_per_trade', 0.0),
        'commission_ratio_profit_pct': kwargs.get('commission_ratio_profit', 0.0),
        'avg_var_dollar': kwargs.get('avg_var', 0.0),
        'last_var_dollar': kwargs.get('last_var', 0.0),
        'avg_var_percent': kwargs.get('var_per', 0.0),
        'rf_threshold': kwargs.get('rf_threshold', 0.0),
        'leverage': kwargs.get('leverage', 0),
        # 🚨 마진콜 관련 정보 추가
        'margin_called': kwargs.get('margin_called', False),
        'consecutive_losses': kwargs.get('consecutive_losses', 0),
        'total_profit_loss': kwargs.get('total_profit', 0.0),
        'liquidation_price': kwargs.get('margin_call_info', {}).get('liquidation_price', 0.0),
        'current_drop_percentage': kwargs.get('margin_call_info', {}).get('current_drop_percentage', 0.0),
        'actual_leverage_at_margin_call': kwargs.get('margin_call_info', {}).get('actual_leverage', 0.0)
    }
    
    # 수수료 비율 계산
    if kwargs.get('arithmetic_profit', 0) != 0:
        commission_ratio = (kwargs.get('custom_total_fees', 0) / kwargs.get('arithmetic_profit', 1)) * 100
        results_data['commission_ratio_profit_pct'] = commission_ratio
    
    # DataFrame 생성 및 저장
    df_results = pd.DataFrame([results_data])
    df_results.to_csv(filepath, index=False)
    
    print(f"✅ 백테스팅 결과가 저장되었습니다: {filepath}")
    print(f"📊 저장된 결과: {len(results_data)}개 항목")
    
    return filepath


def run_pipeline():
    print()
    # features_df = add_features('binance_btcusdt_1m.csv')
    # rf_df = train_and_predict(features_df)
    # rf_df = train_and_predict_10pct_after_5pct(rf_df)  # 하락 예측 모델 추가
    # var_df = calc_var(rf_df, n_jobs=7)
    df = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_rf_var.csv', index_col=0, parse_dates=True)
    run_backtest(df)

if __name__ == '__main__':
    run_pipeline() 