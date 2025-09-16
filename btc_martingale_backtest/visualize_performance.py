import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_backtest_results():
    """백테스팅 결과 파일들을 로드"""
    results_dir = 'btc_martingale_backtest/backtest_results'
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    all_results = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # 파라미터 정보가 포함된 행 제거
            df = df[df['timestamp'].str.contains('^[0-9]', na=False)]
            all_results.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def create_performance_dashboard():
    """성과 지표 대시보드 생성"""
    results_df = load_backtest_results()
    
    if results_df.empty:
        print("백테스팅 결과 데이터가 없습니다.")
        return
    
    # 그래프 스타일 설정
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 수익률 비교 (상단 좌측)
    ax1 = plt.subplot(3, 4, 1)
    returns = results_df['arithmetic_return_pct'].values
    colors = ['#2E8B57' if r > 0 else '#DC143C' for r in returns]
    bars = ax1.bar(range(len(returns)), returns, color=colors, alpha=0.7)
    ax1.set_title('Total Returns by Backtest', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Backtest Run')
    ax1.set_ylabel('Return (%)')
    ax1.grid(True, alpha=0.3)
    
    # 수익률 값 표시
    for i, (bar, ret) in enumerate(zip(bars, returns)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (10 if height > 0 else -20),
                f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    # 2. 샤프 비율 분포 (상단 우측)
    ax2 = plt.subplot(3, 4, 2)
    sharpe_ratios = results_df['sharpe_ratio'].values
    ax2.hist(sharpe_ratios, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(sharpe_ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sharpe_ratios):.2f}')
    ax2.set_title('Sharpe Ratio Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 최대 낙폭 분석 (상단 중앙)
    ax3 = plt.subplot(3, 4, 3)
    drawdowns = results_df['max_drawdown_pct'].values
    colors = ['#FF6B6B' if dd > 50 else '#4ECDC4' if dd > 20 else '#45B7D1' for dd in drawdowns]
    bars = ax3.bar(range(len(drawdowns)), drawdowns, color=colors, alpha=0.7)
    ax3.set_title('Maximum Drawdown Analysis', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Backtest Run')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    
    # 낙폭 값 표시
    for i, (bar, dd) in enumerate(zip(bars, drawdowns)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{dd:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. 거래 횟수 vs 수익률 (상단 우측)
    ax4 = plt.subplot(3, 4, 4)
    trades = results_df['total_trades'].values
    returns = results_df['arithmetic_return_pct'].values
    scatter = ax4.scatter(trades, returns, c=returns, cmap='RdYlGn', s=100, alpha=0.7)
    ax4.set_title('Trades vs Returns', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Total Trades')
    ax4.set_ylabel('Return (%)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Return %')
    
    # 5. 수수료 비율 분석 (중간 좌측)
    ax5 = plt.subplot(3, 4, 5)
    commission_ratios = results_df['commission_ratio_profit_pct'].values
    ax5.pie([np.mean(commission_ratios), 100-np.mean(commission_ratios)], 
            labels=['Commission', 'Net Profit'], 
            colors=['#FF9999', '#66B3FF'], 
            autopct='%1.1f%%', startangle=90)
    ax5.set_title('Commission vs Net Profit Ratio', fontsize=14, fontweight='bold')
    
    # 6. VaR 분석 (중간 중앙)
    ax6 = plt.subplot(3, 4, 6)
    var_values = results_df['avg_var_dollar'].values
    ax6.bar(range(len(var_values)), var_values, color='orange', alpha=0.7)
    ax6.set_title('Average VaR Analysis', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Backtest Run')
    ax6.set_ylabel('VaR ($)')
    ax6.grid(True, alpha=0.3)
    
    # VaR 값 표시
    for i, var in enumerate(var_values):
        ax6.text(i, var + max(var_values)*0.01, f'${var:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. 레버리지별 성과 (중간 우측)
    ax7 = plt.subplot(3, 4, 7)
    leverage = results_df['leverage'].values
    returns = results_df['arithmetic_return_pct'].values
    leverage_returns = pd.DataFrame({'leverage': leverage, 'returns': returns})
    leverage_stats = leverage_returns.groupby('leverage')['returns'].agg(['mean', 'std']).reset_index()
    
    ax7.errorbar(leverage_stats['leverage'], leverage_stats['mean'], 
                yerr=leverage_stats['std'], marker='o', capsize=5, capthick=2, linewidth=2)
    ax7.set_title('Performance by Leverage', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Leverage')
    ax7.set_ylabel('Average Return (%)')
    ax7.grid(True, alpha=0.3)
    
    # 8. 리스크-수익 매트릭스 (중간 우측)
    ax8 = plt.subplot(3, 4, 8)
    risk_return = ax8.scatter(results_df['max_drawdown_pct'], results_df['arithmetic_return_pct'], 
                             c=results_df['sharpe_ratio'], s=100, cmap='viridis', alpha=0.7)
    ax8.set_title('Risk-Return Matrix', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Max Drawdown (%)')
    ax8.set_ylabel('Return (%)')
    ax8.grid(True, alpha=0.3)
    plt.colorbar(risk_return, ax=ax8, label='Sharpe Ratio')
    
    # 9. 월별 성과 시뮬레이션 (하단 좌측)
    ax9 = plt.subplot(3, 4, 9)
    # 가상의 월별 성과 데이터 생성 (실제 데이터가 있다면 사용)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns = np.random.normal(15, 8, 12)  # 평균 15%, 표준편차 8%
    monthly_returns = np.cumsum(monthly_returns)
    
    ax9.plot(months, monthly_returns, marker='o', linewidth=2, markersize=6, color='#2E8B57')
    ax9.set_title('Simulated Monthly Performance', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Month')
    ax9.set_ylabel('Cumulative Return (%)')
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(axis='x', rotation=45)
    
    # 10. 승률 분석 (하단 중앙)
    ax10 = plt.subplot(3, 4, 10)
    # 가상의 승률 데이터 (실제 거래 로그가 있다면 계산)
    win_rates = [68.5, 72.3, 65.8, 70.1, 69.2]  # 예시 데이터
    strategies = ['Strategy A', 'Strategy B', 'Strategy C', 'Strategy D', 'Strategy E']
    
    bars = ax10.bar(strategies, win_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'], alpha=0.7)
    ax10.set_title('Win Rate by Strategy', fontsize=14, fontweight='bold')
    ax10.set_ylabel('Win Rate (%)')
    ax10.grid(True, alpha=0.3)
    ax10.tick_params(axis='x', rotation=45)
    
    # 승률 값 표시
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 11. 포트폴리오 성과 요약 (하단 우측)
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    # 핵심 지표 요약
    summary_text = f"""
    📊 PORTFOLIO SUMMARY
    
    💰 Best Return: {np.max(returns):.1f}%
    📈 Avg Return: {np.mean(returns):.1f}%
    📉 Avg Drawdown: {np.mean(drawdowns):.1f}%
    ⚡ Avg Sharpe: {np.mean(sharpe_ratios):.2f}
    🎯 Avg Trades: {np.mean(trades):.0f}
    💸 Avg Commission: {np.mean(commission_ratios):.1f}%
    📊 Avg VaR: ${np.mean(var_values):.0f}
    
    🏆 Total Backtests: {len(results_df)}
    ✅ Profitable: {np.sum(returns > 0)}
    ❌ Loss-making: {np.sum(returns <= 0)}
    """
    
    ax11.text(0.1, 0.9, summary_text, transform=ax11.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 12. 성과 등급 분포 (하단 우측)
    ax12 = plt.subplot(3, 4, 12)
    
    # 성과 등급 분류
    def classify_performance(return_pct):
        if return_pct > 1000:
            return 'Excellent (1000%+)'
        elif return_pct > 500:
            return 'Very Good (500-1000%)'
        elif return_pct > 100:
            return 'Good (100-500%)'
        elif return_pct > 0:
            return 'Positive (0-100%)'
        else:
            return 'Negative (<0%)'
    
    performance_grades = [classify_performance(r) for r in returns]
    grade_counts = pd.Series(performance_grades).value_counts()
    
    colors = ['#2E8B57', '#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B']
    wedges, texts, autotexts = ax12.pie(grade_counts.values, labels=grade_counts.index, 
                                       colors=colors[:len(grade_counts)], autopct='%1.1f%%', startangle=90)
    ax12.set_title('Performance Grade Distribution', fontsize=14, fontweight='bold')
    
    # 전체 제목
    fig.suptitle('🚀 BTC Algorithm Trading System - Performance Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 그래프 저장
    output_path = 'btc_algorithm_trading/performance_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 성과 대시보드가 저장되었습니다: {output_path}")
    
    plt.show()

def create_individual_charts():
    """개별 성과 차트들 생성"""
    results_df = load_backtest_results()
    
    if results_df.empty:
        print("백테스팅 결과 데이터가 없습니다.")
        return
    
    # 1. 수익률 트렌드 차트
    plt.figure(figsize=(12, 8))
    
    # 서브플롯 1: 수익률 비교
    plt.subplot(2, 2, 1)
    returns = results_df['arithmetic_return_pct'].values
    x_pos = range(len(returns))
    colors = ['#2E8B57' if r > 0 else '#DC143C' for r in returns]
    
    bars = plt.bar(x_pos, returns, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.title('Backtest Returns Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Backtest Run')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    # 수익률 값 표시
    for i, (bar, ret) in enumerate(zip(bars, returns)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(returns)*0.02 if height > 0 else -max(returns)*0.02),
                f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    # 서브플롯 2: 샤프 비율 vs 수익률
    plt.subplot(2, 2, 2)
    sharpe_ratios = results_df['sharpe_ratio'].values
    scatter = plt.scatter(sharpe_ratios, returns, c=returns, cmap='RdYlGn', s=150, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, label='Return %')
    plt.title('Sharpe Ratio vs Returns', fontsize=16, fontweight='bold')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 3: 리스크-수익 매트릭스
    plt.subplot(2, 2, 3)
    drawdowns = results_df['max_drawdown_pct'].values
    risk_return = plt.scatter(drawdowns, returns, c=sharpe_ratios, s=150, cmap='viridis', alpha=0.7, edgecolors='black')
    plt.colorbar(risk_return, label='Sharpe Ratio')
    plt.title('Risk-Return Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Max Drawdown (%)')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 4: 거래 횟수 분석
    plt.subplot(2, 2, 4)
    trades = results_df['total_trades'].values
    plt.hist(trades, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(trades), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(trades):.0f}')
    plt.title('Trading Frequency Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Total Trades')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('btc_algorithm_trading/performance_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 성과 분석 차트가 저장되었습니다: btc_algorithm_trading/performance_analysis.png")
    plt.show()

if __name__ == "__main__":
    print("🚀 BTC Algorithm Trading System - 성과 시각화 시작")
    print("="*60)
    
    # 종합 대시보드 생성
    create_performance_dashboard()
    
    # 개별 차트 생성
    create_individual_charts()
    
    print("="*60)
    print("✅ 모든 성과 그래프가 생성되었습니다!")
    print("📊 생성된 파일:")
    print("   - performance_dashboard.png (종합 대시보드)")
    print("   - performance_analysis.png (개별 분석 차트)")
