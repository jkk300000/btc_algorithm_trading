import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os

# 한글 폰트 설정
import matplotlib.font_manager as fm

# Windows에서 사용 가능한 한글 폰트 찾기
def find_korean_font():
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    korean_fonts = []
    
    for font_path in font_list:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            # 한글을 지원하는 폰트 찾기
            if any(keyword in font_name.lower() for keyword in ['malgun', 'gulim', 'dotum', 'batang', 'gungsuh', 'nanum', 'noto']):
                korean_fonts.append(font_name)
        except:
            continue
    
    return korean_fonts[0] if korean_fonts else 'DejaVu Sans'

# 한글 폰트 설정
try:
    korean_font = find_korean_font()
    plt.rcParams['font.family'] = korean_font
    print(f"한글 폰트 설정: {korean_font}")
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

plt.rcParams['axes.unicode_minus'] = False

def create_performance_charts():
    """실제 백테스팅 데이터 기반 성과 지표 차트 생성"""
    
    # 실제 백테스팅 결과 데이터
    backtest_data = [
        {
            'timestamp': '20250821_101058',
            'leverage': 8,
            'sharpe_ratio': 1.546,
            'max_drawdown_pct': 66.89,
            'arithmetic_return_pct': 2010.41,
            'total_profit': 20104.07,
            'total_trades': 750,
            'commission_ratio_profit_pct': 17.08,
            'avg_var_dollar': 1450.01,
            'margin_called': False
        },
        {
            'timestamp': '20250821_104113',
            'leverage': 7,
            'sharpe_ratio': 1.494,
            'max_drawdown_pct': 61.73,
            'arithmetic_return_pct': 1524.03,
            'total_profit': 15240.27,
            'total_trades': 779,
            'commission_ratio_profit_pct': 17.51,
            'avg_var_dollar': 1416.61,
            'margin_called': False
        },
        {
            'timestamp': '20250821_105003',
            'leverage': 6,
            'sharpe_ratio': 1.473,
            'max_drawdown_pct': 54.05,
            'arithmetic_return_pct': 991.20,
            'total_profit': 9911.99,
            'total_trades': 786,
            'commission_ratio_profit_pct': 17.30,
            'avg_var_dollar': 1417.04,
            'margin_called': False
        }
    ]
    
    df = pd.DataFrame(backtest_data)
    
    # 1. 수익률 비교 차트
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 차트 1: 수익률 비교
    colors = ['#2E8B57', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(range(len(df)), df['arithmetic_return_pct'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('백테스트 수익률 비교', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('백테스트 실행', fontsize=12)
    ax1.set_ylabel('수익률 (%)', fontsize=12)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([f'Run {i+1}\n(Leverage {lev}x)' for i, lev in enumerate(df['leverage'])])
    ax1.grid(True, alpha=0.3)
    
    # 수익률 값 표시
    for i, (bar, ret) in enumerate(zip(bars, df['arithmetic_return_pct'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(df['arithmetic_return_pct'])*0.02,
                f'{ret:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 차트 2: 샤프 비율 vs 수익률
    scatter = ax2.scatter(df['sharpe_ratio'], df['arithmetic_return_pct'], 
                         c=df['arithmetic_return_pct'], cmap='RdYlGn', s=200, alpha=0.8, edgecolors='black')
    ax2.set_title('샤프 비율 vs 수익률 비교', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('샤프 비율', fontsize=12)
    ax2.set_ylabel('수익률 (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 각 점에 레버리지 표시
    for i, (sharpe, ret, lev) in enumerate(zip(df['sharpe_ratio'], df['arithmetic_return_pct'], df['leverage'])):
        ax2.annotate(f'{lev}x', (sharpe, ret), xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)
    
    # 차트 3: 리스크-수익 매트릭스
    risk_return = ax3.scatter(df['max_drawdown_pct'], df['arithmetic_return_pct'], 
                             c=df['sharpe_ratio'], s=200, cmap='viridis', alpha=0.8, edgecolors='black')
    ax3.set_title('리스크-수익 매트릭스', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('MDD (%)', fontsize=12)
    ax3.set_ylabel('수익률 (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 각 점에 레버리지 표시
    for i, (dd, ret, lev) in enumerate(zip(df['max_drawdown_pct'], df['arithmetic_return_pct'], df['leverage'])):
        ax3.annotate(f'{lev}x', (dd, ret), xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)
    
    # 차트 4: 레버리지별 성과
    ax4.bar(df['leverage'], df['arithmetic_return_pct'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('레버리지별 성과', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('레버리지', fontsize=12)
    ax4.set_ylabel('수익률 (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 수익률 값 표시
    for i, (lev, ret) in enumerate(zip(df['leverage'], df['arithmetic_return_pct'])):
        ax4.text(lev, ret + max(df['arithmetic_return_pct'])*0.02,
                f'{ret:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('performance_charts.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("성과 차트가 저장되었습니다: performance_charts.png")
    
    # 2. 성과 요약 대시보드
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 대시보드 1: 핵심 지표 요약
    ax1.axis('off')
    
    # 핵심 지표 계산
    best_return = df['arithmetic_return_pct'].max()
    avg_return = df['arithmetic_return_pct'].mean()
    avg_sharpe = df['sharpe_ratio'].mean()
    avg_drawdown = df['max_drawdown_pct'].mean()
    total_trades = df['total_trades'].sum()
    avg_commission = df['commission_ratio_profit_pct'].mean()
    avg_var = df['avg_var_dollar'].mean()
    
    summary_text = f"""
    PORTFOLIO PERFORMANCE SUMMARY
    
    Best Return: {best_return:.1f}%
    Average Return: {avg_return:.1f}%
    Average Sharpe: {avg_sharpe:.3f}
    Average Drawdown: {avg_drawdown:.1f}%
    Total Trades: {total_trades:,}
    Avg Commission: {avg_commission:.1f}%
    Average VaR: ${avg_var:.0f}
    
    Total Backtests: {len(df)}
    All Profitable: {len(df)}/{len(df)}
    Margin Calls: 0
    """
    
    ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    # 대시보드 2: 수수료 비율 분석
    commission_data = df['commission_ratio_profit_pct'].values
    profit_data = [100 - comm for comm in commission_data]
    
    wedges, texts, autotexts = ax2.pie([np.mean(commission_data), np.mean(profit_data)], 
                                      labels=['Commission', 'Net Profit'], 
                                      colors=['#FF6B6B', '#4ECDC4'], 
                                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    ax2.set_title('💸 Commission vs Net Profit Ratio', fontsize=16, fontweight='bold', pad=20)
    
    # 대시보드 3: VaR 분석
    var_values = df['avg_var_dollar'].values
    bars = ax3.bar(range(len(var_values)), var_values, color='orange', alpha=0.8, edgecolor='black')
    ax3.set_title('📊 Value at Risk (VaR) Analysis', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Backtest Run', fontsize=12)
    ax3.set_ylabel('VaR ($)', fontsize=12)
    ax3.set_xticks(range(len(var_values)))
    ax3.set_xticklabels([f'Run {i+1}' for i in range(len(var_values))])
    ax3.grid(True, alpha=0.3)
    
    # VaR 값 표시
    for i, (bar, var) in enumerate(zip(bars, var_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(var_values)*0.01,
                f'${var:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 대시보드 4: 성과 등급 분포
    def classify_performance(return_pct):
        if return_pct > 2000:
            return 'Excellent (2000%+)'
        elif return_pct > 1000:
            return 'Outstanding (1000-2000%)'
        elif return_pct > 500:
            return 'Very Good (500-1000%)'
        else:
            return 'Good (0-500%)'
    
    performance_grades = [classify_performance(r) for r in df['arithmetic_return_pct']]
    grade_counts = pd.Series(performance_grades).value_counts()
    
    colors_pie = ['#2E8B57', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax4.pie(grade_counts.values, labels=grade_counts.index, 
                                       colors=colors_pie[:len(grade_counts)], autopct='%1.1f%%', 
                                       startangle=90, textprops={'fontsize': 10})
    ax4.set_title('🏆 Performance Grade Distribution', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("성과 대시보드가 저장되었습니다: performance_dashboard.png")
    
    # 3. 거래 분석 차트
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 거래 횟수 분석
    trades = df['total_trades'].values
    bars = ax1.bar(range(len(trades)), trades, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('📈 Trading Frequency Analysis', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Backtest Run', fontsize=12)
    ax1.set_ylabel('Total Trades', fontsize=12)
    ax1.set_xticks(range(len(trades)))
    ax1.set_xticklabels([f'Run {i+1}\n(Leverage {lev}x)' for i, lev in enumerate(df['leverage'])])
    ax1.grid(True, alpha=0.3)
    
    # 거래 횟수 값 표시
    for i, (bar, trade) in enumerate(zip(bars, trades)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(trades)*0.01,
                f'{trade:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 레버리지별 샤프 비율
    ax2.plot(df['leverage'], df['sharpe_ratio'], marker='o', linewidth=3, markersize=10, 
             color='#2E8B57', markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2E8B57')
    ax2.set_title('⚡ Sharpe Ratio by Leverage', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Leverage', fontsize=12)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 샤프 비율 값 표시
    for i, (lev, sharpe) in enumerate(zip(df['leverage'], df['sharpe_ratio'])):
        ax2.annotate(f'{sharpe:.3f}', (lev, sharpe), xytext=(0, 10), textcoords='offset points', 
                    ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('trading_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("거래 분석 차트가 저장되었습니다: trading_analysis.png")
    
    return df

if __name__ == "__main__":
    print("🚀 BTC Algorithm Trading System - 성과 지표 시각화")
    print("="*60)
    
    df = create_performance_charts()
    
    print("="*60)
    print("✅ 모든 성과 그래프가 생성되었습니다!")
    print("📊 생성된 파일:")
    print("   - performance_charts.png (성과 비교 차트)")
    print("   - performance_dashboard.png (성과 대시보드)")
    print("   - trading_analysis.png (거래 분석 차트)")
    
    # 데이터 요약 출력
    print("\n📈 백테스팅 결과 요약:")
    print(f"   - 최고 수익률: {df['arithmetic_return_pct'].max():.1f}%")
    print(f"   - 평균 수익률: {df['arithmetic_return_pct'].mean():.1f}%")
    print(f"   - 평균 샤프 비율: {df['sharpe_ratio'].mean():.3f}")
    print(f"   - 평균 최대 낙폭: {df['max_drawdown_pct'].mean():.1f}%")
    print(f"   - 총 거래 횟수: {df['total_trades'].sum():,}")
