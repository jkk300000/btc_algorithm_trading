import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_rf_threshold(data_path):
    """
    rf_pred 값 분포를 분석하여 임계값 0.6의 엄격성 확인
    """
    print("=" * 60)
    print("rf_pred 임계값 분석")
    print("=" * 60)
    
    # 데이터 로드
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"데이터 로드 완료: {len(df):,}개 행")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
        return
    
    # rf_pred 컬럼 확인
    if 'rf_pred' not in df.columns:
        print("❌ rf_pred 컬럼이 없습니다!")
        return
    
    # 테스트 기간 설정 (2022-09-01 이후)
    test_start = pd.Timestamp('2022-09-01 00:00:00+00:00').tz_localize(None)
    test_df = df[df.index >= test_start].copy()
    test_rf_pred = test_df['rf_pred'].dropna()
    
    print(f"\n테스트 기간 데이터: {len(test_df):,}개")
    print(f"rf_pred가 있는 데이터: {len(test_rf_pred):,}개")
    
    if len(test_rf_pred) == 0:
        print("❌ 테스트 기간에 rf_pred 값이 없습니다!")
        return
    
    # 기본 통계
    print(f"\nrf_pred 기본 통계:")
    print(f"  최소값: {test_rf_pred.min():.4f}")
    print(f"  최대값: {test_rf_pred.max():.4f}")
    print(f"  평균값: {test_rf_pred.mean():.4f}")
    print(f"  중앙값: {test_rf_pred.median():.4f}")
    print(f"  표준편차: {test_rf_pred.std():.4f}")
    
    # 분위수 분석
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n분위수 분석:")
    for p in percentiles:
        value = test_rf_pred.quantile(p/100)
        print(f"  {p}%: {value:.4f}")
    
    # 현재 임계값 0.6 분석
    current_threshold = 0.6
    above_threshold = (test_rf_pred >= current_threshold).sum()
    below_threshold = (test_rf_pred < current_threshold).sum()
    threshold_ratio = above_threshold / len(test_rf_pred) * 100
    
    print(f"\n현재 임계값 {current_threshold} 분석:")
    print(f"  임계값 이상: {above_threshold:,}개 ({threshold_ratio:.2f}%)")
    print(f"  임계값 미만: {below_threshold:,}개 ({100-threshold_ratio:.2f}%)")
    
    # 다양한 임계값에서 신호 개수 비교
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n임계값별 신호 개수:")
    for threshold in thresholds:
        signal_count = (test_rf_pred >= threshold).sum()
        signal_ratio = signal_count / len(test_rf_pred) * 100
        print(f"  ≥{threshold}: {signal_count:,}개 ({signal_ratio:.2f}%)")
    
    # 엄격성 평가
    print(f"\n엄격성 평가:")
    if threshold_ratio < 1.0:
        print(f"  ⚠️ 매우 엄격함: {threshold_ratio:.2f}%만 신호 발생")
        print(f"  → 임계값을 낮춰야 할 가능성 높음")
    elif threshold_ratio < 5.0:
        print(f"  ⚠️ 엄격함: {threshold_ratio:.2f}% 신호 발생")
        print(f"  → 임계값 조정 고려")
    elif threshold_ratio < 15.0:
        print(f"  ✅ 적절함: {threshold_ratio:.2f}% 신호 발생")
        print(f"  → 현재 임계값 유지 가능")
    else:
        print(f"  ⚠️ 너무 관대함: {threshold_ratio:.2f}% 신호 발생")
        print(f"  → 임계값을 높여야 할 가능성")
    
    # 권장 임계값 제안
    print(f"\n권장 임계값 제안:")
    for target_ratio in [1.0, 2.0, 5.0, 10.0]:
        # 목표 비율에 가장 가까운 임계값 찾기
        for threshold in np.arange(0.1, 1.0, 0.01):
            ratio = (test_rf_pred >= threshold).sum() / len(test_rf_pred) * 100
            if ratio <= target_ratio:
                print(f"  {target_ratio}% 신호 목표: 임계값 {threshold:.2f}")
                break
    
    # 시각화
    print(f"\n시각화 생성 중...")
    
    plt.figure(figsize=(15, 10))
    
    # 1) rf_pred 분포 히스토그램
    plt.subplot(2, 3, 1)
    plt.hist(test_rf_pred, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(current_threshold, color='red', linestyle='--', linewidth=2, label=f'현재 임계값 ({current_threshold})')
    plt.xlabel('rf_pred 값')
    plt.ylabel('빈도')
    plt.title('rf_pred 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2) 누적 분포
    plt.subplot(2, 3, 2)
    sorted_values = np.sort(test_rf_pred)
    cumulative_ratio = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
    plt.plot(sorted_values, cumulative_ratio)
    plt.axvline(current_threshold, color='red', linestyle='--', linewidth=2, label=f'현재 임계값 ({current_threshold})')
    plt.xlabel('rf_pred 값')
    plt.ylabel('누적 비율 (%)')
    plt.title('누적 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3) 임계값별 신호 개수
    plt.subplot(2, 3, 3)
    signal_counts = []
    for threshold in thresholds:
        signal_count = (test_rf_pred >= threshold).sum()
        signal_counts.append(signal_count)
    
    plt.bar([str(t) for t in thresholds], signal_counts)
    plt.xlabel('임계값')
    plt.ylabel('신호 개수')
    plt.title('임계값별 신호 개수')
    plt.grid(True, alpha=0.3)
    
    # 4) 임계값별 신호 비율
    plt.subplot(2, 3, 4)
    signal_ratios = []
    for threshold in thresholds:
        signal_ratio = (test_rf_pred >= threshold).sum() / len(test_rf_pred) * 100
        signal_ratios.append(signal_ratio)
    
    plt.bar([str(t) for t in thresholds], signal_ratios)
    plt.xlabel('임계값')
    plt.ylabel('신호 비율 (%)')
    plt.title('임계값별 신호 비율')
    plt.grid(True, alpha=0.3)
    
    # 5) 월별 평균 rf_pred
    plt.subplot(2, 3, 5)
    test_df['month'] = test_df.index.to_period('M')
    monthly_avg = test_df.groupby('month')['rf_pred'].mean()
    monthly_avg.plot(kind='bar')
    plt.axhline(y=current_threshold, color='red', linestyle='--', linewidth=2, label=f'현재 임계값 ({current_threshold})')
    plt.xlabel('월')
    plt.ylabel('평균 rf_pred')
    plt.title('월별 평균 rf_pred')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6) 시간별 rf_pred (샘플)
    plt.subplot(2, 3, 6)
    sample_data = test_rf_pred.head(1000)
    plt.plot(sample_data.index, sample_data.values, alpha=0.7)
    plt.axhline(y=current_threshold, color='red', linestyle='--', linewidth=2, label=f'현재 임계값 ({current_threshold})')
    plt.xlabel('시간')
    plt.ylabel('rf_pred')
    plt.title('시간별 rf_pred (샘플)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rf_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'current_threshold': current_threshold,
        'signal_ratio': threshold_ratio,
        'total_signals': above_threshold,
        'mean_rf_pred': test_rf_pred.mean(),
        'median_rf_pred': test_rf_pred.median()
    }

if __name__ == "__main__":
    # 데이터 파일 경로 설정
    data_path = "C:/선물데이터/binance_btcusdt_1m_rf_var.csv"
    
    try:
        results = analyze_rf_threshold(data_path)
        print(f"\n분석 완료!")
        print(f"현재 임계값: {results['current_threshold']}")
        print(f"신호 비율: {results['signal_ratio']:.2f}%")
        print(f"총 신호 개수: {results['total_signals']:,}")
        print(f"평균 rf_pred: {results['mean_rf_pred']:.4f}")
        print(f"중앙값 rf_pred: {results['median_rf_pred']:.4f}")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}") 