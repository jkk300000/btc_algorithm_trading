import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def analyze_signal_frequency(data_path):
    """
    신호 빈도를 자세히 분석
    """
    print("=" * 60)
    print("신호 빈도 상세 분석")
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
    
    # 1. 임계값별 신호 분석
    print("\n" + "="*50)
    print("1. 임계값별 신호 분석")
    print("="*50)
    
    thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        signal_count = (test_rf_pred >= threshold).sum()
        signal_ratio = signal_count / len(test_rf_pred) * 100
        
        # 월별 신호 개수도 계산
        test_df['signal'] = (test_df['rf_pred'] >= threshold)
        monthly_signals = test_df.groupby(test_df.index.to_period('M'))['signal'].sum()
        avg_monthly_signals = monthly_signals.mean()
        
        print(f"임계값 {threshold}:")
        print(f"  총 신호: {signal_count:,}개 ({signal_ratio:.2f}%)")
        print(f"  월평균 신호: {avg_monthly_signals:.1f}개")
        
        # 신호 간격 분석
        signal_points = test_rf_pred[test_rf_pred >= threshold]
        if len(signal_points) > 1:
            signal_dates = signal_points.index.sort_values()
            signal_intervals = signal_dates.to_series().diff().dropna()
            avg_interval_hours = signal_intervals.dt.total_seconds().mean() / 3600
            print(f"  평균 신호 간격: {avg_interval_hours:.1f}시간")
        else:
            print(f"  평균 신호 간격: 신호 부족")
        print()
    
    # 2. 월별 신호 패턴 분석
    print("\n" + "="*50)
    print("2. 월별 신호 패턴 분석")
    print("="*50)
    
    test_df['month'] = test_df.index.to_period('M')
    test_df['signal_06'] = (test_df['rf_pred'] >= 0.6)
    
    monthly_stats = test_df.groupby('month').agg({
        'rf_pred': ['count', 'mean', 'std'],
        'signal_06': 'sum'
    }).round(4)
    
    monthly_stats.columns = ['총_데이터', '평균_rf_pred', '표준편차', '신호_개수']
    monthly_stats['신호_비율'] = (monthly_stats['신호_개수'] / monthly_stats['총_데이터'] * 100).round(2)
    
    print(monthly_stats)
    
    # 3. 신호 없는 구간 분석
    print("\n" + "="*50)
    print("3. 신호 없는 구간 분석")
    print("="*50)
    
    for threshold in [0.5, 0.6, 0.7]:
        signal_points = test_rf_pred[test_rf_pred >= threshold]
        if len(signal_points) > 1:
            signal_dates = signal_points.index.sort_values()
            signal_intervals = signal_dates.to_series().diff().dropna()
            
            # 긴 간격 구간 찾기
            long_intervals = signal_intervals[signal_intervals > timedelta(days=7)]
            
            print(f"\n임계값 {threshold} 기준:")
            print(f"  총 신호: {len(signal_points):,}개")
            print(f"  평균 간격: {signal_intervals.mean()}")
            print(f"  최대 간격: {signal_intervals.max()}")
            print(f"  7일 이상 신호 없는 구간: {len(long_intervals)}개")
            
            if len(long_intervals) > 0:
                print(f"  긴 간격 구간들:")
                for date, interval in long_intervals.head(5).items():
                    print(f"    {date}: {interval}")
    
    # 4. 권장 임계값 제안
    print("\n" + "="*50)
    print("4. 권장 임계값 제안")
    print("="*50)
    
    print("현재 상황 분석:")
    print("- 임계값 0.6: 19.56% 신호")
    print("- 월평균 약 1,500개 신호")
    print("- 평균 간격: 약 2-3시간")
    
    print("\n권장사항:")
    print("1) 임계값 0.5 사용 시:")
    signal_count_05 = (test_rf_pred >= 0.5).sum()
    signal_ratio_05 = signal_count_05 / len(test_rf_pred) * 100
    print(f"   - 신호 비율: {signal_ratio_05:.2f}%")
    print(f"   - 월평균 신호: 약 {signal_ratio_05/19.56*1500:.0f}개")
    
    print("\n2) 임계값 0.55 사용 시:")
    signal_count_055 = (test_rf_pred >= 0.55).sum()
    signal_ratio_055 = signal_count_055 / len(test_rf_pred) * 100
    print(f"   - 신호 비율: {signal_ratio_055:.2f}%")
    print(f"   - 월평균 신호: 약 {signal_ratio_055/19.56*1500:.0f}개")
    
    print("\n3) 적응형 임계값 사용:")
    print("   - 시장 상황에 따라 0.5~0.7 범위에서 동적 조정")
    print("   - 변동성이 높을 때: 0.5")
    print("   - 변동성이 낮을 때: 0.7")
    
    # 5. 시각화
    print("\n" + "="*50)
    print("5. 시각화 생성")
    print("="*50)
    
    plt.figure(figsize=(15, 12))
    
    # 1) 임계값별 신호 비율
    plt.subplot(3, 2, 1)
    signal_ratios = []
    for threshold in thresholds:
        signal_ratio = (test_rf_pred >= threshold).sum() / len(test_rf_pred) * 100
        signal_ratios.append(signal_ratio)
    
    plt.bar([str(t) for t in thresholds], signal_ratios)
    plt.xlabel('임계값')
    plt.ylabel('신호 비율 (%)')
    plt.title('임계값별 신호 비율')
    plt.grid(True, alpha=0.3)
    
    # 2) 월별 신호 개수
    plt.subplot(3, 2, 2)
    monthly_signals = test_df.groupby('month')['signal_06'].sum()
    monthly_signals.plot(kind='bar')
    plt.xlabel('월')
    plt.ylabel('신호 개수')
    plt.title('월별 신호 개수 (임계값 0.6)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3) 월별 평균 rf_pred
    plt.subplot(3, 2, 3)
    monthly_avg = test_df.groupby('month')['rf_pred'].mean()
    monthly_avg.plot(kind='bar')
    plt.axhline(y=0.6, color='red', linestyle='--', label='임계값 (0.6)')
    plt.xlabel('월')
    plt.ylabel('평균 rf_pred')
    plt.title('월별 평균 rf_pred')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4) 신호 간격 분포
    plt.subplot(3, 2, 4)
    signal_points = test_rf_pred[test_rf_pred >= 0.6]
    if len(signal_points) > 1:
        signal_dates = signal_points.index.sort_values()
        signal_intervals = signal_dates.to_series().diff().dropna()
        signal_intervals_hours = signal_intervals.dt.total_seconds() / 3600
        
        plt.hist(signal_intervals_hours, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('신호 간격 (시간)')
        plt.ylabel('빈도')
        plt.title('신호 간격 분포 (임계값 0.6)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, '신호가 부족합니다', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('신호 간격 분포')
    
    # 5) 시간별 rf_pred (샘플)
    plt.subplot(3, 2, 5)
    sample_data = test_rf_pred.head(2000)
    plt.plot(sample_data.index, sample_data.values, alpha=0.7)
    plt.axhline(y=0.6, color='red', linestyle='--', label='임계값 (0.6)')
    plt.axhline(y=0.5, color='orange', linestyle='--', label='제안 임계값 (0.5)')
    plt.xlabel('시간')
    plt.ylabel('rf_pred')
    plt.title('시간별 rf_pred (샘플)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6) 누적 신호 비율
    plt.subplot(3, 2, 6)
    sorted_values = np.sort(test_rf_pred)
    cumulative_ratio = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
    plt.plot(sorted_values, cumulative_ratio)
    plt.axvline(0.6, color='red', linestyle='--', label='현재 임계값 (0.6)')
    plt.axvline(0.5, color='orange', linestyle='--', label='제안 임계값 (0.5)')
    plt.xlabel('rf_pred 값')
    plt.ylabel('누적 비율 (%)')
    plt.title('누적 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('signal_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'current_threshold': 0.6,
        'current_signal_ratio': 19.56,
        'suggested_threshold_05': 0.5,
        'suggested_signal_ratio_05': signal_ratio_05,
        'suggested_threshold_055': 0.55,
        'suggested_signal_ratio_055': signal_ratio_055
    }

if __name__ == "__main__":
    # 데이터 파일 경로 설정
    data_path = "rf_down_good_results.csv"
    
    try:
        results = analyze_signal_frequency(data_path)
        print(f"\n분석 완료!")
        print(f"현재 임계값: {results['current_threshold']}")
        print(f"현재 신호 비율: {results['current_signal_ratio']:.2f}%")
        print(f"제안 임계값 0.5: {results['suggested_signal_ratio_05']:.2f}%")
        print(f"제안 임계값 0.55: {results['suggested_signal_ratio_055']:.2f}%")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}") 