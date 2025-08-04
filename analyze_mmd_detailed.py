import pandas as pd
import numpy as np

# 로그 파일 읽기
df = pd.read_csv('trade_logs/trade_logs_20250801_155952.csv')

# broker_value 컬럼만 추출
broker_values = df['broker_value'].dropna()

print(f"총 데이터 수: {len(broker_values)}")
print(f"초기 자본: {broker_values.iloc[0]:.2f}")
print(f"최종 자본: {broker_values.iloc[-1]:.2f}")

# 최고점과 최저점 찾기
max_value = broker_values.max()
max_index = broker_values.idxmax()
min_value = broker_values.min()
min_index = broker_values.idxmin()

print(f"\n=== 최고점/최저점 분석 ===")
print(f"최고점: {max_value:.2f} (인덱스: {max_index})")
print(f"최저점: {min_value:.2f} (인덱스: {min_index})")

# 최고점 이후의 최저점 찾기
if max_index < min_index:
    print("✅ 최고점 이후 최저점이 발생했습니다.")
    # 최고점 이후의 MMD 계산
    after_peak = broker_values[max_index:]
    min_after_peak = after_peak.min()
    min_after_peak_index = after_peak.idxmin()
    
    drawdown_after = max_value - min_after_peak
    drawdown_percent_after = (drawdown_after / max_value) * 100
    
    print(f"최고점 이후 최저점: {min_after_peak:.2f} (인덱스: {min_after_peak_index})")
    print(f"최고점 이후 MMD: {drawdown_percent_after:.2f}%")
else:
    print("❌ 최고점 이후 최저점이 발생하지 않았습니다.")
    # 최고점 이후의 최저점 찾기
    after_peak = broker_values[max_index:]
    min_after_peak = after_peak.min()
    min_after_peak_index = after_peak.idxmin()
    
    drawdown_after = max_value - min_after_peak
    drawdown_percent_after = (drawdown_after / max_value) * 100
    
    print(f"최고점 이후 최저점: {min_after_peak:.2f} (인덱스: {min_after_peak_index})")
    print(f"최고점 이후 MMD: {drawdown_percent_after:.2f}%")

# 전체 기간 동안의 최고점 대비 각 시점의 하락률 계산
peak_so_far = broker_values.iloc[0]
drawdowns = []
peak_values = []

for value in broker_values:
    if value > peak_so_far:
        peak_so_far = value
    
    drawdown_percent = ((peak_so_far - value) / peak_so_far) * 100
    drawdowns.append(drawdown_percent)
    peak_values.append(peak_so_far)

# 최대 하락률과 해당 시점 찾기
max_dd = max(drawdowns)
max_dd_index = drawdowns.index(max_dd)

print(f"\n=== 전체 기간 MMD 분석 ===")
print(f"전체 기간 최대 하락률: {max_dd:.2f}%")
print(f"최대 하락률 발생 시점: 인덱스 {max_dd_index}")
print(f"최대 하락률 발생 시점의 자본: {broker_values.iloc[max_dd_index]:.2f}")
print(f"최대 하락률 발생 시점의 최고점: {peak_values[max_dd_index]:.2f}")

# 23.81%와 비교
if abs(max_dd - 23.81) < 0.1:
    print("✅ MMD 23.81%가 정확합니다!")
else:
    print(f"❌ MMD 23.81%가 부정확합니다. 실제 MMD: {max_dd:.2f}%")
    print(f"차이: {abs(max_dd - 23.81):.2f}%")

# 하락률이 20% 이상인 구간 찾기
high_drawdown_periods = []
for i, dd in enumerate(drawdowns):
    if dd >= 20:
        high_drawdown_periods.append((i, dd, broker_values.iloc[i]))

if high_drawdown_periods:
    print(f"\n=== 20% 이상 하락 구간 ===")
    for period in high_drawdown_periods:
        print(f"인덱스 {period[0]}: {period[1]:.2f}% 하락 (자본: {period[2]:.2f})")
else:
    print(f"\n=== 20% 이상 하락 구간 없음 ===")

# 하락률이 15% 이상인 구간 찾기
high_drawdown_periods_15 = []
for i, dd in enumerate(drawdowns):
    if dd >= 15:
        high_drawdown_periods_15.append((i, dd, broker_values.iloc[i]))

if high_drawdown_periods_15:
    print(f"\n=== 15% 이상 하락 구간 ===")
    for period in high_drawdown_periods_15:
        print(f"인덱스 {period[0]}: {period[1]:.2f}% 하락 (자본: {period[2]:.2f})") 