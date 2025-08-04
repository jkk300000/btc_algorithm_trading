import pandas as pd
import numpy as np

# 로그 파일 읽기
df = pd.read_csv('trade_logs/trade_logs_20250801_155952.csv')

# broker_value 컬럼만 추출
broker_values = df['broker_value'].dropna()

print(f"총 데이터 수: {len(broker_values)}")
print(f"초기 자본: {broker_values.iloc[0]:.2f}")
print(f"최종 자본: {broker_values.iloc[-1]:.2f}")

# MMD 계산
peak = broker_values.iloc[0]
max_drawdown = 0
max_drawdown_percent = 0

for value in broker_values:
    if value > peak:
        peak = value
    
    drawdown = peak - value
    drawdown_percent = (drawdown / peak) * 100
    
    if drawdown_percent > max_drawdown_percent:
        max_drawdown_percent = drawdown_percent
        max_drawdown = drawdown

print(f"\n=== MMD 분석 결과 ===")
print(f"최고점: {peak:.2f}")
print(f"최대 손실: {max_drawdown:.2f}")
print(f"MMD: {max_drawdown_percent:.2f}%")

# 최저점 찾기
min_value = broker_values.min()
min_index = broker_values.idxmin()
print(f"최저점: {min_value:.2f} (인덱스: {min_index})")

# 최고점 찾기
max_value = broker_values.max()
max_index = broker_values.idxmax()
print(f"최고점: {max_value:.2f} (인덱스: {max_index})")

# 최고점 이후의 최저점 찾기
if max_index < min_index:
    print("✅ 최고점 이후 최저점이 발생했습니다.")
else:
    print("❌ 최고점 이후 최저점이 발생하지 않았습니다.")
    # 최고점 이후의 최저점 찾기
    after_peak = broker_values[max_index:]
    min_after_peak = after_peak.min()
    min_after_peak_index = after_peak.idxmin()
    print(f"최고점 이후 최저점: {min_after_peak:.2f} (인덱스: {min_after_peak_index})")
    
    # 최고점 이후의 MMD 계산
    peak_after = broker_values[max_index]
    drawdown_after = peak_after - min_after_peak
    drawdown_percent_after = (drawdown_after / peak_after) * 100
    print(f"최고점 이후 MMD: {drawdown_percent_after:.2f}%")

# 전체 기간 동안의 최고점 대비 각 시점의 하락률 계산
peak_so_far = broker_values.iloc[0]
drawdowns = []

for value in broker_values:
    if value > peak_so_far:
        peak_so_far = value
    
    drawdown_percent = ((peak_so_far - value) / peak_so_far) * 100
    drawdowns.append(drawdown_percent)

# 최대 하락률 확인
max_dd = max(drawdowns)
print(f"\n전체 기간 최대 하락률: {max_dd:.2f}%")

# 23.81%와 비교
if abs(max_dd - 23.81) < 0.1:
    print("✅ MMD 23.81%가 정확합니다!")
else:
    print(f"❌ MMD 23.81%가 부정확합니다. 실제 MMD: {max_dd:.2f}%") 