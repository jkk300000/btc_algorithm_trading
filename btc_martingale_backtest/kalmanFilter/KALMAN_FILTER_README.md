# 비트코인 무기한 선물용 적응형 칼만 필터

## 개요

이 모듈은 비트코인 무기한 선물의 높은 변동성을 고려한 적응형 칼만 필터를 제공합니다. 극한 변동성 구간에서도 중요한 가격 신호를 보존하면서, 일반적인 구간에서는 효과적인 노이즈 제거를 수행합니다.

## 주요 특징

### 🎯 적응형 필터링
- **변동성 기반 파라미터 조정**: 시장 상황에 따라 자동으로 필터 강도 조정
- **극한 변동성 보존**: 중요한 가격 신호 손실 방지
- **다중 스케일 분석**: 다양한 시간대 고려

### 📊 성능 최적화
- **파라미터 자동 최적화**: 데이터에 맞는 최적 파라미터 자동 탐색
- **성능 검증**: 변동성 보존과 노이즈 제거 균형 확인
- **실시간 모니터링**: 필터링 과정 실시간 추적

## 설치 및 의존성

```bash
pip install numpy pandas scipy scikit-learn
```

## 사용법

### 1. 기본 사용법

```python
from feature_engineering import add_features

# 칼만 필터 없이 특성 계산
df_original = add_features(
    input_path='btc_data.csv',
    use_kalman_filter=False
)

# 칼만 필터 적용 (기본 파라미터)
df_filtered = add_features(
    input_path='btc_data.csv',
    use_kalman_filter=True
)
```

### 2. 고급 사용법

```python
# 칼만 필터 파라미터 최적화
df_optimized = add_features(
    input_path='btc_data.csv',
    use_kalman_filter=True,
    optimize_kalman=True  # 파라미터 자동 최적화
)

# 커스텀 파라미터 적용
custom_params = {
    'base_Q': 0.005,           # 프로세스 노이즈
    'base_R': 2.0,             # 측정 노이즈
    'volatility_threshold': 0.08,  # 극한 변동성 임계값
    'preservation_factor': 0.75,   # 신호 보존 비율
    'volatility_window': 20,       # 변동성 계산 윈도우
    'adaptive_factor': 5.0         # 적응형 조정 계수
}

df_custom = add_features(
    input_path='btc_data.csv',
    use_kalman_filter=True,
    kalman_params=custom_params
)
```

### 3. 직접 칼만 필터 사용

```python
from kalman_filter_btc import apply_btc_kalman_filtering, validate_kalman_performance

# 데이터 로드
df = pd.read_csv('btc_data.csv')

# 칼만 필터 적용
df_filtered = apply_btc_kalman_filtering(df)

# 성능 검증
performance = validate_kalman_performance(df, df_filtered)
print(f"신호 보존율: {performance['preservation_ratio']:.2%}")
print(f"노이즈 제거 효과: {performance['noise_reduction']:.2%}")
```

## 파라미터 설명

### 기본 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `base_Q` | 0.005 | 프로세스 노이즈 (시스템 불확실성) |
| `base_R` | 2.0 | 측정 노이즈 (관측 불확실성) |
| `volatility_threshold` | 0.08 | 극한 변동성 임계값 (8%) |
| `preservation_factor` | 0.75 | 원본 신호 보존 비율 (75%) |
| `volatility_window` | 20 | 변동성 계산 윈도우 (20분) |
| `adaptive_factor` | 5.0 | 적응형 조정 계수 |

### 시장 상황별 파라미터

```python
from kalman_filter_btc import get_dynamic_kalman_params

# 상승장 파라미터
bull_params = get_dynamic_kalman_params('bull_market')

# 하락장 파라미터
bear_params = get_dynamic_kalman_params('bear_market')

# 횡보장 파라미터
sideways_params = get_dynamic_kalman_params('sideways')
```

## 성능 지표

### 1. 신호 보존율 (Preservation Ratio)
- 극한 변동성 구간에서 원본 신호가 보존되는 비율
- 높을수록 중요한 가격 움직임이 유지됨

### 2. 노이즈 제거 효과 (Noise Reduction)
- 전체적인 노이즈 제거 효과
- 높을수록 더 부드러운 가격 곡선

### 3. 극한 변동성 구간 수
- 필터링 후에도 극한 변동성이 감지되는 구간 수
- 적절한 균형이 중요

## 테스트 및 검증

### 통합 테스트 실행

```bash
python test_kalman_integration.py
```

### 테스트 결과 예시

```
📊 칼만 필터 성능 비교 결과
============================================================
📈 평균 변동성:
   원본 데이터: 0.0234
   칼만 필터 (기본): 0.0187
   칼만 필터 (최적화): 0.0192

🎯 극한 변동성 구간 (>10%):
   원본 데이터: 45개
   칼만 필터 (기본): 38개
   칼만 필터 (최적화): 42개

📉 노이즈 제거 효과:
   칼만 필터 (기본): 20.09%
   칼만 필터 (최적화): 17.95%

🔄 신호 보존율:
   칼만 필터 (기본): 79.91%
   칼만 필터 (최적화): 82.05%
============================================================
```

## 주의사항

### 1. 데이터 요구사항
- OHLCV 데이터가 필요합니다
- 최소 20개 이상의 데이터 포인트가 필요합니다
- 결측치가 없는 깨끗한 데이터를 사용하세요

### 2. 성능 고려사항
- 대용량 데이터의 경우 처리 시간이 오래 걸릴 수 있습니다
- 파라미터 최적화는 시간이 많이 소요됩니다
- 메모리 사용량을 고려하여 청크 단위 처리를 고려하세요

### 3. 파라미터 튜닝
- 너무 강한 필터링은 중요한 신호를 손실시킬 수 있습니다
- 너무 약한 필터링은 노이즈 제거 효과가 미미합니다
- 시장 상황에 따라 파라미터를 조정하세요

## 고급 기능

### 1. 다중 스케일 필터링

```python
from kalman_filter_btc import MultiScaleKalmanFilter

# 다중 스케일 필터 적용
multi_scale_filter = MultiScaleKalmanFilter(scales=[1, 5, 15])
df_multi_scale = multi_scale_filter.multi_scale_filtering(df)
```

### 2. 파라미터 최적화

```python
from kalman_filter_btc import optimize_btc_kalman_parameters

# 파라미터 최적화
optimal_params, best_score = optimize_btc_kalman_parameters(
    df, 
    max_combinations=50
)
print(f"최적 파라미터: {optimal_params}")
print(f"최적 점수: {best_score:.4f}")
```

### 3. 실시간 필터링

```python
from kalman_filter_btc import AdaptiveBTCKalmanFilter

# 실시간 필터 초기화
filter = AdaptiveBTCKalmanFilter()

# 실시간 데이터 처리
for new_data in real_time_data:
    filtered_data = filter.filter_ohlcv(new_data)
    # 필터링된 데이터 사용
```

## 문제 해결

### 1. 메모리 부족 오류
```python
# 청크 단위 처리 사용
df_chunked = add_features_chunked(
    input_path='large_data.csv',
    use_kalman_filter=True
)
```

### 2. 성능 저하
```python
# 파라미터 최적화 비활성화
df_fast = add_features(
    input_path='data.csv',
    use_kalman_filter=True,
    optimize_kalman=False  # 최적화 비활성화
)
```

### 3. 과도한 필터링
```python
# 보존 비율 증가
conservative_params = {
    'preservation_factor': 0.9,  # 90% 보존
    'volatility_threshold': 0.05  # 5% 임계값
}
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

## 연락처

문의사항이 있으시면 이슈를 생성해 주세요. 