import numpy as np
import pandas as pd
from scipy.linalg import inv
import logging
from typing import Dict, Tuple, Optional
from joblib import Parallel, delayed
import multiprocessing



# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# GPU 가속을 위한 CuPy 지원 (선택적)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("✅ CuPy GPU 가속 사용 가능")
except ImportError:
    cp = np  # CuPy가 없으면 NumPy 사용
    CUPY_AVAILABLE = False
    logger.info("⚠️ CuPy 없음 - CPU 모드로 실행")




class KalmanFilter:
    """기본 칼만 필터 클래스"""
    
    def __init__(self, initial_state: float, initial_P: float = 1.0, 
                 Q: float = 0.01, R: float = 1.0):
        """
        칼만 필터 초기화
        
        Args:
            initial_state: 초기 상태 (가격)
            initial_P: 초기 상태 불확실성
            Q: 프로세스 노이즈 (시스템 노이즈)
            R: 측정 노이즈 (관측 노이즈)
        """
        self.x = initial_state  # 상태 추정값
        self.P = initial_P      # 상태 불확실성
        self.Q = Q              # 프로세스 노이즈
        self.R = R              # 측정 노이즈
        
        # 상태 전이 행렬 (1차원 가격 모델)
        self.F = 1.0            # 상태 전이
        self.H = 1.0            # 관측 행렬
        
    def predict(self):
        """예측 단계"""
        # 상태 예측
        self.x = self.F * self.x
        # 불확실성 예측
        self.P = self.F * self.P * self.F + self.Q
        
    def update(self, measurement: float) -> float:
        """업데이트 단계"""
        # 칼만 게인 계산
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        
        # 상태 업데이트
        self.x = self.x + K * (measurement - self.H * self.x)
        
        # 불확실성 업데이트
        self.P = (1 - K * self.H) * self.P
        
        return self.x


class AdaptiveBTCKalmanFilter:
    """비트코인 무기한 선물용 적응형 칼만 필터"""
    
    def __init__(self, base_Q: float = 0.005, base_R: float = 2.0, 
                 volatility_window: int = 20, volatility_threshold: float = 0.08,
                 preservation_factor: float = 0.75, adaptive_factor: float = 5.0,
                 use_gpu: bool = False, n_jobs: int = 1):
        """
        비트코인용 적응형 칼만 필터 초기화
        
        Args:
            base_Q: 기본 프로세스 노이즈
            base_R: 기본 측정 노이즈
            volatility_window: 변동성 계산 윈도우
            volatility_threshold: 극한 변동성 임계값 (8%)
            preservation_factor: 원본 신호 보존 비율 (75%)
            adaptive_factor: 적응형 조정 계수
            use_gpu: GPU 가속 사용 여부 (CuPy 필요)
            n_jobs: 병렬 처리 프로세스 수 (-1: 모든 CPU 사용)
        """
        self.base_Q = base_Q
        self.base_R = base_R
        self.volatility_window = volatility_window
        self.volatility_threshold = volatility_threshold
        self.preservation_factor = preservation_factor
        self.adaptive_factor = adaptive_factor
        self.filters = {}
        self.volatility_history = []
        
        # 성능 최적화 설정
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        
        # GPU 사용 여부 로깅
        if self.use_gpu:
            logger.info(f"🚀 GPU 가속 모드 활성화 (CuPy)")
        if self.n_jobs > 1:
            logger.info(f"⚡ 병렬 처리 모드 활성화 ({self.n_jobs} 프로세스)")
    
    def _get_array_lib(self):
        """GPU/CPU 배열 라이브러리 선택"""
        return cp if self.use_gpu else np
    
    def _to_gpu(self, data):
        """GPU로 데이터 이동 (GPU 모드일 때만)"""
        if self.use_gpu and isinstance(data, np.ndarray):
            return cp.asarray(data)
        return data
    
    def _to_cpu(self, data):
        """CPU로 데이터 이동 (결과 반환용)"""
        if self.use_gpu and hasattr(data, 'get'):
            return data.get()
        return data
        
    def calculate_adaptive_parameters(self, price_series: pd.Series, 
                                    current_idx: int) -> Tuple[float, float]:
        """변동성 기반 적응형 파라미터 계산"""
        
        if current_idx < self.volatility_window:
            # 초기 구간: 기본 파라미터 사용
            return self.base_Q, self.base_R
        
        # 최근 변동성 계산
        recent_prices = price_series.iloc[current_idx-self.volatility_window:current_idx+1]
        returns = np.diff(np.log(recent_prices))
        current_volatility = np.std(returns, ddof=1)
        
        # 변동성 정규화 (0~1 범위)
        normalized_vol = min(current_volatility / 0.1, 1.0)  # 10% 변동성을 최대값으로
        
        # 적응형 파라미터 계산
        adaptive_Q = self.base_Q * (1 + normalized_vol * self.adaptive_factor)
        adaptive_R = self.base_R * (1 + normalized_vol * (self.adaptive_factor * 0.5))
        
        return adaptive_Q, adaptive_R
    
    def detect_extreme_volatility(self, price_series: pd.Series, 
                                 window: int = 5) -> bool:
        """극한 변동성 구간 감지"""
        if len(price_series) < window:
            return False
        
        recent_prices = price_series.iloc[-window:]
        returns = np.diff(np.log(recent_prices))
        volatility = np.std(returns, ddof=1)
        
        return volatility > self.volatility_threshold
    
    def _filter_single_column(self, df: pd.DataFrame, col: str) -> Tuple[list, list]:
        """단일 컬럼 필터링 (병렬 처리용)"""
        xp = self._get_array_lib()  # GPU/CPU 배열 라이브러리
        
        # 필터 초기화
        if col not in self.filters:
            self.filters[col] = KalmanFilter(
                initial_state=df[col].iloc[0],
                Q=self.base_Q,
                R=self.base_R
            )
        
        filtered_values = []
        volatility_tracker = []
        
        # GPU 가속 시 데이터를 GPU 메모리로 이동
        col_values = self._to_gpu(df[col].values) if self.use_gpu else df[col].values
        total_rows = len(col_values)
        
        for i, value in enumerate(col_values):
            # 진행률 표시 (매 10000행마다)
            if i > 0 and i % 10000 == 0:
                progress = (i / total_rows) * 100
                logger.info(f"  📈 {col} 컬럼 진행률: {progress:.1f}% ({i:,}/{total_rows:,})")
            
            # 극한 변동성 구간 감지 (CPU에서 수행)
            cpu_value = self._to_cpu(value) if self.use_gpu else value
            price_history = df[col].iloc[:i+1]
            is_extreme_vol = self.detect_extreme_volatility(price_history)
            
            # 적응형 파라미터 기본값 설정
            adaptive_Q = self.base_Q
            adaptive_R = self.base_R
            
            if is_extreme_vol:
                # 극한 변동성: 원본 신호 보존
                if i > 0:
                    filtered_value = (
                        self.preservation_factor * cpu_value + 
                        (1 - self.preservation_factor) * self.filters[col].x
                    )
                else:
                    filtered_value = cpu_value
            else:
                # 일반 변동성: 적응형 칼만 필터 적용
                adaptive_Q, adaptive_R = self.calculate_adaptive_parameters(df[col], i)
                
                # 필터 파라미터 업데이트
                self.filters[col].Q = adaptive_Q
                self.filters[col].R = adaptive_R
                
                # 칼만 필터 적용
                self.filters[col].predict()
                filtered_value = self.filters[col].update(cpu_value)
            
            filtered_values.append(filtered_value)
            volatility_tracker.append(adaptive_Q)
        
        return filtered_values, volatility_tracker
    
    def filter_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV 데이터 적응형 필터링 (GPU 가속 및 병렬 처리 지원)"""
        filtered_df = df.copy()
        
        logger.info("🚀 적응형 칼만 필터 (1단계) 시작")
        logger.info(f"📊 데이터 크기: {len(df):,} 행")
        
        columns = ['open', 'high', 'low', 'close']
        
        if self.n_jobs > 1 and len(df) > 10000:  # 대용량 데이터에서만 병렬 처리
            logger.info(f"⚡ 병렬 처리 모드로 {len(columns)}개 컬럼 동시 처리")
            
            # 병렬 처리로 모든 컬럼 동시 처리
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._filter_single_column)(df, col) for col in columns
            )
            
            # 결과 병합
            for i, col in enumerate(columns):
                filtered_values, volatility_tracker = results[i]
                filtered_df[col] = filtered_values
                filtered_df[f'{col}_volatility'] = volatility_tracker
                logger.info(f"✅ {col} 컬럼 처리 완료")
                
        else:
            # 순차 처리 (기존 방식)
            for col_idx, col in enumerate(columns):
                logger.info(f"🔄 {col} 컬럼 처리 중... ({col_idx + 1}/4)")
                filtered_values, volatility_tracker = self._filter_single_column(df, col)
                filtered_df[col] = filtered_values
                filtered_df[f'{col}_volatility'] = volatility_tracker
                logger.info(f"✅ {col} 컬럼 처리 완료")
        
        # 거래량 필터링 (로그 스케일)
        if 'volume' in df.columns:
            filtered_df['volume'] = self.filter_volume(df['volume'])
        
        logger.info("✅ 적응형 칼만 필터 (1단계) 완료")
        return filtered_df
    
    def filter_volume(self, volume_series: pd.Series) -> pd.Series:
        """거래량 필터링 (로그 스케일)"""
        log_volume = np.log(volume_series + 1)  # +1 to avoid log(0)
        
        if 'volume' not in self.filters:
            self.filters['volume'] = KalmanFilter(
                initial_state=log_volume.iloc[0],
                Q=self.base_Q * 0.1,  # 거래량은 더 작은 노이즈
                R=self.base_R * 2.0   # 거래량은 더 큰 측정 노이즈
            )
        
        filtered_log_volume = []
        for value in log_volume.values:
            self.filters['volume'].predict()
            filtered_value = self.filters['volume'].update(value)
            filtered_log_volume.append(filtered_value)
        
        # 로그 역변환
        return np.exp(filtered_log_volume) - 1


class MultiScaleKalmanFilter:
    """다중 스케일 칼만 필터"""
    
    def __init__(self, scales: list = [1, 5, 15]):
        """
        다중 스케일 칼만 필터 초기화
        
        Args:
            scales: 분석 스케일 (1분, 5분, 15분)
        """
        self.scales = scales
        self.filters = {}
        
    def multi_scale_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """다중 스케일 필터링"""
        filtered_df = df.copy()
        
        logger.info("🔢 2단계: 다중 스케일 칼만 필터 적용 시작")
        logger.info(f"📐 적용 스케일: {self.scales}")
        
        for col in ['open', 'high', 'low', 'close']:
            # 각 스케일별 필터 초기화
            scale_filters = {}
            for scale in self.scales:
                scale_filters[scale] = KalmanFilter(
                    initial_state=df[col].iloc[0],
                    Q=0.01 * scale,  # 스케일에 비례하는 노이즈
                    R=1.0 / scale    # 스케일에 반비례하는 측정 노이즈
                )
            
            filtered_values = []
            
            for i, value in enumerate(df[col].values):
                # 각 스케일별 필터링
                scale_predictions = {}
                for scale in self.scales:
                    if i >= scale:
                        # 스케일별 데이터 추출
                        scale_data = df[col].iloc[i-scale:i+1]
                        scale_filter = scale_filters[scale]
                        
                        # 스케일별 예측
                        scale_filter.predict()
                        scale_predictions[scale] = scale_filter.update(value)
                    else:
                        scale_predictions[scale] = value
                
                # 가중 평균으로 최종 예측
                weights = [1/scale for scale in self.scales]
                total_weight = sum(weights)
                final_prediction = sum(
                    pred * weight for pred, weight in zip(scale_predictions.values(), weights)
                ) / total_weight
                
                filtered_values.append(final_prediction)
            
            filtered_df[col] = filtered_values
        
        logger.info("✅ 2단계: 다중 스케일 칼만 필터 적용 완료")
        return filtered_df


def create_btc_kalman_filter(params: Optional[Dict] = None, use_gpu: bool = False, n_jobs: int = 1) -> AdaptiveBTCKalmanFilter:
    """비트코인 전용 칼만 필터 생성"""
    
    # 기본 파라미터
    default_params = {
        'base_Q': 0.005,           # 작은 프로세스 노이즈
        'base_R': 2.0,             # 큰 측정 노이즈
        'volatility_threshold': 0.08,  # 8% 변동성 임계값
        'preservation_factor': 0.75,   # 75% 원본 신호 보존
        'volatility_window': 20,       # 20분 변동성 윈도우
        'adaptive_factor': 5.0,        # 적응형 조정 계수
        'use_gpu': use_gpu,           # GPU 가속 사용
        'n_jobs': n_jobs              # 병렬 처리 프로세스 수
    }
    
    if params:
        default_params.update(params)
    
    return AdaptiveBTCKalmanFilter(**default_params)


def apply_btc_kalman_filtering(df: pd.DataFrame, 
                              use_multi_scale: bool = False,
                              params: Optional[Dict] = None,
                              use_gpu: bool = False,
                              n_jobs: int = 1) -> pd.DataFrame:
    """비트코인 데이터 칼만 필터 적용"""
    
    logger.info("🎯 비트코인 칼만 필터 파이프라인 시작")
    
    # 1단계: 적응형 칼만 필터 적용
    logger.info("🔄 1단계: 적응형 칼만 필터 적용")
    adaptive_filter = create_btc_kalman_filter(params, use_gpu=use_gpu, n_jobs=n_jobs)
    df_filtered = adaptive_filter.filter_ohlcv(df)
    logger.info("✅ 1단계: 적응형 칼만 필터 완료")
    
    # 2단계: 다중 스케일 필터링 (선택적)
    if use_multi_scale:
        multi_scale_filter = MultiScaleKalmanFilter()
        df_filtered = multi_scale_filter.multi_scale_filtering(df_filtered)
    
    logger.info("🏁 비트코인 칼만 필터 파이프라인 완료")
    return df_filtered


def validate_kalman_performance(df_original: pd.DataFrame, 
                               df_filtered: pd.DataFrame) -> Dict:
    """칼만 필터 성능 검증"""
    
    logger.info("📊 칼만 필터 성능 검증 시작")
    
    # 변동성 보존 확인
    original_volatility = df_original['close'].pct_change().rolling(20).std()
    filtered_volatility = df_filtered['close'].pct_change().rolling(20).std()
    
    # 극한 변동성 구간 확인
    extreme_vol_periods = original_volatility > 0.1  # 10% 이상 변동성
    
    # 극한 구간에서의 신호 보존율
    if extreme_vol_periods.sum() > 0:
        preservation_ratio = (
            filtered_volatility[extreme_vol_periods] / 
            original_volatility[extreme_vol_periods]
        ).mean()
    else:
        preservation_ratio = 1.0
    
    # 노이즈 제거 효과
    noise_reduction = (
        original_volatility.mean() - filtered_volatility.mean()
    ) / original_volatility.mean()
    
    # 결과 출력
    logger.info(f"극한 변동성 구간 신호 보존율: {preservation_ratio:.2%}")
    logger.info(f"노이즈 제거 효과: {noise_reduction:.2%}")
    
    return {
        'preservation_ratio': preservation_ratio,
        'noise_reduction': noise_reduction,
        'original_volatility_mean': original_volatility.mean(),
        'filtered_volatility_mean': filtered_volatility.mean(),
        'extreme_vol_periods_count': extreme_vol_periods.sum()
    }


def optimize_btc_kalman_parameters(df: pd.DataFrame, 
                                  max_combinations: int = 50) -> Tuple[Dict, float]:
    """비트코인 칼만 필터 파라미터 최적화"""
    
    logger.info("🔧 칼만 필터 파라미터 최적화 시작")
    
    # 파라미터 그리드 (제한된 조합으로 최적화)
    param_grid = {
        'base_Q': [0.001, 0.005, 0.01, 0.02],
        'base_R': [0.5, 1.0, 2.0, 5.0],
        'volatility_threshold': [0.05, 0.08, 0.1, 0.15],
        'preservation_factor': [0.6, 0.7, 0.8, 0.9]
    }
    
    best_score = 0
    best_params = None
    combinations_tested = 0
    
    # 제한된 조합으로 테스트
    for base_Q in param_grid['base_Q']:
        for base_R in param_grid['base_R']:
            for threshold in param_grid['volatility_threshold']:
                for factor in param_grid['preservation_factor']:
                    
                    if combinations_tested >= max_combinations:
                        break
                    
                    # 파라미터로 필터링
                    filter_params = {
                        'base_Q': base_Q,
                        'base_R': base_R,
                        'volatility_threshold': threshold,
                        'preservation_factor': factor
                    }
                    
                    try:
                        df_filtered = apply_btc_kalman_filtering(df, params=filter_params)
                        
                        # 성능 평가
                        performance = validate_kalman_performance(df, df_filtered)
                        
                        # 종합 점수 (보존율 70% + 노이즈 제거 30%)
                        score = (
                            performance['preservation_ratio'] * 0.7 + 
                            performance['noise_reduction'] * 0.3
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_params = filter_params
                            
                        combinations_tested += 1
                        
                    except Exception as e:
                        logger.warning(f"파라미터 조합 실패: {filter_params}, 오류: {e}")
                        continue
    
    logger.info(f"✅ 파라미터 최적화 완료 (테스트 조합: {combinations_tested})")
    logger.info(f"최적 파라미터: {best_params}")
    logger.info(f"최적 점수: {best_score:.4f}")
    
    return best_params, best_score


def get_dynamic_kalman_params(market_condition: str) -> Dict:
    """시장 상황별 동적 파라미터"""
    
    base_params = {
        'base_Q': 0.005,
        'base_R': 2.0,
        'volatility_threshold': 0.08,
        'preservation_factor': 0.75,
        'volatility_window': 20,
        'adaptive_factor': 5.0
    }
    
    if market_condition == 'bull_market':
        # 상승장: 더 적극적인 필터링
        base_params['base_Q'] *= 0.5
        base_params['preservation_factor'] *= 0.9
        
    elif market_condition == 'bear_market':
        # 하락장: 보수적인 필터링
        base_params['base_Q'] *= 2.0
        base_params['preservation_factor'] *= 1.1
        
    elif market_condition == 'sideways':
        # 횡보장: 균형잡힌 필터링
        pass  # 기본 파라미터 유지
        
    return base_params


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 비트코인 칼만 필터 모듈 테스트")
    
    # 샘플 데이터 생성
    dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    # 비트코인 가격 시뮬레이션 (높은 변동성)
    base_price = 50000
    returns = np.random.normal(0, 0.02, len(dates))  # 2% 일일 변동성
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 극한 변동성 구간 추가
    extreme_periods = [200, 400, 600, 800]
    for period in extreme_periods:
        prices[period:period+10] *= np.random.uniform(0.9, 1.1, 10)
    
    test_df = pd.DataFrame({
        'open': prices * np.random.uniform(0.999, 1.001, len(prices)),
        'high': prices * np.random.uniform(1.001, 1.005, len(prices)),
        'low': prices * np.random.uniform(0.995, 0.999, len(prices)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, len(prices))
    }, index=dates)
    
    # 칼만 필터 적용
    filtered_df = apply_btc_kalman_filtering(test_df)
    
    # 성능 검증
    performance = validate_kalman_performance(test_df, filtered_df)
    
    print("✅ 테스트 완료") 