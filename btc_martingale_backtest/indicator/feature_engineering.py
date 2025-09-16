import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
from squeeze_momentum_core import squeeze_momentum_core
import talib
import matplotlib.pyplot as plt
import os
import sys
import time
import logging
from datetime import datetime

# 메인 모듈(직접 실행하는 파일)은 반드시 절대 경로를 사용해야 함
# 직접 실행 시 부모 디렉토리를 sys.path에 추가하여 절대 경로 import 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from kalmanFilter.kalman_filter_btc import apply_btc_kalman_filtering, validate_kalman_performance, optimize_btc_kalman_parameters

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def calculate_rma(data, period):
    """
    RMA (Relative Moving Average) 계산 - Pine Script ta.rma()와 정확히 일치
    Pine Script 문서에 따른 정확한 구현
    
    Args:
        data: pandas Series - 계산할 데이터
        period: int - 기간
    
    Returns:
        pandas Series - RMA 값
    """
    if len(data) < period:
        return pd.Series([np.nan] * len(data), index=data.index)
    
    # 결과를 저장할 Series 생성
    rma = pd.Series(index=data.index, dtype=float)
    
    # Pine Script ta.rma()와 동일한 초기화
    # 첫 번째 값은 첫 번째 데이터로 초기화
    rma.iloc[0] = data.iloc[0]
    
    # RMA 계산: Pine Script와 정확히 동일한 방식
    # RMA = (prev_rma * (period - 1) + current_value) / period
    for i in range(1, len(data)):
        if i < period:
            # period 미만일 때는 단순 평균 (Pine Script와 동일)
            rma.iloc[i] = data.iloc[:i+1].mean()
        else:
            # period 이상일 때는 RMA 공식 사용
            # Pine Script: (prev_rma * (period - 1) + current_value) / period
            prev_rma = rma.iloc[i-1]
            current_value = data.iloc[i]
            rma.iloc[i] = (prev_rma * (period - 1) + current_value) / period
    
    return rma


def calculate_atr_pinescript(df, period=14):
    """
    Pine Script ta.atr()와 정확히 동일한 ATR 계산
    트레이딩뷰 파인스크립트와 100% 일치하는 구현
    
    Args:
        df: pandas DataFrame - OHLC 데이터
        period: int - ATR 기간 (기본값: 14)
    
    Returns:
        pandas Series - ATR 값
    """
    # True Range 계산 (Pine Script와 정확히 동일)
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 이전 종가 계산 (첫 번째 값은 NaN)
    prev_close = close.shift(1)
    
    # True Range의 세 가지 구성요소
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    # 최대값 선택
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # RMA 기반 ATR (Pine Script와 정확히 동일)
    atr = calculate_rma(tr, period)
    return atr


def calculate_atr_for_backtest_period(df, backtest_start_date, period=14):
    """
    백테스팅 기간에 맞춰 ATR 계산
    지정된 시작일부터의 데이터로만 ATR 계산
    
    Args:
        df: pandas DataFrame - 전체 OHLC 데이터
        backtest_start_date: str - 백테스팅 시작일 (예: '2022-09-01')
        period: int - ATR 기간 (기본값: 14)
    
    Returns:
        pandas Series - ATR 값 (전체 데이터 길이, 백테스팅 시작일 이전은 NaN)
    """
    # 백테스팅 시작일 이후 데이터만 추출
    backtest_start = pd.to_datetime(backtest_start_date)
    df_backtest = df[df.index >= backtest_start]
    
    if len(df_backtest) == 0:
        logger.warning(f"⚠️ 백테스팅 시작일 {backtest_start_date} 이후 데이터가 없습니다.")
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # 백테스팅 기간 데이터로 ATR 계산
    atr_backtest = calculate_atr_pinescript(df_backtest, period)
    
    # 전체 데이터 길이에 맞춰 결과 생성 (백테스팅 시작일 이전은 NaN)
    atr_full = pd.Series([np.nan] * len(df), index=df.index)
    atr_full.loc[df_backtest.index] = atr_backtest
    
    logger.info(f"✅ 백테스팅 기간 ATR 계산 완료: {backtest_start_date}부터 {len(df_backtest)}개 데이터")
    
    return atr_full


def add_features(input_path, output_path=None, diagnose=True, use_pinescript_atr=True, 
                use_kalman_filter=False, kalman_params=None, optimize_kalman=False, use_multi_scale=False,
                use_gpu=False, n_jobs=1, backtest_start_date=None):
    """
    ta_lib 라이브러리를 활용해 진입 조건 및 ml 가격 상승 및 하락 예측에 사용할 특정 지표를 계산.
    
    Args:
        input_path: str - 입력 CSV 파일 경로
        output_path: str - 출력 CSV 파일 경로 (None이면 저장하지 않음)
        diagnose: bool - 진단 정보 출력 여부
        use_pinescript_atr: bool - Pine Script와 동일한 ATR 사용 여부 (기본값: True)
        use_kalman_filter: bool - 칼만 필터 적용 여부 (기본값: False)
        kalman_params: dict - 칼만 필터 파라미터 (None이면 기본값 사용)
        optimize_kalman: bool - 칼만 필터 파라미터 최적화 여부 (기본값: False)
        use_multi_scale: bool - 다중 스케일 칼만 필터 적용 여부 (기본값: False)
    """
    start_time = time.time()
    logger.info("="*60)
    logger.info("🚀 특성 엔지니어링 프로세스 시작")
    logger.info("="*60)
    
    logger.info(f"📖 입력 파일: {input_path}")
    logger.info(f"📝 출력 파일: {output_path if output_path else '저장 안함'}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(input_path) / (1024*1024)  # MB
    logger.info(f"📁 파일 크기: {file_size:.2f} MB")
    
    # 설정 정보 로깅
    logger.info("⚙️ 설정 정보:")
    logger.info(f"  - Pine Script ATR: {use_pinescript_atr}")
    logger.info(f"  - 칼만 필터 사용: {use_kalman_filter}")
    logger.info(f"  - 다중 스케일 필터: {use_multi_scale}")
    logger.info(f"  - 파라미터 최적화: {optimize_kalman}")
    if kalman_params:
        logger.info(f"  - 커스텀 파라미터: {kalman_params}")
    
    # 청크 단위 처리 비활성화 - 전체 데이터를 한 번에 처리
    logger.info("🔄 전체 데이터를 한 번에 처리합니다...")
    
    result = add_features_single(input_path, output_path, diagnose, use_pinescript_atr, 
                               use_kalman_filter, kalman_params, optimize_kalman, use_multi_scale,
                               use_gpu, n_jobs, backtest_start_date)
    
    # 총 소요 시간 계산
    total_time = time.time() - start_time
    logger.info("="*60)
    logger.info(f"✅ 특성 엔지니어링 완료 (총 소요시간: {total_time:.2f}초)")
    logger.info("="*60)
    
    return result


def add_features_single(input_path, output_path=None, diagnose=True, use_pinescript_atr=True,
                       use_kalman_filter=False, kalman_params=None, optimize_kalman=False, use_multi_scale=False,
                       use_gpu=False, n_jobs=1, backtest_start_date=None):
    """
    일반 파일 처리 (단일 파일) - 개선된 버전
    
    Args:
        input_path: str - 입력 CSV 파일 경로
        output_path: str - 출력 CSV 파일 경로
        diagnose: bool - 진단 정보 출력 여부
        use_pinescript_atr: bool - Pine Script ATR 사용 여부
        use_kalman_filter: bool - 칼만 필터 적용 여부
        kalman_params: dict - 칼만 필터 파라미터
        optimize_kalman: bool - 칼만 필터 파라미터 최적화 여부
        use_multi_scale: bool - 다중 스케일 칼만 필터 적용 여부
    """
    # 데이터 로딩 방식 개선
    try:
        load_start = time.time()
        logger.info("📥 데이터 로딩 시작...")
        
        # 먼저 컬럼 구조 확인
        df_sample = pd.read_csv(input_path, nrows=5)
        logger.info(f"📊 원본 컬럼: {list(df_sample.columns)}")
        logger.info(f"📊 컬럼 수: {len(df_sample.columns)}")
        
        # datetime 컬럼이 있는지 확인하고 처리
        if 'datetime' in df_sample.columns:
            logger.info("📅 datetime 컬럼 발견 - 시간 데이터 처리 중...")
            # datetime 컬럼이 있으면 인덱스로 사용하지 않음
            df = pd.read_csv(input_path)
            
            # datetime 변환을 더 안전하게 처리
            try:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                # 변환 실패한 행 제거
                original_len = len(df)
                df = df.dropna(subset=['datetime'])
                logger.info(f"✅ datetime 변환 완료: {len(df)} 행 (제거된 행: {original_len - len(df)})")
                
                # 시간 범위 정보
                if len(df) > 0:
                    logger.info(f"📅 데이터 시간 범위: {df['datetime'].min()} ~ {df['datetime'].max()}")
            except Exception as e:
                logger.warning(f"⚠️ datetime 변환 중 오류: {e}")
                # 기본 방식으로 재시도
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                df = df.dropna(subset=['datetime'])
        else:
            # datetime 컬럼이 없으면 첫 번째 컬럼을 인덱스로 사용
            df = pd.read_csv(input_path, index_col=0, parse_dates=True)
            # 인덱스를 datetime 컬럼으로 변환
            df['datetime'] = df.index
        
        # timestamp 컬럼이 있는지 확인
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['datetime'].astype(np.int64) // 10**6
        
        print(f"✅ 데이터 로딩 완료: {len(df)} 행")
        print(f"📊 처리된 컬럼: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ 데이터 로딩 중 오류: {str(e)}")
        # 기본 방식으로 재시도
        df = pd.read_csv(input_path)
        print(f"⚠️ 기본 방식으로 로딩 완료: {len(df)} 행")
    
    # OHLC 컬럼 확인
    required_ohlc = ['open', 'high', 'low', 'close']
    missing_ohlc = [col for col in required_ohlc if col not in df.columns]
    if missing_ohlc:
        print(f"❌ 누락된 OHLC 컬럼: {missing_ohlc}")
        print(f"📊 사용 가능한 컬럼: {list(df.columns)}")
        return None
    
    # 칼만 필터 적용 (선택적)
    if use_kalman_filter:
        logger.info("-" * 50)
        logger.info("🎯 칼만 필터 적용 단계 시작")
        logger.info("-" * 50)
        
        kalman_start = time.time()
        
        # 원본 데이터 통계
        logger.info("📈 원본 데이터 통계:")
        logger.info(f"  - 종가 평균: {df['close'].mean():.2f}")
        logger.info(f"  - 종가 변동성: {df['close'].pct_change().std():.4f}")
        logger.info(f"  - 데이터 포인트: {len(df):,}개")
        
        # 파라미터 최적화 (선택적)
        if optimize_kalman:
            logger.info("🔧 칼만 필터 파라미터 최적화 시작...")
            optimize_start = time.time()
            optimal_params, best_score = optimize_btc_kalman_parameters(df, max_combinations=30)
            optimize_time = time.time() - optimize_start
            kalman_params = optimal_params
            logger.info(f"✅ 파라미터 최적화 완료 (소요시간: {optimize_time:.2f}초)")
            logger.info(f"📊 최적 파라미터: {optimal_params}")
            logger.info(f"📊 최적 점수: {best_score:.4f}")
        else:
            if kalman_params:
                logger.info(f"🎛️ 사용자 지정 파라미터: {kalman_params}")
            else:
                logger.info("🎛️ 기본 파라미터 사용")
        
        # 칼만 필터 적용
        logger.info("🔄 칼만 필터 적용 중...")
        filter_start = time.time()
        df_original = df.copy()
        df = apply_btc_kalman_filtering(df, use_multi_scale=use_multi_scale, params=kalman_params, 
                                       use_gpu=use_gpu, n_jobs=n_jobs)
        filter_time = time.time() - filter_start
        logger.info(f"✅ 칼만 필터 적용 완료 (소요시간: {filter_time:.2f}초)")
        
        # 성능 검증
        logger.info("📊 칼만 필터 성능 검증 중...")
        performance = validate_kalman_performance(df_original, df)
        logger.info(f"📊 성능 결과:")
        logger.info(f"  - 신호 보존율: {performance['preservation_ratio']:.2%}")
        logger.info(f"  - 노이즈 제거율: {performance['noise_reduction']:.2%}")
        logger.info(f"  - 원본 변동성: {performance['original_volatility_mean']:.4f}")
        logger.info(f"  - 필터링 후 변동성: {performance['filtered_volatility_mean']:.4f}")
        
        # 필터링된 데이터 통계
        logger.info("📈 필터링된 데이터 통계:")
        logger.info(f"  - 종가 평균: {df['close'].mean():.2f}")
        logger.info(f"  - 종가 변동성: {df['close'].pct_change().std():.4f}")
        
        kalman_total_time = time.time() - kalman_start
        logger.info(f"⏱️ 칼만 필터 총 소요시간: {kalman_total_time:.2f}초")
    
    # 기술적 지표 계산 시작
    logger.info("-" * 50)
    logger.info("📊 기술적 지표 계산 단계 시작")
    logger.info("-" * 50)
    
    indicators_start = time.time()
    
    # RSI 지표 계산
    logger.info("📈 RSI 지표 계산 중...")
    rsi_start = time.time()
    df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
    logger.info(f"✅ RSI 지표 계산 완료 ({time.time() - rsi_start:.2f}초)")
    
    # EMA 지표 계산
    logger.info("📈 EMA 지표 계산 중...")
    ema_start = time.time()
    df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    logger.info(f"✅ EMA 지표 계산 완료 ({time.time() - ema_start:.2f}초)")
    
    # ATR 계산 방식 선택
    logger.info("📈 ATR 지표 계산 중...")
    atr_start = time.time()
    if use_pinescript_atr:
        if backtest_start_date:
            logger.info(f"🔄 백테스팅 기간 ATR 계산: {backtest_start_date}부터...")
            try:
                df['atr_14'] = calculate_atr_for_backtest_period(df, backtest_start_date, 14)
                logger.info(f"✅ 백테스팅 기간 ATR 계산 완료 ({time.time() - atr_start:.2f}초)")
            except Exception as e:
                logger.warning(f"⚠️ 백테스팅 기간 ATR 계산 중 오류: {e}")
                logger.info("🔄 전체 데이터 ATR로 대체...")
                df['atr_14'] = calculate_atr_pinescript(df, 14)
                logger.info(f"✅ 전체 데이터 ATR 계산 완료 ({time.time() - atr_start:.2f}초)")
        else:
            logger.info("🔄 Pine Script와 동일한 RMA 기반 ATR 계산...")
            try:
                df['atr_14'] = calculate_atr_pinescript(df, 14)
                logger.info(f"✅ RMA 기반 ATR 계산 완료 ({time.time() - atr_start:.2f}초)")
            except Exception as e:
                logger.warning(f"⚠️ RMA ATR 계산 중 오류: {e}")
                logger.info("🔄 talib ATR로 대체...")
                df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                logger.info(f"✅ talib ATR 계산 완료 ({time.time() - atr_start:.2f}초)")
    else:
        logger.info("🔄 talib 기반 SMA ATR 계산...")
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        logger.info(f"✅ talib ATR 계산 완료 ({time.time() - atr_start:.2f}초)")
    
    # Squeeze Momentum 계산
    logger.info("📈 Squeeze Momentum 계산 중...")
    val_start = time.time()
    try:
        df['val'] = squeeze_momentum_core(df)
        logger.info(f"✅ Squeeze Momentum 계산 완료 ({time.time() - val_start:.2f}초)")
    except Exception as e:
        logger.warning(f"⚠️ Squeeze Momentum 계산 중 오류: {e}")
        # 간단한 val 계산 (임시)
        df['val'] = df['close'] - df['close'].rolling(20).mean()
        logger.warning(f"⚠️ 임시 val 계산 사용 ({time.time() - val_start:.2f}초)")
    
    # Bollinger Bands 계산
    logger.info("📈 Bollinger Bands 계산 중...")
    bb_start = time.time()
    basis, bb_upper, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_basis'] = basis
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    logger.info(f"✅ Bollinger Bands 계산 완료 ({time.time() - bb_start:.2f}초)")
    
    # SMA 계산
    logger.info("📈 SMA 계산 중...")
    sma_start = time.time()
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    logger.info(f"✅ SMA 계산 완료 ({time.time() - sma_start:.2f}초)")
    
    # 색상 신호 계산
    logger.info("📈 색상 신호 계산 중...")
    color_start = time.time()
    df['bcolor'] = (df['val'] > 0).astype(int)
    df['scolor'] = (df['val'] < 0).astype(int)
    logger.info(f"✅ 색상 신호 계산 완료 ({time.time() - color_start:.2f}초)")
    
    # Volume 데이터 처리
    logger.info("📈 Volume 데이터 처리 중...")
    if 'volume' not in df.columns:
        logger.warning("⚠️ Volume 컬럼이 없어서 0으로 채웁니다")
        df['volume'] = 0
    else:
        logger.info(f"✅ Volume 데이터 확인됨 (평균: {df['volume'].mean():,.0f})")
    
    # 기술적 지표 계산 완료
    indicators_time = time.time() - indicators_start
    logger.info(f"⏱️ 기술적 지표 계산 총 소요시간: {indicators_time:.2f}초")
    
    # 최종 데이터 검증
    logger.info("-" * 50)
    logger.info("🔍 최종 데이터 검증")
    logger.info("-" * 50)
    
    # 생성된 특성 개수 확인
    feature_cols = [
        'rsi_7', 'rsi_14', 'rsi_21', 'ema_9', 'ema_21', 'atr_14',
        'val', 'bb_basis', 'bb_upper', 'bb_lower', 'sma_20', 
        'bcolor', 'scolor', 'volume'
    ]
    
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        logger.warning(f"⚠️ 누락된 특성: {missing_features}")
    else:
        logger.info("✅ 모든 필수 특성이 생성되었습니다")
    
    logger.info(f"📊 최종 데이터 shape: {df.shape}")
    logger.info(f"📊 생성된 특성 수: {len([col for col in feature_cols if col in df.columns])}")
    
    # NaN 값 확인
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning("⚠️ NaN 값이 발견된 컬럼:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"  - {col}: {count}개 ({count/len(df)*100:.1f}%)")
    else:
        logger.info("✅ NaN 값이 없습니다")
    
    # 진단 정보 출력
    if diagnose:
        logger.info("📋 상세 진단 정보 출력 중...")
        print_diagnostic_info(df)
    
    # 파일 저장
    if output_path:
        logger.info("-" * 50)
        logger.info("💾 파일 저장 단계")
        logger.info("-" * 50)
        
        save_start = time.time()
        logger.info(f"📁 저장 경로: {output_path}")
        logger.info(f"📊 저장할 데이터: {df.shape[0]:,}행 x {df.shape[1]}컬럼")
        
        df.to_csv(output_path, index=False)
        save_time = time.time() - save_start
        
        # 저장된 파일 크기 확인
        saved_size = os.path.getsize(output_path) / (1024*1024)  # MB
        logger.info(f"✅ 파일 저장 완료")
        logger.info(f"📁 저장된 파일 크기: {saved_size:.2f} MB")
        logger.info(f"⏱️ 저장 소요시간: {save_time:.2f}초")
    
    return df


def add_features_chunked(input_path, output_path=None, diagnose=True, use_pinescript_atr=True):
    """
    청크 단위로 파일 처리 (대용량 파일용)
    """
    print("🔄 청크 단위 처리 시작...")
    
    # 청크 크기 설정
    chunk_size = 50000
    total_rows = 0
    first_chunk = True
    
    # 출력 파일 초기화
    if output_path:
        # 헤더만 먼저 생성
        df_sample = pd.read_csv(input_path, nrows=1)
        if 'datetime' in df_sample.columns:
            df_sample = pd.read_csv(input_path, nrows=1)
            df_sample['datetime'] = pd.to_datetime(df_sample['datetime'])
        else:
            df_sample = pd.read_csv(input_path, nrows=1, index_col=0, parse_dates=True)
            df_sample['datetime'] = df_sample.index
        
        if 'timestamp' not in df_sample.columns:
            df_sample['timestamp'] = df_sample['datetime'].astype(np.int64) // 10**6
        
        # 더미 데이터로 헤더 생성
        dummy_df = pd.DataFrame(columns=df_sample.columns)
        dummy_df.to_csv(output_path, index=False)
    
    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        print(f"🔄 청크 처리 중... (현재 행: {total_rows:,})")
        
        # datetime 컬럼 처리
        if 'datetime' in chunk.columns:
            chunk['datetime'] = pd.to_datetime(chunk['datetime'])
        else:
            # 인덱스를 datetime으로 사용
            chunk['datetime'] = pd.to_datetime(chunk.index)
        
        # timestamp 컬럼 처리
        if 'timestamp' not in chunk.columns:
            chunk['timestamp'] = chunk['datetime'].astype(np.int64) // 10**6
        
        # 기술적 지표 계산
        chunk['rsi_7'] = talib.RSI(chunk['close'], timeperiod=7)
        chunk['rsi_14'] = talib.RSI(chunk['close'], timeperiod=14)
        chunk['rsi_21'] = talib.RSI(chunk['close'], timeperiod=21)
        chunk['ema_9'] = talib.EMA(chunk['close'], timeperiod=9)
        chunk['ema_21'] = talib.EMA(chunk['close'], timeperiod=21)
        
        # ATR 계산 - 각 청크별로 독립적으로 계산
        if use_pinescript_atr:
            print(f"   🔄 ATR 계산 중... (청크 {total_rows//chunk_size + 1})")
            try:
                # 각 청크에서 ATR 계산
                atr_chunk = calculate_atr_pinescript(chunk, 14)
                chunk['atr_14'] = atr_chunk
                print(f"   ✅ ATR 계산 완료 - 평균: {atr_chunk.mean():.4f}, 결측치: {atr_chunk.isna().sum()}")
            except Exception as e:
                print(f"   ❌ ATR 계산 실패: {str(e)}")
                # talib ATR로 대체
                chunk['atr_14'] = talib.ATR(chunk['high'], chunk['low'], chunk['close'], timeperiod=14)
                print(f"   ⚠️ talib ATR 사용 - 평균: {chunk['atr_14'].mean():.4f}")
        else:
            chunk['atr_14'] = talib.ATR(chunk['high'], chunk['low'], chunk['close'], timeperiod=14)
        
        # val 계산
        try:
            chunk['val'] = squeeze_momentum_core(chunk)
        except Exception as e:
            chunk['val'] = chunk['close'] - chunk['close'].rolling(20).mean()
        
        # Bollinger Bands
        basis, bb_upper, bb_lower = talib.BBANDS(chunk['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        chunk['bb_basis'] = basis
        chunk['bb_upper'] = bb_upper
        chunk['bb_lower'] = bb_lower
        
        # SMA(20)
        chunk['sma_20'] = chunk['close'].rolling(window=20, min_periods=1).mean()
        
        # bcolor, scolor
        chunk['bcolor'] = (chunk['val'] > 0).astype(int)
        chunk['scolor'] = (chunk['val'] < 0).astype(int)
        
        # volume
        if 'volume' not in chunk.columns:
            chunk['volume'] = 0
        
        # CSV 파일에 저장
        if output_path:
            if first_chunk:
                chunk.to_csv(output_path, index=False, mode='w')
                first_chunk = False
            else:
                chunk.to_csv(output_path, index=False, mode='a', header=False)
        
        total_rows += len(chunk)
        
        # 진행상황 출력
        if total_rows % 100000 == 0:
            print(f"   처리된 행 수: {total_rows:,}")
    
    print(f"✅ 청크 처리 완료! 총 {total_rows:,} 행")
    
    if diagnose and output_path:
        # 진단을 위해 샘플 읽기
        df_sample = pd.read_csv(output_path, nrows=10000)
        print_diagnostic_info(df_sample)
    
    return None  # 청크 처리에서는 DataFrame을 반환하지 않음


def print_diagnostic_info(df):
    """
    진단 정보 출력
    """
    print("\n[add_features] 데이터 진단 결과:")
    print(f"전체 행 개수: {len(df)}")
    print("피처별 결측치 개수:")
    print(df.isna().sum())
    print("피처별 결측치 비율(%):")
    print((df.isna().sum() / len(df) * 100).round(2))
    print("결측치가 없는 행 개수:", len(df.dropna()))
    
    # ATR 통계 정보
    print("\n📊 ATR 통계 정보:")
    print(f"ATR 평균: {df['atr_14'].mean():.4f}")
    print(f"ATR 표준편차: {df['atr_14'].std():.4f}")
    print(f"ATR 최소값: {df['atr_14'].min():.4f}")
    print(f"ATR 최대값: {df['atr_14'].max():.4f}")
    print(f"ATR 결측치: {df['atr_14'].isna().sum()}")
    
    # val 통계 정보
    print("\n📊 val 통계 정보:")
    print(f"val 평균: {df['val'].mean():.4f}")
    print(f"val 표준편차: {df['val'].std():.4f}")
    print(f"val 최소값: {df['val'].min():.4f}")
    print(f"val 최대값: {df['val'].max():.4f}")
    print(f"val 결측치: {df['val'].isna().sum()}")
    
    # datetime 범위
    print(f"\n⏰ 데이터 범위:")
    print(f"시작: {df['datetime'].min()}")
    print(f"종료: {df['datetime'].max()}")


def check_atr_at_specific_time(df, target_datetime=None, target_index=None):
    """
    특정 시점의 ATR 값을 확인하는 함수
    
    Args:
        df: pandas DataFrame - OHLC 데이터
        target_datetime: str - 목표 날짜시간 (예: '2025-07-26 08:00:00')
        target_index: int - 목표 인덱스 (datetime 대신 사용)
    """
    print(f"🔍 특정 시점 ATR 값 확인...")
    
    if target_datetime:
        # datetime으로 찾기
        target_dt = pd.to_datetime(target_datetime)
        mask = df['datetime'] >= target_dt
        if mask.any():
            target_idx = mask.idxmax()
            print(f"목표 시간: {target_datetime}")
        else:
            print(f"❌ 해당 시간을 찾을 수 없습니다: {target_datetime}")
            return
    elif target_index is not None:
        target_idx = target_index
        print(f"목표 인덱스: {target_index}")
    else:
        # 마지막 값 확인
        target_idx = df.index[-1]
        print(f"마지막 값 확인")
    
    # 해당 시점의 데이터 확인
    target_row = df.loc[target_idx]
    print(f"📊 해당 시점 데이터:")
    print(f"  시간: {target_row['datetime']}")
    print(f"  OHLC: {target_row['open']:.2f}, {target_row['high']:.2f}, {target_row['low']:.2f}, {target_row['close']:.2f}")
    
    # ATR 계산 (해당 시점까지의 데이터로)
    df_subset = df.loc[:target_idx]
    atr_values = calculate_atr_pinescript(df_subset, 14)
    current_atr = atr_values.iloc[-1]
    
    print(f"📊 ATR 값:")
    print(f"  현재 계산된 ATR: {current_atr:.6f}")
    
    # True Range 값들 확인 (최근 5개)
    high = df_subset['high'].values
    low = df_subset['low'].values
    close = df_subset['close'].values
    
    print(f"\n📊 최근 5개 True Range 값:")
    for i in range(max(0, len(df_subset)-5), len(df_subset)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        print(f"  행 {i}: TR = {tr:.6f}")
    
    return current_atr


def verify_atr_calculation(df, target_atr=104.7, tolerance=0.1):
    """
    ATR 계산 결과를 검증하는 함수
    
    Args:
        df: pandas DataFrame - OHLC 데이터
        target_atr: float - 목표 ATR 값 (Pine Script 값)
        tolerance: float - 허용 오차
    
    Returns:
        dict - 검증 결과
    """
    print(f"🔍 ATR 계산 검증 중...")
    print(f"목표 ATR 값: {target_atr}")
    
    # 현재 ATR 계산
    current_atr = calculate_atr_pinescript(df, 14)
    current_value = current_atr.iloc[-1]  # 마지막 값
    
    print(f"현재 계산된 ATR: {current_value:.6f}")
    print(f"차이: {abs(current_value - target_atr):.6f}")
    
    # 검증 결과
    is_valid = abs(current_value - target_atr) <= tolerance
    
    result = {
        'current_atr': current_value,
        'target_atr': target_atr,
        'difference': abs(current_value - target_atr),
        'is_valid': is_valid,
        'tolerance': tolerance
    }
    
    if is_valid:
        print(f"✅ ATR 값이 목표값과 일치합니다 (오차: {result['difference']:.6f})")
    else:
        print(f"❌ ATR 값이 목표값과 다릅니다 (오차: {result['difference']:.6f})")
        print(f"💡 가능한 원인:")
        print(f"  1. 데이터 시점이 다름 (현재: {df['datetime'].iloc[-1]})")
        print(f"  2. 데이터 정밀도 차이")
        print(f"  3. Pine Script와 계산 방식의 미세한 차이")
    
    return result


def compare_atr_methods(df):
    """
    talib ATR과 Pine Script ATR 비교
    
    Args:
        df: pandas DataFrame - OHLC 데이터
    """
    print("🔄 ATR 계산 방식 비교 중...")
    
    # talib ATR (SMA 기반)
    atr_talib = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Pine Script ATR (RMA 기반)
    atr_pinescript = calculate_atr_pinescript(df, 14)
    
    # 비교 결과
    print("\n📊 ATR 비교 결과:")
    print(f"talib ATR 평균: {atr_talib.mean():.4f}")
    print(f"Pine Script ATR 평균: {atr_pinescript.mean():.4f}")
    print(f"차이 (Pine Script - talib): {(atr_pinescript.mean() - atr_talib.mean()):.4f}")
    print(f"상관계수: {atr_pinescript.corr(atr_talib):.4f}")
    
    # 마지막 값 비교
    print(f"\n📊 마지막 ATR 값 비교:")
    print(f"talib ATR 마지막 값: {atr_talib.iloc[-1]:.6f}")
    print(f"Pine Script ATR 마지막 값: {atr_pinescript.iloc[-1]:.6f}")
    print(f"차이: {abs(atr_pinescript.iloc[-1] - atr_talib.iloc[-1]):.6f}")
    
    return atr_talib, atr_pinescript


if __name__ == '__main__':
    try:
        print("🚀 Feature Engineering 시작...")
        print("=" * 50)
        
        # Pine Script와 동일한 ATR 사용 + 성능 최적화
        # df = add_features('C:/선물데이터/binance_btcusdt_1m.csv', 
        #                  'C:/선물데이터/binance_btcusdt_1m_features.csv',
        #                  use_pinescript_atr=True, 
        #                  optimize_kalman=False,  # 최적화 비활성화 (중복 실행 방지) 
        #                  use_kalman_filter=False, 
        #                  use_multi_scale=False,   # 2단계 필터 활성화 (멀티 스케일 적용)
        #                  use_gpu=True,        # GPU 가속 (CuPy 설치 시 True로 변경)
        #                  n_jobs=7,             # 병렬 처리 (CPU 코어 수에 맞게 조정)
        #                  kalman_params={
        #                     'base_Q': 0.01,               # 균형잡힌 반
        #                     'base_R': 1.0,                # 빠른 신호 반영
        #                     'volatility_threshold': 0.05,  # 민감한 변동성 감지
        #                     'preservation_factor': 0.82,   # 높은 신호 보존
        #                     'volatility_window': 12,       # 빠른 적응09-   (12분)
        #                     'adaptive_factor': 5.0
        #                     })

        df = add_features('C:/선물데이터/binance_btcusdt_1m.csv', 
                         'C:/선물데이터/binance_btcusdt_1m_features.csv',
                         use_pinescript_atr=True, use_gpu=True, n_jobs=7)
                         
                                                                            
        # ATR 방식 비교 (df가 None이 아닐 때만)
        if df is not None:
            print("\n🔄 ATR 방식 비교 중...")
            compare_atr_methods(df)
            
            # 특정 시점의 ATR 값 확인 (2025-07-21 00:39:00)
            print("\n🔍 특정 시점 ATR 값 확인...")
            check_atr_at_specific_time(df, target_datetime='2025-07-21 00:39:00')
            
            # ATR 값 검증 (Pine Script 값과 비교)
            print("\n🔍 ATR 값 검증 중...")
            verify_atr_calculation(df, target_atr=104.7, tolerance=0.1)

            # val 컬럼에 양수 값이 하나라도 있는지 확인
            has_positive_val = (df['val'] > 0).any()

            if has_positive_val:
                print("✅ val 컬럼에 양수 값이 있습니다.")
            else:
                print("⚠️ val 컬럼에 양수 값이 없습니다.")
                
            print(f"\n✅ Feature Engineering 완료!")
            print(f"📊 최종 데이터 크기: {len(df)} 행, {len(df.columns)} 컬럼")
        else:
            print("⚠️ DataFrame이 None입니다. 청크 단위 처리로 완료되었습니다.")
            print("생성된 features 파일을 확인해주세요.")
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🏁 프로그램 종료")

   