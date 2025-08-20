#!/usr/bin/env python3
"""
비트코인 칼만 필터 통합 테스트 스크립트

이 스크립트는 칼만 필터가 통합된 feature_engineering 모듈을 테스트합니다.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(output_path='test_btc_data.csv', periods=1000):
    """테스트용 비트코인 데이터 생성"""
    
    logger.info("🧪 테스트 데이터 생성 중...")
    
    # 날짜 범위 생성
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(minutes=i) for i in range(periods)]
    
    # 비트코인 가격 시뮬레이션 (높은 변동성)
    np.random.seed(42)
    base_price = 50000
    
    # 기본 가격 움직임
    returns = np.random.normal(0, 0.02, periods)  # 2% 일일 변동성
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 극한 변동성 구간 추가
    extreme_periods = [200, 400, 600, 800]
    for period in extreme_periods:
        if period + 10 < periods:
            prices[period:period+10] *= np.random.uniform(0.9, 1.1, 10)
    
    # OHLCV 데이터 생성
    test_data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # OHLC 생성
        open_price = price * np.random.uniform(0.999, 1.001)
        high_price = price * np.random.uniform(1.001, 1.005)
        low_price = price * np.random.uniform(0.995, 0.999)
        close_price = price
        
        # 거래량 생성
        volume = np.random.uniform(1000, 5000)
        
        test_data.append({
            'datetime': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # DataFrame 생성
    df = pd.DataFrame(test_data)
    
    # CSV 저장
    df.to_csv(output_path, index=False)
    logger.info(f"✅ 테스트 데이터 생성 완료: {output_path}")
    
    return output_path

def test_kalman_filter_integration():
    """칼만 필터 통합 테스트"""
    
    logger.info("🚀 칼만 필터 통합 테스트 시작")
    
    # 1. 테스트 데이터 생성
    test_file = create_test_data()
    
    # 2. 기존 방식 (칼만 필터 없음)
    logger.info("📊 기존 방식 테스트 (칼만 필터 없음)")
    from feature_engineering import add_features
    
    df_original = add_features(
        input_path=test_file,
        output_path='test_original_features.csv',
        use_kalman_filter=False,
        diagnose=True
    )
    
    # 3. 칼만 필터 적용 (기본 파라미터)
    logger.info("🎯 칼만 필터 적용 테스트 (기본 파라미터)")
    df_kalman_basic = add_features(
        input_path=test_file,
        output_path='test_kalman_basic_features.csv',
        use_kalman_filter=True,
        kalman_params=None,  # 기본 파라미터 사용
        diagnose=True
    )
    
    # 4. 칼만 필터 적용 (최적화된 파라미터)
    logger.info("🔧 칼만 필터 최적화 테스트")
    df_kalman_optimized = add_features(
        input_path=test_file,
        output_path='test_kalman_optimized_features.csv',
        use_kalman_filter=True,
        optimize_kalman=True,  # 파라미터 최적화
        diagnose=True
    )
    
    # 5. 성능 비교
    logger.info("📈 성능 비교 분석")
    compare_performance(df_original, df_kalman_basic, df_kalman_optimized)
    
    # 6. 파일 정리
    cleanup_test_files()
    
    logger.info("✅ 칼만 필터 통합 테스트 완료")

def compare_performance(df_original, df_kalman_basic, df_kalman_optimized):
    """성능 비교 분석"""
    
    logger.info("📊 성능 비교 분석 시작")
    
    # 변동성 계산
    def calculate_volatility(df):
        return df['close'].pct_change().rolling(20).std()
    
    vol_original = calculate_volatility(df_original)
    vol_kalman_basic = calculate_volatility(df_kalman_basic)
    vol_kalman_optimized = calculate_volatility(df_kalman_optimized)
    
    # 극한 변동성 구간 확인
    extreme_threshold = 0.1  # 10%
    extreme_original = (vol_original > extreme_threshold).sum()
    extreme_basic = (vol_kalman_basic > extreme_threshold).sum()
    extreme_optimized = (vol_kalman_optimized > extreme_threshold).sum()
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 칼만 필터 성능 비교 결과")
    print("="*60)
    
    print(f"📈 평균 변동성:")
    print(f"   원본 데이터: {vol_original.mean():.4f}")
    print(f"   칼만 필터 (기본): {vol_kalman_basic.mean():.4f}")
    print(f"   칼만 필터 (최적화): {vol_kalman_optimized.mean():.4f}")
    
    print(f"\n🎯 극한 변동성 구간 (>{extreme_threshold*100}%):")
    print(f"   원본 데이터: {extreme_original}개")
    print(f"   칼만 필터 (기본): {extreme_basic}개")
    print(f"   칼만 필터 (최적화): {extreme_optimized}개")
    
    print(f"\n📉 노이즈 제거 효과:")
    noise_reduction_basic = (vol_original.mean() - vol_kalman_basic.mean()) / vol_original.mean()
    noise_reduction_optimized = (vol_original.mean() - vol_kalman_optimized.mean()) / vol_original.mean()
    print(f"   칼만 필터 (기본): {noise_reduction_basic:.2%}")
    print(f"   칼만 필터 (최적화): {noise_reduction_optimized:.2%}")
    
    print(f"\n🔄 신호 보존율:")
    preservation_basic = vol_kalman_basic.mean() / vol_original.mean()
    preservation_optimized = vol_kalman_optimized.mean() / vol_original.mean()
    print(f"   칼만 필터 (기본): {preservation_basic:.2%}")
    print(f"   칼만 필터 (최적화): {preservation_optimized:.2%}")
    
    print("="*60)

def cleanup_test_files():
    """테스트 파일 정리"""
    
    test_files = [
        'test_btc_data.csv',
        'test_original_features.csv',
        'test_kalman_basic_features.csv',
        'test_kalman_optimized_features.csv'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"🗑️ 테스트 파일 삭제: {file}")

def test_kalman_parameters():
    """칼만 필터 파라미터 테스트"""
    
    logger.info("🔧 칼만 필터 파라미터 테스트")
    
    from kalman_filter_btc import get_dynamic_kalman_params
    
    # 시장 상황별 파라미터 테스트
    market_conditions = ['bull_market', 'bear_market', 'sideways']
    
    for condition in market_conditions:
        params = get_dynamic_kalman_params(condition)
        logger.info(f"📊 {condition} 파라미터: {params}")

def main():
    """메인 함수"""
    
    print("🚀 비트코인 칼만 필터 통합 테스트")
    print("="*50)
    
    try:
        # 1. 기본 통합 테스트
        test_kalman_filter_integration()
        
        # 2. 파라미터 테스트
        test_kalman_parameters()
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        logger.error(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 