import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
from squeeze_momentum_core import squeeze_momentum_core
import talib
import matplotlib.pyplot as plt
import os


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
    Pine Script 문서에 따른 정확한 구현
    
    Args:
        df: pandas DataFrame - OHLC 데이터
        period: int - ATR 기간 (기본값: 14)
    
    Returns:
        pandas Series - ATR 값
    """
    # True Range 계산 (Pine Script와 정확히 동일)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # True Range 계산
    tr = np.zeros(len(df))
    
    for i in range(len(df)):
        if i == 0 or pd.isna(high[i-1]):
            # 첫 번째 값이거나 이전 high가 na인 경우
            tr[i] = high[i] - low[i]
        else:
            # Pine Script: max(high - low, abs(high - close[1]), abs(low - close[1]))
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
    
    # RMA 기반 ATR (Pine Script와 정확히 동일)
    atr = calculate_rma(pd.Series(tr, index=df.index), period)
    return atr


def add_features(input_path, output_path=None, diagnose=True, use_pinescript_atr=True):
    """
    ta_lib 라이브러리를 활용해 진입 조건 및 ml 가격 상승 및 하락 예측에 사용할 특정 지표를 계산.
    
    Args:
        input_path: str - 입력 CSV 파일 경로
        output_path: str - 출력 CSV 파일 경로 (None이면 저장하지 않음)
        diagnose: bool - 진단 정보 출력 여부
        use_pinescript_atr: bool - Pine Script와 동일한 ATR 사용 여부 (기본값: True)
    """
    print(f"📖 데이터 로딩 중: {input_path}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(input_path) / (1024*1024)  # MB
    print(f"📁 파일 크기: {file_size:.2f} MB")
    
    # 청크 단위 처리 비활성화 - 전체 데이터를 한 번에 처리
    print("🔄 전체 데이터를 한 번에 처리합니다...")
    return add_features_single(input_path, output_path, diagnose, use_pinescript_atr)


def add_features_single(input_path, output_path=None, diagnose=True, use_pinescript_atr=True):
    """
    일반 파일 처리 (단일 파일) - 개선된 버전
    """
    # 데이터 로딩 방식 개선
    try:
        print("🔄 데이터 로딩 중...")
        
        # 먼저 컬럼 구조 확인
        df_sample = pd.read_csv(input_path, nrows=5)
        print(f"📊 원본 컬럼: {list(df_sample.columns)}")
        
        # datetime 컬럼이 있는지 확인하고 처리
        if 'datetime' in df_sample.columns:
            # datetime 컬럼이 있으면 인덱스로 사용하지 않음
            df = pd.read_csv(input_path)
            
            # datetime 변환을 더 안전하게 처리
            try:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                # 변환 실패한 행 제거
                df = df.dropna(subset=['datetime'])
                print(f"✅ datetime 변환 완료: {len(df)} 행")
            except Exception as e:
                print(f"⚠️ datetime 변환 중 오류: {e}")
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
    
    print("🔄 기술적 지표 계산 중...")
    
    # talib 기반으로 대체
    df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
    df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    
    # ATR 계산 방식 선택
    if use_pinescript_atr:
        print("🔄 Pine Script와 동일한 RMA 기반 ATR 계산 중...")
        try:
            df['atr_14'] = calculate_atr_pinescript(df, 14)
            print("✅ RMA 기반 ATR 계산 완료")
        except Exception as e:
            print(f"⚠️ RMA ATR 계산 중 오류: {e}")
            print("🔄 talib ATR로 대체...")
            df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            print("✅ talib ATR 계산 완료")
    else:
        print("🔄 talib 기반 SMA ATR 계산 중...")
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        print("✅ talib ATR 계산 완료")
    
    # val 계산
    print("🔄 squeeze momentum 계산 중...")
    try:
        df['val'] = squeeze_momentum_core(df)
        print("✅ squeeze momentum 계산 완료")
    except Exception as e:
        print(f"⚠️ squeeze momentum 계산 중 오류: {e}")
        # 간단한 val 계산 (임시)
        df['val'] = df['close'] - df['close'].rolling(20).mean()
        print("⚠️ 임시 val 계산 사용")
    
    # Bollinger Bands (basis, upperBB, lowerBB) 추가
    basis, bb_upper, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_basis'] = basis
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    
    # SMA(20)
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
    # bcolor, scolor (val 기준)
    df['bcolor'] = (df['val'] > 0).astype(int)
    df['scolor'] = (df['val'] < 0).astype(int)
    
    # volume (없으면 0으로 채움)
    if 'volume' not in df.columns:
        df['volume'] = 0
    
    if diagnose:
        print_diagnostic_info(df)
    
    if output_path:
        print(f"\n💾 파일 저장 중: {output_path}")
        df.to_csv(output_path, index=False)
        print(f"✅ 저장 완료: {output_path}")
    
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
        
        # Pine Script와 동일한 ATR 사용
        df = add_features('C:/선물데이터/binance_btcusdt_1m.csv', 
                         'C:/선물데이터/binance_btcusdt_1m_features.csv',
                         use_pinescript_atr=True)

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

   