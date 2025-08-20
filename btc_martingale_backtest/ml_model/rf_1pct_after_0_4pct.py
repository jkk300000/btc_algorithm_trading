import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import logging
import os
import sys



# 메인 모듈(직접 실행하는 파일)은 반드시 절대 경로를 사용해야 함
# 직접 실행 시 부모 디렉토리를 sys.path에 추가하여 절대 경로 import 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml_model.metrics import evaluate_model_performance



# logger 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

"""
특정 기간 동안 0.4% 상승 후 1% 상승할 확률을 예측하는 모델 훈련 및 예측
"""



# 0.4% 상승 달성 시점 필터링 함수
def filter_0_4_rise(df, horizon=190):
    rolling_max = df['close'].shift(-1).rolling(window=horizon, min_periods=1).max()
    return (rolling_max >= df['close'] * 1.004)

# horizon 내 1% 상승 달성 여부 타겟 생성 함수
def make_target_1pct_after_0_4pct(df, horizon=360):
    # 0.4% 상승 달성 시점만 필터링
    cond_0_4 = filter_0_4_rise(df, horizon=190)
    # 1% 상승 타겟 생성
    rolling_max = df['close'].shift(-1).rolling(window=horizon, min_periods=1).max()
    target = (rolling_max >= df['close'] * 1.01).astype(int)
    # 0.4% 상승 달성 시점만 남김
    target = target.where(cond_0_4, np.nan)
    return target

def train_and_predict_1pct_after_0_4pct(df, output_path=None, horizon=300):
    train_end = pd.Timestamp('2022-08-31 23:59:00+00:00').tz_localize(None)
    test_start = pd.Timestamp('2022-09-01 00:00:00+00:00').tz_localize(None)
    feature_cols = [
        'close', 'sma_20',
        'bb_upper', 'bb_lower', 'bb_basis',
        'val', 'bcolor', 'scolor', 'volume', 'atr_14'
    ]
    # 타겟 생성: 0.4% 상승 후 총 1% 상승
    df['target'] = make_target_1pct_after_0_4pct(df, horizon=horizon)
    # 0.4% 상승 달성 시점만 추출
    df = df[df['target'].notna()].copy()
    df['target'] = df['target'].astype(int)
    # 시간 분할
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)
    train_df = df[df.index <= train_end].copy()
    test_df = df[df.index >= test_start].copy()
    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, 'target']
    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'y_train shape: {y_train.shape}')
    logger.info(f"train_df['target'] value counts:\n{train_df['target'].value_counts()}")
    if len(X_train) == 0:
        logger.error("No training data available after dropna. Check your data and feature engineering steps.")
        raise ValueError("No training data available after dropna. Check your data and feature engineering steps.")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=7)
    logger.info('Training RandomForestClassifier...')
    model.fit(X_train, y_train)
    logger.info('Model training complete.')
    for name, importance in zip(feature_cols, model.feature_importances_):
        logger.info(f"Feature importance - {name}: {importance:.4f}")
    X_test = test_df[feature_cols].dropna()
    logger.info(f'X_test shape: {X_test.shape}')
    rf_pred = model.predict_proba(X_test)[:, 1]
    y_test = test_df.loc[X_test.index, 'target']
    y_pred = (rf_pred >= 0.5).astype(int)
    eval_result = evaluate_model_performance(y_test, y_pred, verbose=True)
    if eval_result['f1'] >= 0.5 and eval_result['precision'] >= 0.6 and eval_result['recall'] >= 0.4:
        result_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': y_pred
        })
        result_path = 'btc_martingale_backtest/rf_1pct_after_0_4pct_results.csv'
        result_df.to_csv(result_path, index=False)
        print("✅ F1, 정밀도, 재현율 기준을 모두 만족하여 결과를 저장했습니다.")
        df['rf_pred'] = np.nan
        df.loc[X_test.index, 'rf_pred'] = rf_pred
    else:
        print("❌ F1, 정밀도, 재현율 중 하나라도 기준 미달이므로 결과를 저장하지 않습니다.")
        df['rf_pred'] = np.nan
    print("[train_and_predict_1pct_after_0_4pct] 데이터 진단 결과:")
    print(f"전체 행 개수: {len(df)}")
    print("피처별 결측치 개수:")
    print(df.isna().sum())
    print("피처별 결측치 비율(%):")
    print((df.isna().sum() / len(df) * 100).round(2))
    print("결측치가 없는 행 개수:", len(df.dropna()))
    logger.info('Prediction complete. rf_pred column updated.')
    if output_path:
        df.to_csv(output_path)
        logger.info(f'Saved to {output_path}')
    return df

if __name__ == '__main__':
    df = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_features.csv', index_col=0, parse_dates=True)
    train_and_predict_1pct_after_0_4pct(df, 'C:/선물데이터/binance_btcusdt_1m_rf_1pct_after_0_4pct.csv') 