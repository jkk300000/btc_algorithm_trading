import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
가격 10% 하락 확률 예측

"""


def make_target_down(df, horizon=22000, threshold=0.06):
    # horizon 내 최저가가 기준 하락률을 넘으면 1
    rolling_min = df['close'].shift(-1).rolling(window=horizon, min_periods=1).min()
    return (rolling_min <= df['close'] * (1 - threshold)).astype(int)

def train_and_predict_down(df, output_path=None):
    # train_end = pd.Timestamp('2022-08-31 23:59:00',tz= 'utc')
    train_end = pd.Timestamp('2022-08-31 23:59:00+00:00')
    train_end = train_end.tz_localize(None)
    # test_start = pd.Timestamp('2022-09-01 00:00:00', tz='UTC')
    test_start = pd.Timestamp('2022-09-01 00:00:00+00:00')
    test_start = test_start.tz_localize(None)
    df['target_down'] = make_target_down(df)
    train_df = df[df.index <= train_end].copy()
    test_df = df[df.index >= test_start].copy()
    feature_cols = [
        'close', 'sma_20',
        'bb_upper', 'bb_lower', 'bb_basis',
        'val', 'bcolor', 'scolor', 'volume', 'atr_14'
    ]
    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, 'target_down']
    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'y_train shape: {y_train.shape}')
    logger.info(f"train_df['target_down'] value counts:\n{train_df['target_down'].value_counts()}")
    if len(X_train) == 0:
        logger.error("No training data available after dropna. Check your data and feature engineering steps.")
        raise ValueError("No training data available after dropna. Check your data and feature engineering steps.")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=7)
    logger.info('Training RandomForestClassifier for 10% drop prediction...')
    model.fit(X_train, y_train)
    logger.info('Model training complete.')
    # feature importance 출력
    for name, importance in zip(feature_cols, model.feature_importances_):
        logger.info(f"Feature importance - {name}: {importance:.4f}")
    X_test = test_df[feature_cols].dropna()
    logger.info(f'X_test shape: {X_test.shape}')
    rf_pred_down = model.predict_proba(X_test)[:, 1]
    # df['rf_pred_down'] = np.nan
    # df.loc[X_test.index, 'rf_pred_down'] = rf_pred_down

    # ====== 평가 및 조건부 저장 ======
    # 테스트셋에 대해 실제값과 예측값 준비
    y_test = test_df.loc[X_test.index, 'target_down']
    y_pred = (rf_pred_down >= 0.5).astype(int)
    eval_result = evaluate_model_performance(y_test, y_pred, verbose=True)
    # 실전 기준: F1 ≥ 0.5, 정밀도 ≥ 0.5, 재현율 ≥ 0.4 (하락 예측은 완화된 기준)
    if eval_result['f1'] >= 0.7 and eval_result['precision'] >= 0.5 and eval_result['recall'] >= 0.6:
        result_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': y_pred
        })
        result_path = 'btc_martingale_backtest/rf_down_good_results.csv'
        result_df.to_csv(result_path, index=False)
        print("✅ F1, 정밀도, 재현율 기준을 모두 만족하여 결과를 저장했습니다.")
        # df에 rf_pred_down 값을 반영 (조건 만족 시)
        df['rf_pred_down'] = np.nan
        df.loc[X_test.index, 'rf_pred_down'] = rf_pred_down
    else:
        print("❌ F1, 정밀도, 재현율 중 하나라도 기준 미달이므로 결과를 저장하지 않습니다.")
        # 조건 미달 시 df의 rf_pred_down 전체를 결측치로
        df['rf_pred_down'] = np.nan

    print("[train_and_predict_down] 데이터 진단 결과:")
    print(f"전체 행 개수: {len(df)}")
    print("피처별 결측치 개수:")
    print(df.isna().sum())
    print("피처별 결측치 비율(%):")
    print((df.isna().sum() / len(df) * 100).round(2))
    print("결측치가 없는 행 개수:", len(df.dropna()))

    logger.info('Prediction complete. rf_pred_down column updated.')
    if output_path:
        df.to_csv(output_path)
        logger.info(f'Saved to {output_path}')
    return df

if __name__ == '__main__':
    df = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_features.csv', index_col=0, parse_dates=True)
    train_and_predict_down(df, 'C:/선물데이터/binance_btcusdt_1m_rf_down.csv') 