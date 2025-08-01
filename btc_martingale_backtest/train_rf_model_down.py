import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import logging
from metrics import evaluate_model_performance

# logger 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


"""
특정 기간 동안 5% 하락 후 10% 하락할 확률을 예측하는 모델 훈련 및 예측
"""



# 5% 하락 달성 시점 필터링 함수
def filter_5pct_drop(df, horizon=23000):
    """
    5% 하락이 달성되는 시점을 필터링
    """
    rolling_min = df['close'].shift(-1).rolling(window=horizon, min_periods=1).min()
    return (rolling_min <= df['close'] * 0.95)

# horizon 내 10% 하락 달성 여부 타겟 생성 함수
def make_target_10pct_after_5pct(df, horizon=26000):
    """
    5% 하락 달성 후 10% 이상 하락할 확률을 예측하는 타겟 생성
    """
    # 5% 하락 달성 시점만 필터링
    cond_5pct = filter_5pct_drop(df, horizon=23000)
    
    # 10% 하락 타겟 생성
    rolling_min = df['close'].shift(-1).rolling(window=horizon, min_periods=1).min()
    target = (rolling_min <= df['close'] * 0.90).astype(int)
    
    # 5% 하락 달성 시점만 남김
    target = target.where(cond_5pct, np.nan)
    return target

def train_and_predict_10pct_after_5pct(df, output_path=None, horizon=26000):
    """
    5% 하락 후 10% 이상 하락할 확률을 예측하는 모델 훈련 및 예측
    """
    train_end = pd.Timestamp('2022-08-31 23:59:00+00:00').tz_localize(None)
    test_start = pd.Timestamp('2022-09-01 00:00:00+00:00').tz_localize(None)
    
    feature_cols = [
        'close', 'sma_20',
        'bb_upper', 'bb_lower', 'bb_basis',
        'val', 'bcolor', 'scolor', 'volume', 'atr_14'
    ]
    
    # 타겟 생성: 5% 하락 후 총 10% 하락
    df['target'] = make_target_10pct_after_5pct(df, horizon=horizon)
    
    # 5% 하락 달성 시점만 추출
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
    logger.info('Training RandomForestClassifier for 10% drop after 5% drop prediction...')
    model.fit(X_train, y_train)
    logger.info('Model training complete.')
    
    # 피처 중요도 출력
    for name, importance in zip(feature_cols, model.feature_importances_):
        logger.info(f"Feature importance - {name}: {importance:.4f}")
    
    X_test = test_df[feature_cols].dropna()
    logger.info(f'X_test shape: {X_test.shape}')
    
    # 예측 확률 (10% 하락할 확률)
    rf_pred = model.predict_proba(X_test)[:, 1]
    y_test = test_df.loc[X_test.index, 'target']
    y_pred = (rf_pred >= 0.5).astype(int)
    
    # 모델 성능 평가
    eval_result = evaluate_model_performance(y_test, y_pred, verbose=True)
    
    # 성능 기준: F1 >= 0.5, 정밀도 >= 0.6, 재현율 >= 0.4
    if eval_result['f1'] >= 0.5 and eval_result['precision'] >= 0.55 and eval_result['recall'] >= 0.4:
        result_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': y_pred
        })
        result_path = 'btc_martingale_backtest/rf_10pct_after_5pct_results.csv'
        result_df.to_csv(result_path, index=False)
        print("✅ F1, 정밀도, 재현율 기준을 모두 만족하여 결과를 저장했습니다.")
        
        # 전체 데이터에 예측 결과 추가
        df['rf_pred_down'] = np.nan
        df.loc[X_test.index, 'rf_pred_down'] = rf_pred
    else:
        print("❌ F1, 정밀도, 재현율 중 하나라도 기준 미달이므로 결과를 저장하지 않습니다.")
        df['rf_pred_down'] = np.nan
    
    # 데이터 진단 결과 출력
    print("[train_and_predict_10pct_after_5pct] 데이터 진단 결과:")
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
    train_and_predict_10pct_after_5pct(df, 'C:/선물데이터/binance_btcusdt_1m_rf_10pct_after_5pct.csv') 