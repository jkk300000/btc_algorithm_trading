import pandas as pd
from typing import List

def update_columns_from_rf(
    var_path: str,
    rf_path: str,
    columns: List[str],
    output_path: str = None
):
    """
    var_path: 갱신 대상 CSV (binance_btcusdt_1m_rf_var.csv)
    rf_path: 참조 CSV (binance_btcusdt_1m_rf.csv)
    columns: 갱신할 컬럼 리스트
    output_path: 저장 경로 (None이면 var_path에 덮어씀)
    """
    df_var = pd.read_csv(var_path, index_col=0, parse_dates=True)
    df_rf = pd.read_csv(rf_path, index_col=0, parse_dates=True)
    # 인덱스 align
    common_idx = df_var.index.intersection(df_rf.index)
    for col in columns:
        if col in df_rf.columns:
            df_var.loc[common_idx, col] = df_rf.loc[common_idx, col]
        else:
            print(f"[경고] {col} 컬럼이 rf 파일에 없습니다.")
    save_path = output_path if output_path else var_path
    df_var.to_csv(save_path)
    print(f"{save_path} 파일이 갱신되었습니다.")

if __name__ == "__main__":
    # columns_to_update = [
    #     'rsi_7', 'rsi_14', 'rsi_21',
    #     'ema_9', 'ema_21',
    #     'atr_14', 'val', 'target', 'rf_pred'
    # ]

    columns_to_update = [
     'target', 'rf_pred'
    ]
    update_columns_from_rf(
        var_path='C:/선물데이터/binance_btcusdt_1m_rf_var.csv',
        rf_path='C:/선물데이터/binance_btcusdt_1m_rf_1pct_after_0_4pct.csv',
        columns=columns_to_update
    ) 