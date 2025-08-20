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
    rf_path: 참조 CSV (binance_btcusdt_1m_rf_1pct_after_0_4pct.csv)
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
    # 상승 예측 컬럼 업데이트
    print("="*60)
    print("상승 예측 컬럼 업데이트")
    print("="*60)
    
    columns_to_update = ['target', 'rf_pred']
    update_columns_from_rf(
        var_path='C:/선물데이터/binance_btcusdt_1m_rf_var.csv',
        rf_path='C:/선물데이터/binance_btcusdt_1m_rf_1pct_after_0_4pct.csv',
        columns=columns_to_update
    )
    
    # 하락 예측 컬럼 추가
    print("\n" + "="*60) 
    print("하락 예측 컬럼 추가")
    print("="*60)
    
    # 하락 예측 파일에서 컬럼 추가
    df_var = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_rf_var.csv', index_col=0, parse_dates=True)
    df_down = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_rf_down.csv', index_col=0, parse_dates=True)
    # df_down = pd.read_csv('C:/선물데이터/binance_btcusdt_1m_rf_10pct_after_5pct.csv', index_col=0, parse_dates=True)
    
    # 하락 예측 컬럼 추가
    if 'rf_pred_down' in df_down.columns:
        common_idx = df_var.index.intersection(df_down.index)
        df_var.loc[common_idx, 'rf_pred_down'] = df_down.loc[common_idx, 'rf_pred_down']
        print(f"하락 예측 컬럼 추가 완료: {df_var['rf_pred_down'].notna().sum():,}개")
    else:
        print("[경고] rf_pred_down 컬럼이 하락 예측 파일에 없습니다.")
    
    # 결과 저장
    df_var.to_csv('C:/선물데이터/binance_btcusdt_1m_rf_var.csv')
    print("하락 예측 컬럼이 포함된 파일이 저장되었습니다.") 