import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
import os

def analyze_trade_accuracy_from_csv(csv_file_path):
    """
    CSV 거래 로그를 분석하여 ML 모델의 실제 예측률 계산
    """
    print("=" * 60)
    print("CSV 거래 로그 분석 - ML 모델 예측률 계산")
    print("=" * 60)
    
    try:
        # CSV 파일 로드
        df = pd.read_csv(csv_file_path, parse_dates=['timestamp'])
        print(f"CSV 파일 로드 완료: {len(df):,}개 거래 로그")
        print(f"컬럼: {list(df.columns)}")
        
    except FileNotFoundError:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_file_path}")
        return
    except Exception as e:
        print(f"❌ CSV 파일 로드 오류: {e}")
        return
    
    # 기본 통계
    print("\n" + "="*50)
    print("1. 기본 거래 통계")
    print("="*50)
    
    print(f"총 거래 로그 수: {len(df)}")
    print(f"고유 거래 ID 수: {df['trade_id'].nunique()}")
    
    # 액션 타입별 통계
    action_counts = df['action_type'].value_counts()
    print(f"\n액션 타입별 통계:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}개")
    
    # rf_pred 통계
    if 'rf_pred' in df.columns:
        rf_pred_stats = df['rf_pred'].describe()
        print(f"\nrf_pred 통계:")
        print(rf_pred_stats)
    
    # 2. 예측률 분석
    print("\n" + "="*50)
    print("2. ML 모델 예측률 분석")
    print("="*50)
    
    # 물타기 없는 거래만 필터링 (entry만 있는 거래)
    entry_trades = df[df['action_type'] == 'entry']['trade_id'].unique()
    simple_trades = []
    
    for trade_id in entry_trades:
        trade_data = df[df['trade_id'] == trade_id]
        
        # 물타기가 없는 거래인지 확인
        has_martingale = len(trade_data[trade_data['action_type'] == 'martingale']) > 0
        
        if not has_martingale:
            # 최종 청산이 있는지 확인
            final_exit = trade_data[trade_data['action_type'] == 'final_exit']
            if len(final_exit) > 0:
                simple_trades.append({
                    'trade_id': trade_id,
                    'entry_data': trade_data[trade_data['action_type'] == 'entry'].iloc[0],
                    'exit_data': final_exit.iloc[0],
                    'rf_pred': trade_data[trade_data['action_type'] == 'entry'].iloc[0]['rf_pred'],
                    'threshold': trade_data[trade_data['action_type'] == 'entry'].iloc[0]['threshold'],
                    'profit_ratio': final_exit.iloc[0]['profit_ratio'],
                    'profit_loss': final_exit.iloc[0]['profit_loss']
                })
    
    print(f"물타기 없는 거래 수: {len(simple_trades)}")
    
    if len(simple_trades) > 0:
        # 수익/손실 거래 분석
        profitable_trades = [t for t in simple_trades if t['profit_ratio'] > 0]
        loss_trades = [t for t in simple_trades if t['profit_ratio'] <= 0]
        
        print(f"수익 거래: {len(profitable_trades)}개")
        print(f"손실 거래: {len(loss_trades)}개")
        
        accuracy = len(profitable_trades) / len(simple_trades) * 100
        print(f"예측 정확도: {accuracy:.2f}%")
        
        # 평균 수익률
        if profitable_trades:
            avg_profit = np.mean([t['profit_ratio'] for t in profitable_trades])
            print(f"평균 수익률: {avg_profit:.2f}%")
        
        if loss_trades:
            avg_loss = np.mean([t['profit_ratio'] for t in loss_trades])
            print(f"평균 손실률: {avg_loss:.2f}%")
    
    # 3. 임계값별 예측률 분석
    print("\n" + "="*50)
    print("3. 임계값별 예측률 분석")
    print("="*50)
    
    if len(simple_trades) > 0:
        threshold_analysis = []
        
        # 임계값별로 그룹화
        threshold_groups = {}
        for trade in simple_trades:
            threshold = trade['threshold']
            if threshold not in threshold_groups:
                threshold_groups[threshold] = []
            threshold_groups[threshold].append(trade)
        
        for threshold, trades in threshold_groups.items():
            profitable = [t for t in trades if t['profit_ratio'] > 0]
            accuracy = len(profitable) / len(trades) * 100
            avg_profit = np.mean([t['profit_ratio'] for t in profitable]) if profitable else 0
            
            threshold_analysis.append({
                'threshold': threshold,
                'total_trades': len(trades),
                'profitable_trades': len(profitable),
                'accuracy': accuracy,
                'avg_profit': avg_profit
            })
            
            print(f"임계값 {threshold}:")
            print(f"  총 거래: {len(trades)}개")
            print(f"  수익 거래: {len(profitable)}개")
            print(f"  정확도: {accuracy:.2f}%")
            print(f"  평균 수익률: {avg_profit:.2f}%")
            print()
    
    # 4. rf_pred 값별 예측률 분석
    print("\n" + "="*50)
    print("4. rf_pred 값별 예측률 분석")
    print("="*50)
    
    if len(simple_trades) > 0:
        # rf_pred 구간별 분석
        rf_pred_values = [t['rf_pred'] for t in simple_trades]
        rf_pred_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rf_pred_labels = ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        # 구간별로 거래 분류
        rf_pred_analysis = []
        
        for i in range(len(rf_pred_bins) - 1):
            min_pred = rf_pred_bins[i]
            max_pred = rf_pred_bins[i + 1]
            label = rf_pred_labels[i]
            
            bin_trades = [t for t in simple_trades if min_pred <= t['rf_pred'] < max_pred]
            
            if len(bin_trades) > 0:
                profitable = [t for t in bin_trades if t['profit_ratio'] > 0]
                accuracy = len(profitable) / len(bin_trades) * 100
                avg_profit = np.mean([t['profit_ratio'] for t in profitable]) if profitable else 0
                
                rf_pred_analysis.append({
                    'rf_pred_range': label,
                    'total_trades': len(bin_trades),
                    'profitable_trades': len(profitable),
                    'accuracy': accuracy,
                    'avg_profit': avg_profit
                })
                
                print(f"rf_pred {label}:")
                print(f"  총 거래: {len(bin_trades)}개")
                print(f"  수익 거래: {len(profitable)}개")
                print(f"  정확도: {accuracy:.2f}%")
                print(f"  평균 수익률: {avg_profit:.2f}%")
                print()
    
    # 5. 시각화
    print("\n" + "="*50)
    print("5. 시각화 생성")
    print("="*50)
    
    plt.figure(figsize=(15, 10))
    
    # 1) rf_pred 분포
    plt.subplot(2, 3, 1)
    if 'rf_pred' in df.columns:
        plt.hist(df['rf_pred'].dropna(), bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('rf_pred 값')
        plt.ylabel('거래 개수')
        plt.title('rf_pred 분포')
        plt.grid(True, alpha=0.3)
    
    # 2) 수익률 분포
    plt.subplot(2, 3, 2)
    if len(simple_trades) > 0:
        profit_ratios = [t['profit_ratio'] for t in simple_trades]
        plt.hist(profit_ratios, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('수익률 (%)')
        plt.ylabel('거래 개수')
        plt.title('수익률 분포')
        plt.grid(True, alpha=0.3)
    
    # 3) rf_pred vs 수익률
    plt.subplot(2, 3, 3)
    if len(simple_trades) > 0:
        rf_preds = [t['rf_pred'] for t in simple_trades]
        profit_ratios = [t['profit_ratio'] for t in simple_trades]
        plt.scatter(rf_preds, profit_ratios, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('rf_pred')
        plt.ylabel('수익률 (%)')
        plt.title('rf_pred vs 수익률')
        plt.grid(True, alpha=0.3)
    
    # 4) 임계값별 정확도
    plt.subplot(2, 3, 4)
    if len(threshold_analysis) > 0:
        thresholds = [t['threshold'] for t in threshold_analysis]
        accuracies = [t['accuracy'] for t in threshold_analysis]
        plt.bar(thresholds, accuracies)
        plt.xlabel('임계값')
        plt.ylabel('정확도 (%)')
        plt.title('임계값별 예측 정확도')
        plt.grid(True, alpha=0.3)
    
    # 5) rf_pred 구간별 정확도
    plt.subplot(2, 3, 5)
    if len(rf_pred_analysis) > 0:
        ranges = [r['rf_pred_range'] for r in rf_pred_analysis]
        accuracies = [r['accuracy'] for r in rf_pred_analysis]
        plt.bar(ranges, accuracies)
        plt.xlabel('rf_pred 구간')
        plt.ylabel('정확도 (%)')
        plt.title('rf_pred 구간별 예측 정확도')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 6) 액션 타입 분포
    plt.subplot(2, 3, 6)
    action_counts = df['action_type'].value_counts()
    plt.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
    plt.title('액션 타입 분포')
    
    plt.tight_layout()
    plt.savefig('trade_accuracy_analysis_from_csv.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'total_trades': len(df),
        'unique_trades': df['trade_id'].nunique(),
        'simple_trades': len(simple_trades),
        'profitable_trades': len(profitable_trades) if 'profitable_trades' in locals() else 0,
        'accuracy': accuracy if 'accuracy' in locals() else 0,
        'threshold_analysis': threshold_analysis if 'threshold_analysis' in locals() else [],
        'rf_pred_analysis': rf_pred_analysis if 'rf_pred_analysis' in locals() else []
    }

def analyze_trade_accuracy(log_file_path):
    """
    거래 로그를 분석하여 ML 모델의 실제 예측률 계산
    """
    print("=" * 60)
    print("거래 로그 분석 - ML 모델 예측률 계산")
    print("=" * 60)
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        print(f"로그 파일 로드 완료: {len(log_content)} 문자")
    except FileNotFoundError:
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file_path}")
        return
    
    # 거래 패턴 분석
    trades = []
    current_trade = None
    
    # 로그 라인별 분석
    lines = log_content.split('\n')
    
    for line in lines:
        # 진입 패턴 찾기
        if '[진입]' in line:
            # 기존 거래가 있으면 저장
            if current_trade:
                trades.append(current_trade)
            
            # 새로운 거래 시작
            current_trade = {
                'entry_time': None,
                'entry_price': None,
                'rf_pred': None,
                'threshold': None,
                'entry_count': None,
                'avg_price': None,
                'exit_time': None,
                'exit_price': None,
                'exit_type': None,  # 'profit', 'partial', 'margin_call'
                'profit_loss': None,
                'profit_ratio': None,
                'martingale_count': 0
            }
            
            # 진입 정보 추출
            try:
                # 시간 추출
                time_match = re.search(r'\[([^\]]+)\]', line)
                if time_match:
                    current_trade['entry_time'] = time_match.group(1)
                
                # 가격 추출
                price_match = re.search(r'진입가: ([\d.]+)', line)
                if price_match:
                    current_trade['entry_price'] = float(price_match.group(1))
                
                # rf_pred 추출
                rf_match = re.search(r'rf_pred: ([\d.]+)', line)
                if rf_match:
                    current_trade['rf_pred'] = float(rf_match.group(1))
                
                # 임계값 추출
                threshold_match = re.search(r'임계값: ([\d.]+)', line)
                if threshold_match:
                    current_trade['threshold'] = float(threshold_match.group(1))
                
                # entryCount 추출
                entry_count_match = re.search(r'entryCount: (\d+)', line)
                if entry_count_match:
                    current_trade['entry_count'] = int(entry_count_match.group(1))
                
                # 평균가 추출
                avg_price_match = re.search(r'평균가: ([\d.]+)', line)
                if avg_price_match:
                    current_trade['avg_price'] = float(avg_price_match.group(1))
                    
            except Exception as e:
                print(f"진입 정보 파싱 오류: {e}")
        
        # 물타기 패턴 찾기
        elif '[물타기]' in line and current_trade:
            current_trade['martingale_count'] += 1
        
        # 청산 패턴 찾기
        elif '[최종청산]' in line and current_trade:
            try:
                # 청산 시간 추출
                time_match = re.search(r'\[([^\]]+)\]', line)
                if time_match:
                    current_trade['exit_time'] = time_match.group(1)
                
                # 청산 가격 추출
                price_match = re.search(r'청산가: ([\d.]+)', line)
                if price_match:
                    current_trade['exit_price'] = float(price_match.group(1))
                
                # 수익률 계산
                if current_trade['avg_price'] and current_trade['exit_price']:
                    current_trade['profit_ratio'] = (current_trade['exit_price'] / current_trade['avg_price'] - 1) * 100
                    current_trade['profit_loss'] = current_trade['exit_price'] - current_trade['avg_price']
                
                current_trade['exit_type'] = 'profit'
                
            except Exception as e:
                print(f"청산 정보 파싱 오류: {e}")
        
        # 부분 청산 패턴 찾기
        elif '[부분청산]' in line and current_trade:
            current_trade['exit_type'] = 'partial'
        
        # 마진콜 패턴 찾기
        elif '마진콜' in line and current_trade:
            current_trade['exit_type'] = 'margin_call'
    
    # 마지막 거래 추가
    if current_trade:
        trades.append(current_trade)
    
    print(f"\n총 거래 수: {len(trades)}")
    
    # 거래 데이터프레임 생성
    trades_df = pd.DataFrame(trades)
    
    # 기본 통계
    print("\n" + "="*50)
    print("1. 기본 거래 통계")
    print("="*50)
    
    print(f"총 거래 수: {len(trades_df)}")
    print(f"완료된 거래 수: {len(trades_df[trades_df['exit_type'].notna()])}")
    print(f"진행 중인 거래 수: {len(trades_df[trades_df['exit_type'].isna()])}")
    
    # rf_pred 분포
    if 'rf_pred' in trades_df.columns:
        rf_pred_stats = trades_df['rf_pred'].describe()
        print(f"\nrf_pred 통계:")
        print(rf_pred_stats)
    
    # 2. 예측률 분석
    print("\n" + "="*50)
    print("2. ML 모델 예측률 분석")
    print("="*50)
    
    # 물타기 없는 거래만 필터링
    simple_trades = trades_df[trades_df['martingale_count'] == 0].copy()
    print(f"물타기 없는 거래 수: {len(simple_trades)}")
    
    # 수익 거래 분석
    profitable_trades = simple_trades[simple_trades['exit_type'] == 'profit'].copy()
    loss_trades = simple_trades[simple_trades['exit_type'] == 'profit'].copy()
    
    if len(profitable_trades) > 0:
        profitable_trades = profitable_trades[profitable_trades['profit_ratio'] > 0]
        loss_trades = loss_trades[loss_trades['profit_ratio'] <= 0]
        
        print(f"수익 거래: {len(profitable_trades)}개")
        print(f"손실 거래: {len(loss_trades)}개")
        
        if len(simple_trades) > 0:
            accuracy = len(profitable_trades) / len(simple_trades) * 100
            print(f"예측 정확도: {accuracy:.2f}%")
    
    # 3. 임계값별 예측률 분석
    print("\n" + "="*50)
    print("3. 임계값별 예측률 분석")
    print("="*50)
    
    if 'threshold' in trades_df.columns:
        threshold_analysis = []
        
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
            threshold_trades = simple_trades[simple_trades['threshold'] == threshold]
            
            if len(threshold_trades) > 0:
                profitable = threshold_trades[threshold_trades['profit_ratio'] > 0]
                accuracy = len(profitable) / len(threshold_trades) * 100
                avg_profit = profitable['profit_ratio'].mean() if len(profitable) > 0 else 0
                
                threshold_analysis.append({
                    'threshold': threshold,
                    'total_trades': len(threshold_trades),
                    'profitable_trades': len(profitable),
                    'accuracy': accuracy,
                    'avg_profit': avg_profit
                })
                
                print(f"임계값 {threshold}:")
                print(f"  총 거래: {len(threshold_trades)}개")
                print(f"  수익 거래: {len(profitable)}개")
                print(f"  정확도: {accuracy:.2f}%")
                print(f"  평균 수익률: {avg_profit:.2f}%")
                print()
    
    # 4. rf_pred 값별 예측률 분석
    print("\n" + "="*50)
    print("4. rf_pred 값별 예측률 분석")
    print("="*50)
    
    if 'rf_pred' in trades_df.columns:
        # rf_pred 구간별 분석
        rf_pred_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rf_pred_labels = ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        simple_trades['rf_pred_bin'] = pd.cut(simple_trades['rf_pred'], bins=rf_pred_bins, labels=rf_pred_labels)
        
        rf_pred_analysis = []
        
        for bin_label in rf_pred_labels:
            bin_trades = simple_trades[simple_trades['rf_pred_bin'] == bin_label]
            
            if len(bin_trades) > 0:
                profitable = bin_trades[bin_trades['profit_ratio'] > 0]
                accuracy = len(profitable) / len(bin_trades) * 100
                avg_profit = profitable['profit_ratio'].mean() if len(profitable) > 0 else 0
                
                rf_pred_analysis.append({
                    'rf_pred_range': bin_label,
                    'total_trades': len(bin_trades),
                    'profitable_trades': len(profitable),
                    'accuracy': accuracy,
                    'avg_profit': avg_profit
                })
                
                print(f"rf_pred {bin_label}:")
                print(f"  총 거래: {len(bin_trades)}개")
                print(f"  수익 거래: {len(profitable)}개")
                print(f"  정확도: {accuracy:.2f}%")
                print(f"  평균 수익률: {avg_profit:.2f}%")
                print()
    
    # 5. 시각화
    print("\n" + "="*50)
    print("5. 시각화 생성")
    print("="*50)
    
    plt.figure(figsize=(15, 10))
    
    # 1) rf_pred 분포
    plt.subplot(2, 3, 1)
    if 'rf_pred' in trades_df.columns:
        plt.hist(trades_df['rf_pred'].dropna(), bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('rf_pred 값')
        plt.ylabel('거래 개수')
        plt.title('rf_pred 분포')
        plt.grid(True, alpha=0.3)
    
    # 2) 수익률 분포
    plt.subplot(2, 3, 2)
    if 'profit_ratio' in trades_df.columns:
        profit_ratios = trades_df['profit_ratio'].dropna()
        plt.hist(profit_ratios, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('수익률 (%)')
        plt.ylabel('거래 개수')
        plt.title('수익률 분포')
        plt.grid(True, alpha=0.3)
    
    # 3) rf_pred vs 수익률
    plt.subplot(2, 3, 3)
    if 'rf_pred' in trades_df.columns and 'profit_ratio' in trades_df.columns:
        valid_trades = trades_df.dropna(subset=['rf_pred', 'profit_ratio'])
        plt.scatter(valid_trades['rf_pred'], valid_trades['profit_ratio'], alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('rf_pred')
        plt.ylabel('수익률 (%)')
        plt.title('rf_pred vs 수익률')
        plt.grid(True, alpha=0.3)
    
    # 4) 임계값별 정확도
    plt.subplot(2, 3, 4)
    if len(threshold_analysis) > 0:
        thresholds = [t['threshold'] for t in threshold_analysis]
        accuracies = [t['accuracy'] for t in threshold_analysis]
        plt.bar(thresholds, accuracies)
        plt.xlabel('임계값')
        plt.ylabel('정확도 (%)')
        plt.title('임계값별 예측 정확도')
        plt.grid(True, alpha=0.3)
    
    # 5) rf_pred 구간별 정확도
    plt.subplot(2, 3, 5)
    if len(rf_pred_analysis) > 0:
        ranges = [r['rf_pred_range'] for r in rf_pred_analysis]
        accuracies = [r['accuracy'] for r in rf_pred_analysis]
        plt.bar(ranges, accuracies)
        plt.xlabel('rf_pred 구간')
        plt.ylabel('정확도 (%)')
        plt.title('rf_pred 구간별 예측 정확도')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 6) 거래 결과 분포
    plt.subplot(2, 3, 6)
    if 'exit_type' in trades_df.columns:
        exit_types = trades_df['exit_type'].value_counts()
        plt.pie(exit_types.values, labels=exit_types.index, autopct='%1.1f%%')
        plt.title('거래 결과 분포')
    
    plt.tight_layout()
    plt.savefig('trade_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'total_trades': len(trades_df),
        'simple_trades': len(simple_trades),
        'profitable_trades': len(profitable_trades) if 'profitable_trades' in locals() else 0,
        'accuracy': accuracy if 'accuracy' in locals() else 0,
        'threshold_analysis': threshold_analysis if 'threshold_analysis' in locals() else [],
        'rf_pred_analysis': rf_pred_analysis if 'rf_pred_analysis' in locals() else []
    }

def analyze_trade_accuracy_auto(file_path):
    """
    파일 확장자에 따라 자동으로 적절한 분석 함수 선택
    """
    if file_path.endswith('.csv'):
        return analyze_trade_accuracy_from_csv(file_path)
    else:
        return analyze_trade_accuracy(file_path)

if __name__ == "__main__":
    # 파일 경로 설정 (CSV 또는 텍스트 로그 파일)
    file_path = "trade_logs/trade_logs_20250729_110119.csv"  # 실제 파일 경로로 수정
    
    try:
        results = analyze_trade_accuracy_auto(file_path)
        print(f"\n분석 완료!")
        print(f"총 거래 수: {results['total_trades']}")
        print(f"물타기 없는 거래 수: {results['simple_trades']}")
        print(f"수익 거래 수: {results['profitable_trades']}")
        print(f"예측 정확도: {results['accuracy']:.2f}%")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc() 