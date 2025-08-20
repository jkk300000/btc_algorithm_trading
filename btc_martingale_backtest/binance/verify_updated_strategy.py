#!/usr/bin/env python3
"""
수정된 전략의 청산가 계산 검증 스크립트
목표: 60,603.03과 비슷한 청산가 계산
"""

def calculate_actual_leverage(entry_count, leverage, divided_count, total_actual_quantity, avgPrice, initial_capital):
    """
    수정된 실제레버리지 계산 함수
    """
    if entry_count <= 0 or leverage <= 0 or divided_count <= 0:
        return 0.0
    else:
        # 🆕 실제 진입 수량을 고려한 레버리지 계산
        return (total_actual_quantity * avgPrice) / (initial_capital * leverage / divided_count)

def calculate_martingale_liquidation_price(avg_price, entry_count, leverage, divided_count, total_actual_quantity, initial_capital):
    """
    수정된 비트겟 청산가 계산 함수
    """
    if entry_count <= 2:
        return None
    else:
        # 🆕 수정된 레버리지 계산 함수 사용
        actual_leverage = calculate_actual_leverage(entry_count, leverage, divided_count, total_actual_quantity, avg_price, initial_capital)
        
        # 레버리지가 1.0 미만이면 청산가 계산 불가
        if actual_leverage < 1.0:
            return None
        else:
            # 🆕 비트겟 청산가 공식: 평균가 × (1 - 1/실제레버리지)
            return avg_price * (1 - 1/actual_leverage)

def main():
    # 제공된 데이터
    entry_prices = [
        122941.9,  # 1번째 진입가
        121672.3,  # 2번째 진입가
        119845.1,  # 3번째 진입가
        118564.6,  # 4번째 진입가
        115908.7   # 5번째 진입가
    ]
    
    target_liquidation = 60603.03  # 목표 청산가
    leverage = 10
    initial_capital = 570.0  # 🆕 고정값으로 설정
    divided_count = 20
    
    print("=== 수정된 전략 청산가 계산 검증 (고정 초기자본) ===")
    print(f"목표 청산가: {target_liquidation:,.2f}")
    print()
    
    # 평단가 계산
    total_cost = sum(entry_prices)
    total_quantity = len(entry_prices)
    avg_price = total_cost / total_quantity
    
    print(f"평단가: {avg_price:,.2f}")
    print(f"총 진입 횟수: {total_quantity}")
    print(f"기본 레버리지: {leverage}배")
    print(f"시드 분할: {divided_count}")
    print(f"초기 자본: {initial_capital:,.2f}")
    print()
    
    # 실제 진입 수량 (반올림 고려)
    actual_quantity_per_trade = 0.002  # 반올림된 실제 수량
    total_actual_quantity = actual_quantity_per_trade * total_quantity
    
    print(f"실제 진입 수량 (반올림 후): {actual_quantity_per_trade} BTC")
    print(f"총 실제 수량: {total_actual_quantity} BTC")
    print()
    
    # 진입 횟수별 청산가 계산
    print("--- 진입 횟수별 청산가 계산 ---")
    
    for entry_count in range(3, total_quantity + 1):
        # 해당 진입 횟수까지의 실제 수량
        current_actual_quantity = actual_quantity_per_trade * entry_count
        
        # 수정된 레버리지 계산
        actual_leverage = calculate_actual_leverage(entry_count, leverage, divided_count, current_actual_quantity, avg_price, initial_capital)
        
        # 수정된 청산가 계산
        liq_price = calculate_martingale_liquidation_price(avg_price, entry_count, leverage, divided_count, current_actual_quantity, initial_capital)
        
        if liq_price is not None:
            difference = abs(liq_price - target_liquidation)
            print(f"  {entry_count}번째 진입:")
            print(f"    실제 수량: {current_actual_quantity} BTC")
            print(f"    실제 레버리지: {actual_leverage:.2f}배")
            print(f"    계산된 청산가: {liq_price:,.2f}")
            print(f"    목표 청산가와의 차이: {difference:,.2f}")
            
            # 목표 청산가와 10% 이내 차이인지 확인
            if difference <= target_liquidation * 0.1:
                print(f"    ✅ 목표 청산가와 유사 (10% 이내)")
            else:
                print(f"    ❌ 목표 청산가와 차이 큼")
        else:
            print(f"  {entry_count}번째 진입: 계산 불가")
        
        print()
    
    # 역산 검증
    print("=== 역산 검증 ===")
    print(f"목표 청산가: {target_liquidation:,.2f}")
    print(f"평단가: {avg_price:,.2f}")
    
    # 목표 청산가를 만드는 마진 비율 계산
    margin_ratio = 1 - (target_liquidation / avg_price)
    print(f"필요한 마진 비율: {margin_ratio:.4f} ({margin_ratio*100:.2f}%)")
    
    # 이 마진 비율을 만드는 레버리지 계산
    required_leverage = 1 / margin_ratio
    print(f"필요한 레버리지: {required_leverage:.2f}배")
    
    # 실제 수량으로 필요한 레버리지 달성 가능 여부
    print(f"\n실제 수량으로 필요한 레버리지 달성 가능 여부:")
    for entry_count in range(3, total_quantity + 1):
        current_actual_quantity = actual_quantity_per_trade * entry_count
        achievable_leverage = (current_actual_quantity * avg_price) / (initial_capital * leverage / divided_count)
        
        print(f"  {entry_count}번째 진입:")
        print(f"    달성 가능한 레버리지: {achievable_leverage:.2f}배")
        print(f"    필요한 레버리지: {required_leverage:.2f}배")
        
        if achievable_leverage >= required_leverage:
            print(f"    ✅ 목표 청산가 달성 가능!")
        else:
            print(f"    ❌ 목표 청산가 달성 불가")
        print()

if __name__ == "__main__":
    main()
