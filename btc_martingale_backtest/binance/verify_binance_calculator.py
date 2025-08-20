#!/usr/bin/env python3
"""
바이낸스 계산기로 거래소 거래 기록 검증 (올바른 공식 적용)
목표: 60,603.03과 비슷한 청산가 계산
"""

def calculate_binance_liquidation_price(entry_price, actual_leverage, maintenance_margin=0.005):
    """
    바이낸스 청산가 공식: Entry Price - ((Entry Price / Leverage) * (1 + Maintenance Margin))
    """
    if actual_leverage <= 0:
        return float('inf')
    
    liquidation_price = entry_price - ((entry_price / actual_leverage) * (1 + maintenance_margin))
    return max(liquidation_price, 0)

def main():
    # 거래소 거래 기록
    entry_prices = [
        122940.5,  # 1번째 진입가
        121702.1,  # 2번째 진입가
        119348.6,  # 3번째 진입가
        119138.0,  # 4번째 진입가
        115918.9   # 5번째 진입가
    ]
    
    target_liquidation = 60603.0377  # 목표 청산가
    leverage = 10
    initial_balance = 590
    divided_count = 20
    maintenance_margin = 0.005  # 0.5%
    
    print("=== 바이낸스 계산기로 거래소 거래 기록 검증 (올바른 공식) ===")
    print(f"목표 청산가: {target_liquidation:,.2f}")
    print(f"Maintenance Margin: {maintenance_margin*100:.1f}%")
    print()
    
    # 평단가 계산
    total_cost = sum(entry_prices)
    total_quantity = len(entry_prices)
    avg_price = total_cost / total_quantity
    
    print(f"평단가: {avg_price:,.2f}")
    print(f"총 진입 횟수: {total_quantity}")
    print(f"기본 레버리지: {leverage}배")
    print(f"초기 자본: {initial_balance}")
    print(f"시드 분할: {divided_count}")
    print()
    
    # 실제 진입 수량 (반올림 고려)
    actual_quantity_per_trade = 0.002  # 반올림된 실제 수량
    total_actual_quantity = actual_quantity_per_trade * total_quantity
    
    print(f"실제 진입 수량 (반올림 후): {actual_quantity_per_trade} BTC")
    print(f"총 실제 수량: {total_actual_quantity} BTC")
    print()
    
    # 진입 횟수별 청산가 계산
    print("--- 진입 횟수별 청산가 계산 ---")
    
    for entry_count in range(1, total_quantity + 1):
        # 해당 진입 횟수까지의 실제 수량
        current_actual_quantity = actual_quantity_per_trade * entry_count
        
        # 해당 진입 횟수까지의 평단가 계산
        current_prices = entry_prices[:entry_count]
        current_total_cost = sum(current_prices)
        current_avg_price = current_total_cost / entry_count
        
        # 총 포지션 가치
        total_position_value = current_actual_quantity * current_avg_price
        
        # 총 자산 (초기 자본)
        total_assets = initial_balance
        
        # 실제 사용 레버리지 = 총 포지션 가치 / 총 자산
        actual_leverage = total_position_value / total_assets
        
        print(f"  {entry_count}번째 진입:")
        print(f"    현재 평단가: {current_avg_price:,.2f}")
        print(f"    실제 수량: {current_actual_quantity} BTC")
        print(f"    총 포지션 가치: {total_position_value:,.2f}")
        print(f"    총 자산: {total_assets:,.2f}")
        print(f"    실제 사용 레버리지: {actual_leverage:.2f}배")
        
        # 바이낸스 청산가 공식으로 계산
        if actual_leverage > 0:
            liquidation_price = calculate_binance_liquidation_price(current_avg_price, actual_leverage, maintenance_margin)
            difference = abs(liquidation_price - target_liquidation)
            
            print(f"    계산된 청산가: {liquidation_price:,.2f}")
            print(f"    목표 청산가와의 차이: {difference:,.2f}")
            
            # 목표 청산가와 10% 이내 차이인지 확인
            if difference <= target_liquidation * 0.1:
                print(f"    ✅ 목표 청산가와 유사 (10% 이내)")
            else:
                print(f"    ❌ 목표 청산가와 차이 큼")
        else:
            print(f"    청산가: 계산 불가 (레버리지 <= 0)")
        
        print()
    
    # 역산 검증
    print("=== 역산 검증 ===")
    print(f"목표 청산가: {target_liquidation:,.2f}")
    print(f"최종 평단가: {avg_price:,.2f}")
    
    # 목표 청산가를 만드는 레버리지 계산
    # Liquidation Price = Entry Price - ((Entry Price / Leverage) * (1 + Maintenance Margin))
    # 60,603 = 119,809 - ((119,809 / Leverage) * 1.005)
    # (119,809 / Leverage) * 1.005 = 119,809 - 60,603 = 59,206
    # 119,809 / Leverage = 59,206 / 1.005 = 58,911.44
    # Leverage = 119,809 / 58,911.44 = 2.03
    
    required_leverage = avg_price / ((avg_price - target_liquidation) / (1 + maintenance_margin))
    print(f"목표 청산가를 위한 필요 레버리지: {required_leverage:.2f}배")
    
    # 실제 수량으로 필요한 레버리지 달성 가능 여부
    print(f"\n실제 수량으로 필요한 레버리지 달성 가능 여부:")
    for entry_count in range(1, total_quantity + 1):
        current_actual_quantity = actual_quantity_per_trade * entry_count
        current_prices = entry_prices[:entry_count]
        current_avg_price = sum(current_prices) / entry_count
        total_position_value = current_actual_quantity * current_avg_price
        achievable_leverage = total_position_value / initial_balance
        
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
