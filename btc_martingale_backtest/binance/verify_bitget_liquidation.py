#!/usr/bin/env python3
"""
비트겟 거래소 청산가 계산 방식 검증 스크립트
목표: 60,603.03과 비슷한 청산가 계산
"""

def calculate_bitget_liquidation_price(avg_price, entry_count, leverage, initial_balance, divided_count):
    """
    비트겟 거래소 청산가 계산 방식
    
    비트겟의 특징:
    1. 진입 횟수에 따라 동적 레버리지 적용
    2. 시드 분할을 고려한 유효 레버리지 계산
    3. 마진 비율이 아닌 실제 레버리지 기반 청산가 계산
    """
    if entry_count <= 2:
        return None
    
    # 비트겟 방식: 진입 횟수에 따른 동적 레버리지
    # 유효 레버리지 = (기본레버리지 × 진입횟수) / 시드분할
    effective_leverage = (leverage * entry_count) / divided_count
    
    if effective_leverage < 1.0:
        return None
    
    # 비트겟 청산가 공식: 평균가 × (1 - 1/유효레버리지)
    # 이는 증거금이 0이 되는 지점
    liquidation_price = avg_price * (1 - 1/effective_leverage)
    
    return max(liquidation_price, 0)

def calculate_bitget_alternative_method(avg_price, entry_count, leverage, initial_balance, divided_count):
    """
    비트겟 대안 방식: 진입 횟수별 가중 레버리지
    """
    if entry_count <= 2:
        return None
    
    # 진입 횟수에 따른 가중치 적용
    # 3번째 진입: 1.5배, 4번째 진입: 2.0배, 5번째 진입: 2.5배
    weight_multiplier = 1.0 + (entry_count - 3) * 0.5
    
    # 가중 레버리지 계산
    weighted_leverage = leverage * weight_multiplier / divided_count
    
    if weighted_leverage < 1.0:
        return None
    
    liquidation_price = avg_price * (1 - 1/weighted_leverage)
    return max(liquidation_price, 0)

def calculate_bitget_risk_based_method(avg_price, entry_count, leverage, initial_balance, divided_count):
    """
    비트겟 리스크 기반 방식: 진입 횟수에 따른 리스크 조정
    """
    if entry_count <= 2:
        return None
    
    # 진입 횟수에 따른 리스크 계수
    risk_factors = {
        3: 0.8,   # 3번째 진입: 80% 리스크
        4: 0.6,   # 4번째 진입: 60% 리스크
        5: 0.4,   # 5번째 진입: 40% 리스크
        6: 0.3,   # 6번째 진입: 30% 리스크
    }
    
    risk_factor = risk_factors.get(entry_count, 0.2)
    
    # 리스크 조정된 레버리지
    adjusted_leverage = (leverage * entry_count * risk_factor) / divided_count
    
    if adjusted_leverage < 1.0:
        return None
    
    liquidation_price = avg_price * (1 - 1/adjusted_leverage)
    return max(liquidation_price, 0)

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
    initial_balance = 570
    divided_count = 20
    
    print("=== 비트겟 거래소 청산가 계산 방식 검증 ===")
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
    print()
    
    # 비트겟 방식별 청산가 계산
    methods = [
        ("비트겟 기본 방식", calculate_bitget_liquidation_price),
        ("비트겟 가중 방식", calculate_bitget_alternative_method),
        ("비트겟 리스크 기반", calculate_bitget_risk_based_method)
    ]
    
    for method_name, method_func in methods:
        print(f"--- {method_name} ---")
        
        for entry_count in range(3, total_quantity + 1):
            liq_price = method_func(avg_price, entry_count, leverage, initial_balance, divided_count)
            
            if liq_price is not None:
                difference = abs(liq_price - target_liquidation)
                print(f"  {entry_count}번째 진입: {liq_price:,.2f} (차이: {difference:,.2f})")
                
                # 목표 청산가와 10% 이내 차이인지 확인
                if difference <= target_liquidation * 0.1:
                    print(f"    ✅ 목표 청산가와 유사 (10% 이내)")
                    
                # 유효 레버리지 계산
                effective_leverage = (leverage * entry_count) / divided_count
                print(f"    유효레버리지: {effective_leverage:.2f}배")
            else:
                print(f"  {entry_count}번째 진입: 계산 불가")
        print()
    
    # 역산 검증: 목표 청산가를 만드는 레버리지 찾기
    print("=== 역산 검증 ===")
    print(f"목표 청산가: {target_liquidation:,.2f}")
    print(f"평단가: {avg_price:,.2f}")
    
    # 목표 청산가를 만드는 마진 비율 계산
    margin_ratio = 1 - (target_liquidation / avg_price)
    print(f"필요한 마진 비율: {margin_ratio:.4f} ({margin_ratio*100:.2f}%)")
    
    # 이 마진 비율을 만드는 레버리지 계산
    required_leverage = 1 / margin_ratio
    print(f"필요한 레버리지: {required_leverage:.2f}배")
    
    # 진입 횟수별로 필요한 레버리지 분석
    print("\n진입 횟수별 분석:")
    for entry_count in range(3, total_quantity + 1):
        # 시드 분할 고려한 유효 레버리지
        effective_leverage = (leverage * entry_count) / divided_count
        print(f"  {entry_count}번째 진입:")
        print(f"    유효레버리지 = {effective_leverage:.2f}배")
        
        # 이 레버리지로 계산되는 청산가
        if effective_leverage > 1:
            liq_price = avg_price * (1 - 1/effective_leverage)
            difference = abs(liq_price - target_liquidation)
            print(f"    청산가: {liq_price:,.2f} (차이: {difference:,.2f})")
            
            # 목표 청산가와의 차이 분석
            if difference <= target_liquidation * 0.1:
                print(f"    ✅ 목표 청산가와 유사!")
        else:
            print(f"    청산가: 계산 불가 (레버리지 < 1)")
        
        # 목표 청산가를 만들기 위한 필요한 유효 레버리지
        target_effective_leverage = 1 / margin_ratio
        print(f"    목표 청산가를 위한 필요 유효레버리지: {target_effective_leverage:.2f}배")
        
        # 현재 설정으로는 어떤 진입에서 목표 청산가가 나오는지
        if effective_leverage >= target_effective_leverage:
            print(f"    ✅ 현재 설정으로 목표 청산가 달성 가능")
        else:
            print(f"    ❌ 현재 설정으로는 목표 청산가 달성 불가")
        print()

if __name__ == "__main__":
    main()
