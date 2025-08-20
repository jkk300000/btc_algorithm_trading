#!/usr/bin/env python3
"""
실제 거래소 기록 데이터로 새로운 전략 검증
"""

def calculate_bitget_liquidation_price(avg_price, entry_count, leverage, divided_count):
    """
    비트겟 동적 레버리지 기반 청산가 계산 함수
    """
    if entry_count <= 2:
        return None
    
    # 비트겟 동적 레버리지 공식
    effective_leverage = (leverage * entry_count) / divided_count
    
    if effective_leverage < 1.0:
        return None
    
    # 비트겟 청산가 공식: 평균가 × (1 - 1/유효레버리지)
    liquidation_price = avg_price * (1 - 1/effective_leverage)
    return max(liquidation_price, 0)

def calculate_effective_leverage(entry_count, leverage, divided_count):
    """
    동적 레버리지 계산 함수
    """
    if entry_count <= 0 or leverage <= 0 or divided_count <= 0:
        return 0.0
    return (leverage * entry_count) / divided_count

def main():
    # 실제 거래소 기록 데이터
    entry_prices = [
        122940.5,  # 1번째 진입가
        121702.1,  # 2번째 진입가
        119348.6,  # 3번째 진입가
        119138.0,  # 4번째 진입가
        115918.9   # 5번째 진입가
    ]
    
    # 실제 거래소 기록
    real_avg_price = 119809.62  # 실제 평단가
    real_liquidation_price = 60603.0377  # 실제 청산가
    
    # 전략 파라미터
    leverage = 10
    divided_count = 20
    
    print("=== 실제 거래소 기록 데이터 검증 ===")
    print(f"실제 평단가: {real_avg_price:,.2f}")
    print(f"실제 청산가: {real_liquidation_price:,.4f}")
    print()
    
    # 진입가별 분석
    print("=== 진입가별 분석 ===")
    for i, price in enumerate(entry_prices, 1):
        print(f"{i}번째 진입가: {price:,.1f}")
    print()
    
    # 전략 계산 결과
    print("=== 전략 계산 결과 ===")
    
    # 평단가 계산
    total_cost = sum(entry_prices)
    total_quantity = len(entry_prices)
    calculated_avg_price = total_cost / total_quantity
    
    print(f"계산된 평단가: {calculated_avg_price:,.2f}")
    print(f"실제 평단가: {real_avg_price:,.2f}")
    print(f"평단가 차이: {abs(calculated_avg_price - real_avg_price):,.2f}")
    print()
    
    # 진입 횟수별 청산가 계산
    print("=== 진입 횟수별 청산가 계산 ===")
    
    for entry_count in range(1, len(entry_prices) + 1):
        # 부분 평단가 계산 (1번째부터 entry_count까지)
        partial_prices = entry_prices[:entry_count]
        partial_total_cost = sum(partial_prices)
        partial_avg_price = partial_total_cost / entry_count
        
        # 유효 레버리지 계산
        effective_leverage = calculate_effective_leverage(entry_count, leverage, divided_count)
        
        # 청산가 계산
        liquidation_price = calculate_bitget_liquidation_price(partial_avg_price, entry_count, leverage, divided_count)
        
        print(f"--- {entry_count}번째 진입까지 ---")
        print(f"  부분 평단가: {partial_avg_price:,.2f}")
        print(f"  유효레버리지: {effective_leverage:.2f}배")
        
        if liquidation_price is not None:
            print(f"  계산된 청산가: {liquidation_price:,.4f}")
            
            # 실제 청산가와 비교
            difference = abs(liquidation_price - real_liquidation_price)
            difference_percent = (difference / real_liquidation_price) * 100
            
            print(f"  실제 청산가: {real_liquidation_price:,.4f}")
            print(f"  차이: {difference:,.4f} ({difference_percent:.2f}%)")
            
            # 5% 이내 차이인지 확인
            if difference_percent <= 5.0:
                print(f"  ✅ 정확도 높음 (5% 이내)")
            elif difference_percent <= 10.0:
                print(f"  ⚠️ 보통 정확도 (10% 이내)")
            else:
                print(f"  ❌ 정확도 낮음 (10% 초과)")
        else:
            print(f"  청산가: 계산 불가 (레버리지 < 1.0)")
        print()
    
    # 역산 검증: 실제 청산가를 만드는 레버리지 찾기
    print("=== 역산 검증 ===")
    print(f"실제 청산가: {real_liquidation_price:,.4f}")
    print(f"실제 평단가: {real_avg_price:,.2f}")
    
    # 실제 청산가를 만드는 마진 비율 계산
    margin_ratio = 1 - (real_liquidation_price / real_avg_price)
    print(f"필요한 마진 비율: {margin_ratio:.4f} ({margin_ratio*100:.2f}%)")
    
    # 이 마진 비율을 만드는 레버리지 계산
    required_leverage = 1 / margin_ratio
    print(f"필요한 레버리지: {required_leverage:.2f}배")
    
    # 어떤 진입 횟수에서 이 레버리지가 나오는지 확인
    print(f"\n목표 레버리지 {required_leverage:.2f}배를 만드는 진입 횟수:")
    for entry_count in range(1, len(entry_prices) + 1):
        effective_leverage = calculate_effective_leverage(entry_count, leverage, divided_count)
        print(f"  {entry_count}번째 진입: 유효레버리지 = {effective_leverage:.2f}배")
        
        if abs(effective_leverage - required_leverage) <= 0.1:
            print(f"    ✅ 목표 레버리지와 거의 일치!")
        elif effective_leverage >= required_leverage:
            print(f"    ⚠️ 목표 레버리지 이상 (청산가 계산 가능)")
        else:
            print(f"    ❌ 목표 레버리지 미달 (청산가 계산 불가)")

if __name__ == "__main__":
    main()
