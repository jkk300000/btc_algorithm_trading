"""
바이낸스 거래소 계산 도구
청산가, 마진, 레버리지 등 바이낸스 기준 계산 함수들
"""

def calculate_liquidation_price(entry_price, position_size, leverage, initial_balance):
    """
    바이낸스 선물 청산가 계산
    
    Args:
        entry_price (float): 진입가(평균가)
        position_size (float): 포지션 크기 (달러)
        leverage (float): 레버리지
        initial_balance (float): 초기 자본
    
    Returns:
        float: 청산가
    """
    if position_size <= 0:
        return float('inf')
    
    # 바이낸스 선물 청산가 공식
    # 청산가 = 진입가 - (잔고 × 레버리지) / 포지션_수량
    liquidation_price = entry_price - (initial_balance * leverage) / position_size
    return max(liquidation_price, 0)  # 음수 방지

def calculate_margin_requirement(position_size, leverage):
    """
    마진 요구사항 계산
    
    Args:
        position_size (float): 포지션 크기 (달러)
        leverage (float): 레버리지
    
    Returns:
        float: 필요한 마진
    """
    return position_size / leverage

def calculate_max_loss_allowed(initial_balance, leverage):
    """
    최대 허용 손실 계산
    
    Args:
        initial_balance (float): 초기 자본
        leverage (float): 레버리지
    
    Returns:
        float: 최대 허용 손실
    """
    return initial_balance * (leverage - 1) / leverage

def calculate_position_size(initial_balance, leverage, divided_count, current_price):
    """
    포지션 크기 계산 (분할 투자 고려)
    
    Args:
        initial_balance (float): 초기 자본
        leverage (float): 레버리지
        divided_count (int): 분할 횟수
        current_price (float): 현재 가격
    
    Returns:
        float: 포지션 크기 (BTC)
    """
    initial_capital = initial_balance * leverage
    capital_per_once = initial_capital / divided_count
    position_size = capital_per_once / current_price
    return position_size

def calculate_actual_leverage(total_position_value, initial_balance):
    """
    실제 레버리지 계산
    
    Args:
        total_position_value (float): 총 포지션 가치
        initial_balance (float): 초기 자본
    
    Returns:
        float: 실제 레버리지
    """
    return total_position_value / initial_balance

def validate_liquidation_risk(current_price, liquidation_price, warning_threshold=0.05):
    """
    청산 위험도 검증
    
    Args:
        current_price (float): 현재 가격
        liquidation_price (float): 청산가
        warning_threshold (float): 경고 임계값 (기본값: 5%)
    
    Returns:
        dict: 위험도 정보
    """
    if liquidation_price <= 0:
        return {
            'risk_level': 'safe',
            'distance_to_liquidation': float('inf'),
            'distance_percentage': float('inf'),
            'warning': False
        }
    
    distance = current_price - liquidation_price
    distance_percentage = (distance / liquidation_price) * 100
    
    if current_price <= liquidation_price:
        risk_level = 'liquidated'
        warning = True
    elif distance_percentage <= warning_threshold * 100:
        risk_level = 'high'
        warning = True
    elif distance_percentage <= warning_threshold * 200:
        risk_level = 'medium'
        warning = True
    else:
        risk_level = 'low'
        warning = False
    
    return {
        'risk_level': risk_level,
        'distance_to_liquidation': distance,
        'distance_percentage': distance_percentage,
        'warning': warning
    } 

def calculate_martingale_liquidation_price(avg_price, total_position_value, leverage, initial_balance):
    """
    물타기 전략용 바이낸스 청산가 계산
    
    Args:
        avg_price (float): 평균 진입가
        total_position_value (float): 총 포지션 가치 (달러)
        leverage (float): 레버리지
        initial_balance (float): 초기 자본
    
    Returns:
        float: 청산가
    """
    if total_position_value <= 0:
        return float('inf')
    
    # 물타기 전략 청산가 공식 (롱 포지션)
    # 이론상 청산 하락률 = 100% / 레버리지
    # 청산가 = 평균가 × (1 - 이론상_청산_하락률)
    theoretical_liquidation_drop = 100 / leverage
    liquidation_price = avg_price * (1 - theoretical_liquidation_drop / 100)
    
    return max(liquidation_price, 0)  # 음수 방지 