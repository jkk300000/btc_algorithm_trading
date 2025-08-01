"""
바이낸스 거래소 계산 도구
청산가, 마진, 레버리지 등 바이낸스 기준 계산 함수들
"""

class BinanceAveragePriceCalculator:
    """
    바이낸스 무기한 선물 평균 진입 가격 계산기
    
    바이낸스 공식:
    평균 진입 가격 = (총 매수 금액) / (총 매수 수량)
    = Σ(Pₖ × Qₖ) / Σ(Qₖ)
    
    Pₖ: 각 매수 주문의 진입 가격
    Qₖ: 각 매수 주문의 수량
    """
    
    def __init__(self):
        self.total_cost = 0.0  # 총 매수 금액
        self.total_quantity = 0.0  # 총 매수 수량
        self.positions = []  # 포지션 기록 (가격, 수량)
    
    def add_position(self, price, quantity):
        """
        새로운 포지션 추가 (매수)
        
        Args:
            price (float): 진입 가격
            quantity (float): 수량
        """
        cost = price * quantity
        self.total_cost += cost
        self.total_quantity += quantity
        self.positions.append({
            'price': price,
            'quantity': quantity,
            'cost': cost,
            'timestamp': None  # 필요시 시간 정보 추가 가능
        })
    
    def remove_position(self, quantity):
        """
        포지션 제거 (매도) - 바이낸스 방식
        매도 시 평균 진입 가격은 변경되지 않고 수량만 줄어듦
        
        Args:
            quantity (float): 제거할 수량
        """
        if quantity <= 0 or self.total_quantity <= 0:
            return
        
        # 바이낸스 방식: 매도 시 평균가 유지, 수량만 감소
        # 총 비용은 평균가 × 감소된 수량만큼 차감
        current_avg_price = self.get_average_price()
        removed_cost = current_avg_price * quantity
        
        self.total_quantity -= quantity
        self.total_cost -= removed_cost
        
        # 음수 방지
        self.total_quantity = max(0, self.total_quantity)
        self.total_cost = max(0, self.total_cost)
        
        # 포지션 기록도 업데이트 (FIFO 방식으로 수량만 조정)
        remaining_quantity = quantity
        for position in self.positions:
            if remaining_quantity <= 0:
                break
            
            if position['quantity'] <= remaining_quantity:
                # 전체 포지션 제거
                remaining_quantity -= position['quantity']
                position['quantity'] = 0
                position['cost'] = 0
            else:
                # 부분 포지션 제거
                position['quantity'] -= remaining_quantity
                position['cost'] = position['price'] * position['quantity']
                remaining_quantity = 0
        
        # 수량이 0인 포지션 제거
        self.positions = [pos for pos in self.positions if pos['quantity'] > 0]
    
    def close_all_positions(self):
        """모든 포지션 청산"""
        self.total_cost = 0.0
        self.total_quantity = 0.0
        self.positions.clear()
    
    def get_average_price(self):
        """
        현재 평균 진입 가격 반환
        
        Returns:
            float: 평균 진입 가격, 포지션이 없으면 0
        """
        if self.total_quantity <= 0:
            return 0.0
        return self.total_cost / self.total_quantity
    
    def get_total_quantity(self):
        """
        총 포지션 수량 반환
        
        Returns:
            float: 총 수량
        """
        return self.total_quantity
    
    def get_total_value(self, current_price=None):
        """
        총 포지션 가치 반환
        
        Args:
            current_price (float): 현재 가격 (None이면 평균가 사용)
        
        Returns:
            float: 총 포지션 가치
        """
        if current_price is None:
            current_price = self.get_average_price()
        return self.total_quantity * current_price
    
    def calculate_pnl(self, current_price):
        """
        현재가 기준 손익 계산
        
        Args:
            current_price (float): 현재 가격
        
        Returns:
            dict: 손익 정보
        """
        if self.total_quantity <= 0:
            return {
                'unrealized_pnl': 0.0,
                'unrealized_pnl_percent': 0.0,
                'average_price': 0.0,
                'current_value': 0.0,
                'total_cost': 0.0
            }
        
        current_value = self.total_quantity * current_price
        unrealized_pnl = current_value - self.total_cost
        unrealized_pnl_percent = (unrealized_pnl / self.total_cost) * 100 if self.total_cost > 0 else 0
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_percent': unrealized_pnl_percent,
            'average_price': self.get_average_price(),
            'current_value': current_value,
            'total_cost': self.total_cost
        }
    
    def get_position_history(self):
        """
        포지션 기록 반환
        
        Returns:
            list: 포지션 기록 리스트
        """
        return self.positions.copy()
    
    def reset(self):
        """계산기 초기화"""
        self.total_cost = 0.0
        self.total_quantity = 0.0
        self.positions.clear()

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