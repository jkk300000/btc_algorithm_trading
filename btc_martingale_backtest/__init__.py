"""
BTC Martingale Backtest 프로젝트

비트코인 무기한 선물을 위한 마틴게일 기반 트레이딩 전략의 백테스트 및 분석 시스템입니다.

주요 모듈:
- kalmanFilter: 칼만 필터 기반 노이즈 제거
- indicator: 기술적 지표 및 특성 엔지니어링
- strategy: 다양한 마틴게일 트레이딩 전략
- ml_model: 머신러닝 기반 방향 예측 모델
- binance: 바이낸스 API 및 데이터 처리
- analyze: 백테스트 결과 분석 도구

주요 기능:
- 실시간 데이터 수집 및 전처리
- 칼만 필터를 이용한 노이즈 제거
- 랜덤 포레스트 기반 진입 신호 생성
- 적응형 마틴게일 리스크 관리
- 종합적인 백테스트 및 성능 분석

Author: AI Assistant
Version: 1.0.0
Created: 2025-01-05
"""

from . import kalmanFilter
from . import indicator
from . import strategy
from . import ml_model
from . import binance
from . import analyze

# 주요 함수들을 top-level에서 직접 사용 가능하도록
from .indicator import add_features
from .kalmanFilter import apply_btc_kalman_filtering
from .binance import fetch_klines
from .analyze import analyze_trade_performance

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "kalmanFilter",
    "indicator", 
    "strategy",
    "ml_model",
    "binance",
    "analyze",
    "add_features",
    "apply_btc_kalman_filtering",
    "fetch_klines",
    "analyze_trade_performance"
]

def get_version():
    """프로젝트 버전 정보 반환"""
    return __version__

def list_modules():
    """사용 가능한 모듈 목록 반환"""
    return __all__[:6]  # 서브 모듈들만 반환

def quick_start():
    """빠른 시작 가이드 출력"""
    print("🚀 BTC Martingale Backtest 시스템")
    print("="*50)
    print("1. 데이터 수집: binance.fetch_klines()")
    print("2. 특성 엔지니어링: indicator.add_features()")
    print("3. 백테스트: run_backtest.py 실행")
    print("4. 결과 분석: analyze.analyze_trade_performance()")
    print("="*50)
    print("자세한 사용법은 각 모듈의 docstring을 참조하세요.")