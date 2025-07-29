import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

BINANCE_TIMEZONE = timezone.utc
BINANCE_FUTURES_START = '2019-09-13T00:00:00Z'

class BinanceFuturesFetcher:
    def __init__(self, symbol='BTC/USDT:USDT', timeframe='1m', max_limit=1500, rate_limit=0.5):
        load_dotenv()
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_limit = max_limit  # 바이낸스 1회 최대 1500개
        self.rate_limit = rate_limit  # 초당 요청 제한
        api_key = os.getenv('BINANCE_API_KEY')
        secret = os.getenv('BINANCE_SECRET_KEY')
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,  # 이 줄 추가!
            },
        })

    def fetch_ohlcv(self, since=None, until=None):
        all_ohlcv = []
        # 무기한 선물 시작일 적용
        since = since or self.exchange.parse8601(BINANCE_FUTURES_START)
        now = until or int(datetime.now(BINANCE_TIMEZONE).timestamp() * 1000)
        while since < now:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, self.timeframe, since=since, limit=self.max_limit
                )
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                print(f"Fetched {len(all_ohlcv)} rows, last timestamp: {datetime.utcfromtimestamp(last_ts/1000).strftime('%Y-%m-%d %H:%M:%S')} UTC")
                # 중복 방지 위해 +1ms
                since = last_ts + 1
                # 마지막 캔들이 until을 넘으면 종료
                if since >= now:
                    break
                time.sleep(self.rate_limit)
            except Exception as e:
                print(f"Error: {e}, retrying in 5 seconds...")
                time.sleep(5)
        df = pd.DataFrame(
            all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        # df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df

    def save_to_csv(self, df, filename):
        df.to_csv(filename)
        print(f'Saved to {filename}')

if __name__ == '__main__':
    fetcher = BinanceFuturesFetcher()
    df = fetcher.fetch_ohlcv()
    fetcher.save_to_csv(df, 'C:/선물데이터/binance_btcusdt_1m.csv') 