"""
Automated Paper Trading Bot - V1 Turtle-Donchian Strategy
Runs on Binance Testnet with real-time 4H monitoring
"""
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException

from secrets import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET, validate_credentials
from config import StrategyParams
from strategy import TurtleDonchianStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('paper_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingBot:
    """Automated paper trading bot for V1 strategy"""
    
    def __init__(self, initial_capital: float = 1000.0, check_interval: int = 3600):
        """
        Args:
            initial_capital: Starting capital in USDT
            check_interval: Seconds between checks (3600 = 1 hour for 4H candles)
        """
        validate_credentials()
        
        self.initial_capital = initial_capital
        self.check_interval = check_interval
        self.killswitch_file = Path("KILLSWITCH.txt")
        
        # Initialize Binance client
        if BINANCE_TESTNET:
            self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
            logger.info("✓ Connected to Binance TESTNET")
        else:
            logger.warning("⚠️  WARNING: Connected to MAINNET - real money at risk!")
            self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        
        # Strategy setup
        self.params = StrategyParams()
        self.strategy = TurtleDonchianStrategy(self.params)
        self.symbol = "BTCUSDT"
        self.timeframe = "4h"
        
        # State
        self.equity = initial_capital
        self.position_size = 0.0  # BTC quantity
        self.position_units = []  # Track pyramid units
        self.trade_count = 0
        self.last_check_time = None
        
        logger.info(f"Bot initialized - Capital: ${initial_capital:,.2f}")
        logger.info(f"Strategy: V1 Turtle-Donchian")
        logger.info(f"Params: Entry={self.params.entry_len}, Exit={self.params.exit_len}, Trail={self.params.trail_mult}")
        
    def check_killswitch(self) -> bool:
        """Check if killswitch file exists"""
        if self.killswitch_file.exists():
            logger.critical("🛑 KILLSWITCH ACTIVATED - Stopping bot")
            return True
        return False
    
    def get_recent_candles(self, limit: int = 200) -> pd.DataFrame:
        """Fetch recent 4H candles from Binance"""
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch candles: {e}")
            raise
    
    def calculate_signals(self, df: pd.DataFrame) -> dict:
        """Calculate V1 strategy signals"""
        # Donchian channels
        entry_high = df['high'].rolling(window=self.params.entry_len).max()
        exit_low = df['low'].rolling(window=self.params.exit_len).min()
        
        # ATR for position sizing and trailing stop
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.params.atr_len).mean()
        
        # Current bar values
        current_price = df['close'].iloc[-1]
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        current_atr = atr.iloc[-1]
        current_entry_high = entry_high.iloc[-2]  # Use previous bar to avoid look-ahead
        current_exit_low = exit_low.iloc[-2]
        
        return {
            'price': current_price,
            'atr': current_atr,
            'entry_high': current_entry_high,
            'exit_low': current_exit_low,
            'prev_high': prev_high,
            'prev_low': prev_low,
            'long_entry_signal': prev_high > current_entry_high,
            'long_exit_signal': prev_low < current_exit_low,
        }
    
    def calculate_position_size(self, price: float, atr: float) -> float:
        """Calculate position size based on 1% risk"""
        risk_amount = self.equity * self.params.risk_percent
        position_size = risk_amount / (self.params.size_stop_mult * atr)
        
        # Apply lot step rounding
        position_size = round(position_size / self.params.lot_step) * self.params.lot_step
        
        # Ensure minimum notional (Binance requires ~$10 minimum)
        min_notional = 10.0 / price
        if position_size < min_notional:
            position_size = min_notional
        
        return position_size
    
    def place_market_order(self, side: str, quantity: float) -> dict:
        """Place market order on Binance testnet"""
        try:
            # Round quantity to valid precision (Binance requires specific decimals)
            quantity = round(quantity, 5)
            
            logger.info(f"Placing {side} order: {quantity} BTC")
            
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"✓ Order filled: {order['orderId']}")
            
            # Get fill price
            fill_price = float(order['fills'][0]['price']) if order.get('fills') else float(order['price'])
            
            return {
                'order_id': order['orderId'],
                'price': fill_price,
                'quantity': float(order['executedQty']),
                'commission': sum(float(f['commission']) for f in order.get('fills', []))
            }
            
        except BinanceAPIException as e:
            logger.error(f"Order failed: {e}")
            raise
    
    def enter_position(self, signals: dict):
        """Enter long position"""
        if len(self.position_units) >= self.params.max_units:
            logger.info("Max units reached - no new entry")
            return
        
        size = self.calculate_position_size(signals['price'], signals['atr'])
        
        try:
            order = self.place_market_order('BUY', size)
            
            self.position_size += order['quantity']
            self.position_units.append({
                'entry_price': order['price'],
                'quantity': order['quantity'],
                'entry_time': datetime.now(),
                'trailing_stop': order['price'] - (self.params.trail_mult * signals['atr'])
            })
            
            self.trade_count += 1
            
            logger.info(f"🟢 LONG ENTRY #{len(self.position_units)}")
            logger.info(f"   Price: ${order['price']:,.2f}")
            logger.info(f"   Quantity: {order['quantity']:.5f} BTC")
            logger.info(f"   Total Position: {self.position_size:.5f} BTC")
            logger.info(f"   Trailing Stop: ${self.position_units[-1]['trailing_stop']:,.2f}")
            
        except Exception as e:
            logger.error(f"Failed to enter position: {e}")
    
    def exit_position(self, signals: dict, reason: str):
        """Exit all position units"""
        if self.position_size <= 0:
            return
        
        try:
            order = self.place_market_order('SELL', self.position_size)
            
            # Calculate P&L
            total_cost = sum(u['entry_price'] * u['quantity'] for u in self.position_units)
            total_value = order['price'] * order['quantity']
            pnl = total_value - total_cost
            pnl_pct = (pnl / total_cost) * 100
            
            self.equity += pnl
            
            logger.info(f"🔴 LONG EXIT - {reason}")
            logger.info(f"   Exit Price: ${order['price']:,.2f}")
            logger.info(f"   Quantity: {order['quantity']:.5f} BTC")
            logger.info(f"   P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            logger.info(f"   New Equity: ${self.equity:,.2f}")
            
            # Reset position
            self.position_size = 0.0
            self.position_units = []
            
        except Exception as e:
            logger.error(f"Failed to exit position: {e}")
    
    def update_trailing_stops(self, current_price: float):
        """Update trailing stops for all units"""
        for unit in self.position_units:
            # Update trailing stop if price moved favorably
            new_stop = current_price - (self.params.trail_mult * unit.get('atr', 0))
            if new_stop > unit['trailing_stop']:
                unit['trailing_stop'] = new_stop
    
    def check_trailing_stops(self, current_price: float) -> bool:
        """Check if any trailing stop hit"""
        for unit in self.position_units:
            if current_price <= unit['trailing_stop']:
                return True
        return False
    
    def run_cycle(self):
        """Run one trading cycle"""
        logger.info("=" * 70)
        logger.info(f"Checking signals - {datetime.now()}")
        
        # Fetch data
        df = self.get_recent_candles(limit=200)
        signals = self.calculate_signals(df)
        
        logger.info(f"BTC Price: ${signals['price']:,.2f}")
        logger.info(f"ATR: ${signals['atr']:,.2f}")
        logger.info(f"Entry High: ${signals['entry_high']:,.2f}")
        logger.info(f"Exit Low: ${signals['exit_low']:,.2f}")
        logger.info(f"Current Equity: ${self.equity:,.2f}")
        logger.info(f"Position: {self.position_size:.5f} BTC ({len(self.position_units)} units)")
        
        # Check exits first
        if self.position_size > 0:
            # Donchian exit
            if signals['long_exit_signal']:
                self.exit_position(signals, "Donchian Exit")
            # Trailing stop
            elif self.check_trailing_stops(signals['price']):
                self.exit_position(signals, "Trailing Stop")
            else:
                # Update trailing stops
                self.update_trailing_stops(signals['price'])
        
        # Check entries
        if signals['long_entry_signal']:
            if self.position_size == 0:
                self.enter_position(signals)
            elif len(self.position_units) < self.params.max_units:
                # Pyramid if price moved favorably
                last_entry = self.position_units[-1]['entry_price']
                pyramid_threshold = last_entry + (self.params.pyramid_spacing_n * signals['atr'])
                if signals['price'] >= pyramid_threshold:
                    logger.info(f"Pyramid signal: price ${signals['price']:,.2f} >= ${pyramid_threshold:,.2f}")
                    self.enter_position(signals)
        
        logger.info(f"Total Trades: {self.trade_count}")
        logger.info("=" * 70)
    
    def run(self):
        """Main bot loop"""
        logger.info("\n" + "=" * 70)
        logger.info("🤖 PAPER TRADING BOT STARTED")
        logger.info("=" * 70)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Check Interval: {self.check_interval}s ({self.check_interval/3600:.1f}h)")
        logger.info(f"Testnet: {BINANCE_TESTNET}")
        logger.info(f"Killswitch: Create '{self.killswitch_file}' to stop")
        logger.info("=" * 70 + "\n")
        
        try:
            while True:
                # Check killswitch
                if self.check_killswitch():
                    # Emergency exit if in position
                    if self.position_size > 0:
                        logger.warning("Closing position before shutdown...")
                        df = self.get_recent_candles(limit=50)
                        signals = self.calculate_signals(df)
                        self.exit_position(signals, "Killswitch Shutdown")
                    break
                
                # Run trading cycle
                try:
                    self.run_cycle()
                except Exception as e:
                    logger.error(f"Cycle error: {e}", exc_info=True)
                
                # Wait for next interval
                logger.info(f"Sleeping {self.check_interval}s until next check...\n")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
            # Emergency exit
            if self.position_size > 0:
                logger.warning("Closing position before shutdown...")
                df = self.get_recent_candles(limit=50)
                signals = self.calculate_signals(df)
                self.exit_position(signals, "Manual Shutdown")
        
        # Final stats
        logger.info("\n" + "=" * 70)
        logger.info("📊 FINAL STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Equity: ${self.equity:,.2f}")
        total_return = ((self.equity / self.initial_capital) - 1) * 100
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Total Trades: {self.trade_count}")
        logger.info("=" * 70)


if __name__ == "__main__":
    # Bot configuration
    INITIAL_CAPITAL = 1000.0  # Start with $1000 USDT
    CHECK_INTERVAL = 3600  # Check every hour (4H candles update every 4 hours, but check hourly to catch signals)
    
    # Create and run bot
    bot = PaperTradingBot(
        initial_capital=INITIAL_CAPITAL,
        check_interval=CHECK_INTERVAL
    )
    
    bot.run()
