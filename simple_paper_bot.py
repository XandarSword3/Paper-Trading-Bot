"""
Lightweight Paper Trading Bot - V1 Strategy
No heavy dependencies, direct Binance integration
"""
import time
import logging
from datetime import datetime
from pathlib import Path
from binance.client import Client
from binance.exceptions import BinanceAPIException

from secrets import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET, validate_credentials

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('paper_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimplePaperBot:
    """Lightweight paper trading bot"""
    
    def __init__(self, initial_capital: float = 1000.0):
        validate_credentials()
        
        self.initial_capital = initial_capital
        self.killswitch_file = Path("KILLSWITCH.txt")
        
        # Connect to Binance
        if BINANCE_TESTNET:
            self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
            logger.info("Connected to Binance TESTNET")
        else:
            logger.warning("WARNING: Connected to MAINNET!")
            self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        
        # V1 Parameters
        self.entry_len = 40
        self.exit_len = 16
        self.atr_len = 20
        self.trail_mult = 4.0
        self.risk_pct = 0.01
        self.max_units = 4
        
        # State
        self.equity = initial_capital
        self.position_size = 0.0
        self.position_units = []
        self.entry_high = None
        self.exit_low = None
        self.current_atr = None
        
        logger.info(f"Bot initialized - Capital: ${initial_capital:,.2f}")
        logger.info(f"Params: Entry={self.entry_len}, Exit={self.exit_len}, Trail={self.trail_mult}")
    
    def check_killswitch(self):
        if self.killswitch_file.exists():
            logger.critical("KILLSWITCH - Stopping")
            return True
        return False
    
    def get_candles(self, limit=200):
        """Fetch 4H candles"""
        try:
            klines = self.client.get_klines(symbol="BTCUSDT", interval="4h", limit=limit)
            
            # Parse to simple list of dicts
            candles = []
            for k in klines:
                candles.append({
                    'time': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            
            return candles
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch candles: {e}")
            return []
    
    def calculate_indicators(self, candles):
        """Calculate Donchian and ATR"""
        if len(candles) < max(self.entry_len, self.atr_len) + 1:
            return False
        
        # Entry high (40-period max of highs)
        entry_highs = [c['high'] for c in candles[-(self.entry_len+1):-1]]
        self.entry_high = max(entry_highs)
        
        # Exit low (16-period min of lows)
        exit_lows = [c['low'] for c in candles[-(self.exit_len+1):-1]]
        self.exit_low = min(exit_lows)
        
        # ATR calculation
        trs = []
        for i in range(len(candles) - self.atr_len, len(candles)):
            if i < 1:
                continue
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)
        
        self.current_atr = sum(trs) / len(trs) if trs else 0
        
        return True
    
    def place_order(self, side, quantity):
        """Place market order"""
        try:
            quantity = round(quantity, 5)
            
            order = self.client.create_order(
                symbol="BTCUSDT",
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            fill_price = float(order['fills'][0]['price']) if order.get('fills') else 0
            
            logger.info(f"Order filled: {side} {quantity} BTC @ ${fill_price:,.2f}")
            
            return {
                'price': fill_price,
                'quantity': float(order['executedQty'])
            }
        except BinanceAPIException as e:
            logger.error(f"Order failed: {e}")
            return None
    
    def enter_long(self, price, atr):
        """Enter long position"""
        if len(self.position_units) >= self.max_units:
            return
        
        # Position size: 1% risk / (2*ATR)
        risk_amount = self.equity * self.risk_pct
        size = risk_amount / (2.0 * atr)
        size = max(size, 10.0 / price)  # Min $10 notional
        size = round(size, 5)
        
        order = self.place_order('BUY', size)
        if not order:
            return
        
        self.position_size += order['quantity']
        self.position_units.append({
            'entry_price': order['price'],
            'quantity': order['quantity'],
            'trailing_stop': order['price'] - (self.trail_mult * atr)
        })
        
        logger.info(f"LONG ENTRY #{len(self.position_units)}")
        logger.info(f"   Price: ${order['price']:,.2f}")
        logger.info(f"   Size: {order['quantity']:.5f} BTC")
        logger.info(f"   Stop: ${self.position_units[-1]['trailing_stop']:,.2f}")
    
    def exit_long(self, price, reason):
        """Exit all units"""
        if self.position_size <= 0:
            return
        
        order = self.place_order('SELL', self.position_size)
        if not order:
            return
        
        # Calculate P&L
        total_cost = sum(u['entry_price'] * u['quantity'] for u in self.position_units)
        total_value = order['price'] * order['quantity']
        pnl = total_value - total_cost
        pnl_pct = (pnl / total_cost) * 100
        
        self.equity += pnl
        
        logger.info(f"EXIT - {reason}")
        logger.info(f"   Price: ${order['price']:,.2f}")
        logger.info(f"   P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"   Equity: ${self.equity:,.2f}")
        
        self.position_size = 0.0
        self.position_units = []
    
    def run_cycle(self):
        """One trading cycle"""
        logger.info("=" * 70)
        logger.info(f"Checking - {datetime.now()}")
        
        # Get data
        candles = self.get_candles(limit=200)
        if not candles:
            logger.error("Failed to fetch candles")
            return
        
        if not self.calculate_indicators(candles):
            logger.error("Not enough candles")
            return
        
        current_price = candles[-1]['close']
        prev_high = candles[-2]['high']
        prev_low = candles[-2]['low']
        
        logger.info(f"Price: ${current_price:,.2f}")
        logger.info(f"ATR: ${self.current_atr:,.2f}")
        logger.info(f"Entry High: ${self.entry_high:,.2f}")
        logger.info(f"Exit Low: ${self.exit_low:,.2f}")
        logger.info(f"Equity: ${self.equity:,.2f}")
        logger.info(f"Position: {self.position_size:.5f} BTC")
        
        # Check exits
        if self.position_size > 0:
            # Donchian exit
            if prev_low < self.exit_low:
                self.exit_long(current_price, "Donchian Exit")
            # Trailing stop
            elif any(current_price <= u['trailing_stop'] for u in self.position_units):
                self.exit_long(current_price, "Trailing Stop")
            else:
                # Update stops
                for u in self.position_units:
                    new_stop = current_price - (self.trail_mult * self.current_atr)
                    if new_stop > u['trailing_stop']:
                        u['trailing_stop'] = new_stop
        
        # Check entry
        if prev_high > self.entry_high:
            if self.position_size == 0:
                self.enter_long(current_price, self.current_atr)
        
        logger.info("=" * 70)
    
    def run(self):
        """Main loop"""
        logger.info("\n" + "=" * 70)
        logger.info("PAPER TRADING BOT STARTED")
        logger.info("=" * 70)
        logger.info(f"Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Testnet: {BINANCE_TESTNET}")
        logger.info("=" * 70 + "\n")
        
        try:
            while True:
                if self.check_killswitch():
                    if self.position_size > 0:
                        candles = self.get_candles(50)
                        if candles:
                            self.exit_long(candles[-1]['close'], "Killswitch")
                    break
                
                try:
                    self.run_cycle()
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                
                logger.info("Sleeping 1 hour...\n")
                time.sleep(3600)
                
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
            if self.position_size > 0:
                candles = self.get_candles(50)
                if candles:
                    self.exit_long(candles[-1]['close'], "Manual Stop")
        
        # Final stats
        logger.info("\n" + "=" * 70)
        logger.info("FINAL STATS")
        logger.info("=" * 70)
        logger.info(f"Initial: ${self.initial_capital:,.2f}")
        logger.info(f"Final: ${self.equity:,.2f}")
        ret = ((self.equity / self.initial_capital) - 1) * 100
        logger.info(f"Return: {ret:+.2f}%")
        logger.info("=" * 70)


if __name__ == "__main__":
    bot = SimplePaperBot(initial_capital=1000.0)
    bot.run()
