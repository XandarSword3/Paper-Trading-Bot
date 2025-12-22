"""
Test Binance API credentials
Run this to verify your API keys work before using the bot
"""
from binance.client import Client
from secrets import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET, validate_credentials

def test_connection():
    """Test Binance API connection and permissions"""
    print("=" * 70)
    print("BINANCE API CONNECTION TEST")
    print("=" * 70)
    
    # Validate credentials are set
    try:
        validate_credentials()
        print("✓ Credentials loaded from .env")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return False
    
    # Show which environment
    env = "TESTNET" if BINANCE_TESTNET else "MAINNET (LIVE)"
    print(f"Environment: {env}")
    print(f"API Key (first 8): {BINANCE_API_KEY[:8]}...")
    print()
    
    # Create client
    try:
        if BINANCE_TESTNET:
            client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
        else:
            client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        print("✓ Client created")
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        return False
    
    # Test 1: Ping
    print("\n--- Test 1: Server Ping ---")
    try:
        result = client.ping()
        print(f"✓ Ping successful: {result}")
    except Exception as e:
        print(f"✗ Ping failed: {e}")
        return False
    
    # Test 2: Server time
    print("\n--- Test 2: Server Time ---")
    try:
        result = client.get_server_time()
        print(f"✓ Server time: {result}")
    except Exception as e:
        print(f"✗ Server time failed: {e}")
        return False
    
    # Test 3: Account info
    print("\n--- Test 3: Account Access ---")
    try:
        account = client.get_account()
        print(f"✓ Account access granted")
        print(f"  Account type: {account.get('accountType', 'N/A')}")
        print(f"  Can trade: {account.get('canTrade', False)}")
        print(f"  Can deposit: {account.get('canDeposit', False)}")
        print(f"  Can withdraw: {account.get('canWithdraw', False)}")
        
        # Show first 5 non-zero balances
        balances = [b for b in account.get('balances', []) if float(b['free']) > 0 or float(b['locked']) > 0]
        if balances:
            print(f"\n  Non-zero balances (first 5):")
            for bal in balances[:5]:
                print(f"    {bal['asset']}: {bal['free']} free, {bal['locked']} locked")
        else:
            print(f"\n  No balances found (testnet account may need funding)")
            
    except Exception as e:
        print(f"✗ Account access failed: {e}")
        print(f"  Make sure 'Enable Spot & Margin Trading' is checked in API settings")
        return False
    
    # Test 4: BTC price
    print("\n--- Test 4: Market Data ---")
    try:
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        print(f"✓ BTC/USDT price: ${float(ticker['price']):,.2f}")
    except Exception as e:
        print(f"✗ Market data failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - API credentials are working")
    print("=" * 70)
    
    if not BINANCE_TESTNET:
        print("\n⚠️  WARNING: You are connected to MAINNET (LIVE)")
        print("   Set BINANCE_TESTNET=true in .env for paper trading")
    
    return True


if __name__ == "__main__":
    test_connection()
