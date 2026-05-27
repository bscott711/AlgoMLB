import sys
from fadegoblin.browser_fliff import place_fliff_bet

if __name__ == "__main__":
    print("Testing place_fliff_bet...")
    # The user asked for a 1000 Fliff Coin bet on TOR
    result = place_fliff_bet("TOR", 1000, use_coins=True)
    if result:
        print("✅ Bet was placed successfully!")
        sys.exit(0)
    else:
        print("❌ Bet placement failed.")
        sys.exit(1)
