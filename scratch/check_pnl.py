import pandas as pd
from sqlalchemy import create_engine

def main():
    engine = create_engine('postgresql+psycopg2://postgres:password@localhost:5433/algomlb')
    df = pd.read_sql('select status, pnl, stake from bankroll_ledger', engine)
    
    print("==================================================")
    print("              AlgoMLB BANKROLL LEDGER AUDIT      ")
    print("==================================================")
    print(df.groupby('status').agg({'pnl': ['sum', 'count'], 'stake': 'sum'}))
    print("\n--------------------------------------------------")
    
    settled = df[df['status'] == 'SETTLED']
    total_count = len(settled)
    
    if total_count > 0:
        wins = len(settled[settled['pnl'] > 0])
        losses = len(settled[settled['pnl'] < 0])
        pushes = len(settled[settled['pnl'] == 0])
        
        total_pnl = settled['pnl'].sum()
        total_stake = settled['stake'].sum()
        roi = (total_pnl / total_stake) * 100 if total_stake > 0 else 0
        
        print("📈 PERFORMANCE SUMMARY (SETTLED PLAYS):")
        print(f"👉 Record:                 {wins}W - {losses}L - {pushes}P")
        print(f"👉 Win Percentage:         {(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0:.1f}%")
        print(f"👉 Total Profit/Loss (PnL): ${total_pnl:+.2f}")
        print(f"👉 Total Volume Staked:    ${total_stake:.2f}")
        print(f"👉 Return on Investment:   {roi:+.2f}%")
    else:
        print("⚠️ No settled plays found in bankroll_ledger.")
        
    pending = df[df['status'] == 'PENDING']
    placed = df[df['status'] == 'PLACED']
    print(f"\n⏳ Active/Pending Bets:     {len(pending)} PENDING, {len(placed)} PLACED")
    print("==================================================")

if __name__ == '__main__':
    main()
