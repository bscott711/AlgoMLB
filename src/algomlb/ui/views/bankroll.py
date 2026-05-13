import streamlit as st
import pandas as pd
from algomlb.db.session import get_session_factory
from algomlb.db.models import BankrollLedgerORM
from algomlb.domain import TransactionStatus
from sqlalchemy import func

def render_bankroll_view():
    st.title("💰 Bankroll & Ledger")
    st.markdown("---")
    
    session_factory = get_session_factory()
    session = session_factory()
    
    try:
        import importlib
        import algomlb.db.models as models
        importlib.reload(models)
        
        from algomlb.db.models import GameResultORM
        
        # 1. Calculate Metrics
        # ... (Metrics calculation remains the same)
        total_balance = session.query(func.sum(BankrollLedgerORM.pnl)).scalar() or 0.0
        pending_count = session.query(BankrollLedgerORM).filter(BankrollLedgerORM.status == TransactionStatus.PENDING).count()
        settled_count = session.query(BankrollLedgerORM).filter(BankrollLedgerORM.status == TransactionStatus.SETTLED).count()
        
        wins = session.query(BankrollLedgerORM).filter(BankrollLedgerORM.status == TransactionStatus.SETTLED, BankrollLedgerORM.pnl > 0).count()
        win_rate = wins / settled_count if settled_count > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Balance", f"${total_balance:,.2f}")
        c2.metric("Active Bets", pending_count)
        c3.metric("Win Rate", f"{win_rate:.1%}")

        st.markdown("---")
        
        # 2. Transaction History
        st.subheader("📝 Transaction Ledger")
        
        # Join with GameResultORM to get team names
        ledger_query = (
            session.query(BankrollLedgerORM, GameResultORM)
            .outerjoin(GameResultORM, BankrollLedgerORM.game_id == GameResultORM.game_id)
            .order_by(BankrollLedgerORM.timestamp.desc())
            .all()
        )

        if ledger_query:
            data = []
            for t, g in ledger_query:
                game_name = f"{g.away_team} @ {g.home_team}" if g else "--"
                data.append({
                    "ID": t.transaction_id[:8],
                    "Time": t.timestamp.strftime("%m-%d %H:%M"),
                    "Matchup": game_name,
                    "Selection": getattr(t, 'selection', 'N/A'),
                    "Stake": f"${t.stake:.2f}",
                    "Odds": f"{t.odds:.2f}",
                    "Edge": f"{getattr(t, 'edge', 0.0):+.1%}" if getattr(t, 'edge', None) is not None else "N/A",
                    "Status": t.status.value if hasattr(t, 'status') else 'UNKNOWN',
                    "P&L": f"${t.pnl:+.2f}" if t.pnl is not None else "--"
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No transactions found in the ledger.")

        # 3. Performance Chart
        if settled_count > 0:
            st.markdown("---")
            st.subheader("📈 Equity Curve")
            # Build cumulative P&L
            settled_data = (
                session.query(BankrollLedgerORM.timestamp, BankrollLedgerORM.pnl)
                .filter(BankrollLedgerORM.pnl.isnot(None))
                .order_by(BankrollLedgerORM.timestamp)
                .all()
            )
            
            dates = [d[0] for d in settled_data]
            pnls = [d[1] for d in settled_data]
            cumulative = []
            current = 0
            for p in pnls:
                current += p
                cumulative.append(current)
            
            chart_df = pd.DataFrame({"Date": dates, "Equity": cumulative})
            st.line_chart(chart_df, x="Date", y="Equity")

    finally:
        session.close()

if __name__ == "__main__":
    render_bankroll_view()
else:
    render_bankroll_view()
