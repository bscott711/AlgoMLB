import streamlit as st
import pandas as pd
import json
from algomlb.db.session import get_session_factory
from algomlb.db.models import BankrollLedgerORM
from algomlb.domain import TransactionStatus
from sqlalchemy import func, text

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
        
        tab1, tab2 = st.tabs(["💰 AlgoMLB Portfolio", "💚 FadeGoblin Degeneracy"])
        
        with tab1:
            # 1. Calculate Metrics
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

        with tab2:
            st.subheader("🎲 Degen Paper Trading Ledger")
            st.markdown("Tracks the performance of random parlays and Play of the Day (POTD) slips separately with isolated bankroll management.")
            
            # Fetch degen/potd slips
            slips_query = text("SELECT slip_id, slip_type, legs, final_odds, stake, status, pnl, created_at, settled_at FROM fadegoblin_slips ORDER BY created_at DESC;")
            
            # We wrap execution to handle case where table hasn't been created yet
            try:
                slips = session.execute(slips_query).fetchall()
            except Exception:
                slips = []
            
            if slips:
                # Calculate degen metrics
                degen_starting = 1000.00
                total_pnl = sum([float(s[6]) for s in slips if s[6] is not None])
                current_balance = degen_starting + total_pnl
                
                settled_slips = [s for s in slips if s[5] == "SETTLED"]
                settled_count = len(settled_slips)
                pending_count = len([s for s in slips if s[5] == "PENDING"])
                
                wins = len([s for s in settled_slips if float(s[6]) > 0])
                win_rate = wins / settled_count if settled_count > 0 else 0.0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Degen Balance", f"${current_balance:,.2f}", f"${total_pnl:+,.2f} Net")
                col2.metric("Active Slips", pending_count)
                col3.metric("Win Rate", f"{win_rate:.1%}")
                
                st.markdown("---")
                st.subheader("📝 FadeGoblin Slips")
                
                table_data = []
                for s in slips:
                    legs_list = s[2] if isinstance(s[2], list) else json.loads(s[2])
                    legs_summary = " | ".join([f"{leg.get('pick', 'N/A')} ({leg.get('odds', 'N/A')})" for leg in legs_list])
                    
                    table_data.append({
                        "ID": s[0],
                        "Type": s[1].upper(),
                        "Created": s[7].strftime("%m-%d %H:%M") if s[7] else "--",
                        "Selection / Legs": legs_summary,
                        "Odds": s[3],
                        "Stake": f"${float(s[4]):.2f}",
                        "Status": s[5],
                        "P&L": f"${float(s[6]):+.2f}" if s[6] is not None else "--"
                    })
                
                df_slips = pd.DataFrame(table_data)
                st.dataframe(df_slips, use_container_width=True)
                
                # Equity Curve
                if settled_count > 0:
                    st.markdown("---")
                    st.subheader("📈 Degen Equity Curve")
                    
                    settled_chronological = sorted(settled_slips, key=lambda x: x[7] if x[7] else x[8])
                    dates = [s[7] for s in settled_chronological]
                    pnls = [float(s[6]) for s in settled_chronological if float(s[6]) is not None]
                    
                    cumulative = []
                    current = degen_starting
                    for p in pnls:
                        current += p
                        cumulative.append(current)
                        
                    chart_df = pd.DataFrame({"Date": dates, "Balance": cumulative})
                    st.line_chart(chart_df, x="Date", y="Balance")
            else:
                st.info("No FadeGoblin slips found in the ledger. Once the bot posts a degen parlay or POTD preview, they will show up here!")

    finally:
        session.close()

if __name__ == "__main__":
    render_bankroll_view()
