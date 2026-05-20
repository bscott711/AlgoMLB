import uuid
import datetime
from sqlalchemy.orm import Session
from algomlb.db.models import GameResultORM, LiveOddsORM, BankrollLedgerORM
from algomlb.domain import TransactionStatus, GameStatus
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ui import utils as ui_utils


class BettingService:
    """Automates paper trading placement and settlement."""

    def __init__(self, session: Session):
        self.session = session
        self.loader = MatchupLoader(session)

    def place_daily_bets(
        self, target_date: datetime.date, min_edge: float = 0.05, stake: float = 5.0
    ):
        """Scan for +EV opportunities and place pending bets."""
        games = (
            self.session.query(GameResultORM)
            .filter(GameResultORM.game_date == target_date)
            .all()
        )

        placed_count = 0
        for game in games:
            # Check if we already have a bet for this game
            existing = (
                self.session.query(BankrollLedgerORM)
                .filter(BankrollLedgerORM.game_id == str(game.game_id))
                .first()
            )
            if existing:
                continue

            try:
                ctx = self.loader.load_matchup(int(game.game_id))
                if not ctx:
                    continue

                model_prob, _ = ui_utils.get_uranium_prediction(ctx)

                # Market Odds (Strictly Pre-Game for CLV)
                market_odds = (
                    self.session.query(LiveOddsORM)
                    .filter(LiveOddsORM.game_result_id == str(game.game_id))
                    .filter(LiveOddsORM.market_type.in_(["moneyline", "h2h"]))
                    .filter(LiveOddsORM.timestamp <= game.game_datetime)
                    .order_by(LiveOddsORM.timestamp.desc())
                    .first()
                )

                if market_odds:
                    implied_prob = (
                        1.0 / market_odds.price if market_odds.price > 0 else 0.5
                    )

                    if market_odds.outcome == game.home_team:
                        h_implied = implied_prob
                    else:
                        h_implied = 1.0 - implied_prob

                    edge = model_prob - h_implied

                    # --- NEW: Archive ALL Predictions for CLV Analysis ---
                    from algomlb.db.models import ModelPredictionORM

                    archive = ModelPredictionORM(
                        game_id=str(game.game_id),
                        game_date=target_date,
                        model_version="uranium_v1.0",
                        home_win_prob=model_prob,
                        market_home_implied_at_prediction=h_implied,
                        timestamp=datetime.datetime.now(datetime.UTC),
                    )
                    self.session.add(archive)
                    # ---------------------------------------------------

                    # Determine Selection
                    if abs(edge) >= min_edge:
                        selection = game.home_team if edge > 0 else game.away_team
                        final_odds = (
                            market_odds.price
                            if market_odds.outcome == selection
                            else (1.0 / (1.0 - implied_prob))
                        )

                        bet = BankrollLedgerORM(
                            transaction_id=str(uuid.uuid4()),
                            timestamp=datetime.datetime.now(datetime.UTC),
                            stake=stake,
                            odds=final_odds,
                            selection=selection,
                            edge=abs(edge),
                            status=TransactionStatus.PENDING,
                            pnl=None,
                            game_id=str(game.game_id),
                        )
                        self.session.add(bet)
                        placed_count += 1
            except Exception as e:
                print(f"Failed to place bet for game {game.game_id}: {e}")

        self.session.commit()
        return placed_count

    def settle_bets(self):
        """Check results for PENDING and PLACED bets and calculate P&L."""
        pending_bets = (
            self.session.query(BankrollLedgerORM)
            .filter(
                BankrollLedgerORM.status.in_(
                    [TransactionStatus.PENDING, TransactionStatus.PLACED]
                )
            )
            .all()
        )

        settled_count = 0
        for bet in pending_bets:
            game = (
                self.session.query(GameResultORM)
                .filter(GameResultORM.game_id == bet.game_id)
                .first()
            )
            if not game or game.status != GameStatus.COMPLETED:
                continue

            # Determine winner
            winner = None
            if game.home_score > game.away_score:
                winner = game.home_team
            elif game.away_score > game.home_score:
                winner = game.away_team

            if winner:
                if bet.selection == winner:
                    bet.pnl = bet.stake * (bet.odds - 1)
                else:
                    bet.pnl = -bet.stake

                bet.status = TransactionStatus.SETTLED
                settled_count += 1

        self.session.commit()
        return settled_count
