import uuid
import datetime
import statistics
from sqlalchemy import func
from sqlalchemy.orm import Session
from algomlb.db.models import GameResultORM, LiveOddsORM, BankrollLedgerORM
from algomlb.domain import TransactionStatus, GameStatus
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ui import utils as ui_utils

# A sync run inserts one live_odds row per bookmaker within milliseconds of each
# other, not at one shared timestamp -- so "the same batch" is a short trailing
# window, not an exact match.
_ODDS_BATCH_WINDOW = datetime.timedelta(seconds=30)

# Max disagreement (in implied probability) tolerated across books in one batch,
# and max plausible swing between one batch and the next. A real incident hit
# both: every book on a game jumped from ~56% to ~22% implied within 3 hours
# (all books moving together, so a same-batch spread check alone wouldn't have
# caught it), then to <3% an hour later. That garbage price silently became a
# 38% "edge" bet with no validation in place.
_MAX_BOOK_SPREAD = 0.25
_MAX_LINE_MOVE = 0.25


class BettingService:
    """Automates paper trading placement and settlement."""

    def __init__(self, session: Session):
        self.session = session
        self.loader = MatchupLoader(session)

    def _implied_probs_in_batch(
        self, game_id: str, outcome: str, batch_end: datetime.datetime
    ) -> list[float]:
        """Vig-included implied probabilities from every book quoting `outcome`
        in the sync batch ending at `batch_end`."""
        prices = [
            p
            for (p,) in self.session.query(LiveOddsORM.price)
            .filter(LiveOddsORM.game_result_id == game_id)
            .filter(LiveOddsORM.market_type.in_(["moneyline", "h2h"]))
            .filter(LiveOddsORM.outcome == outcome)
            .filter(LiveOddsORM.timestamp > batch_end - _ODDS_BATCH_WINDOW)
            .filter(LiveOddsORM.timestamp <= batch_end)
            .all()
            if p and p > 1.0
        ]
        return [1.0 / p for p in prices]

    def _get_market_quote(
        self, game_id: str, outcome: str, cutoff: datetime.datetime
    ) -> float | None:
        """Median implied win probability for `outcome` as of the most recent
        odds batch at/before `cutoff`, or None if the data looks unreliable.

        Rejects (with a logged reason) whenever: fewer than 2 books are
        quoting, the books in the latest batch disagree with each other by
        more than `_MAX_BOOK_SPREAD`, or the consensus swung by more than
        `_MAX_LINE_MOVE` versus the prior batch.
        """
        latest_ts = (
            self.session.query(func.max(LiveOddsORM.timestamp))
            .filter(LiveOddsORM.game_result_id == game_id)
            .filter(LiveOddsORM.market_type.in_(["moneyline", "h2h"]))
            .filter(LiveOddsORM.outcome == outcome)
            .filter(LiveOddsORM.timestamp <= cutoff)
            .scalar()
        )
        if latest_ts is None:
            return None

        latest_implieds = self._implied_probs_in_batch(game_id, outcome, latest_ts)
        if len(latest_implieds) < 2:
            print(
                f"⚠️ Only {len(latest_implieds)} book(s) quoting {outcome} for "
                f"game {game_id}; skipping (need consensus)."
            )
            return None

        spread = max(latest_implieds) - min(latest_implieds)
        if spread > _MAX_BOOK_SPREAD:
            print(
                f"⚠️ Books disagree on {outcome} for game {game_id} "
                f"(spread {spread:.2f}); skipping."
            )
            return None

        latest_median = statistics.median(latest_implieds)

        prev_ts = (
            self.session.query(func.max(LiveOddsORM.timestamp))
            .filter(LiveOddsORM.game_result_id == game_id)
            .filter(LiveOddsORM.market_type.in_(["moneyline", "h2h"]))
            .filter(LiveOddsORM.outcome == outcome)
            .filter(LiveOddsORM.timestamp <= latest_ts - _ODDS_BATCH_WINDOW)
            .scalar()
        )
        if prev_ts is not None:
            prev_implieds = self._implied_probs_in_batch(game_id, outcome, prev_ts)
            if len(prev_implieds) >= 2:
                prev_median = statistics.median(prev_implieds)
                move = abs(latest_median - prev_median)
                if move > _MAX_LINE_MOVE:
                    print(
                        f"⚠️ {outcome} line for game {game_id} swung {move:.2f} "
                        f"between batches ({prev_median:.2f} -> {latest_median:.2f}); "
                        "looks like bad data, skipping."
                    )
                    return None

        return latest_median

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

                model_prob, used_fallback = ui_utils.get_uranium_prediction(ctx)
                if used_fallback:
                    print(
                        f"⚠️ Game {game.game_id}: prediction fell back to Elo-only "
                        "(no model file, or feature data >3 days stale). Skipping bet "
                        "-- too low-fidelity to trade on."
                    )
                    continue

                # Market Odds (Strictly Pre-Game for CLV, Vig Removed)
                game_id = str(game.game_id)
                h_implied_raw = self._get_market_quote(
                    game_id, game.home_team, game.game_datetime
                )
                a_implied_raw = self._get_market_quote(
                    game_id, game.away_team, game.game_datetime
                )

                if h_implied_raw is not None and a_implied_raw is not None:
                    total_implied = h_implied_raw + a_implied_raw
                    h_implied = (
                        h_implied_raw / total_implied if total_implied > 0 else 0.5
                    )
                elif h_implied_raw is not None:
                    h_implied = h_implied_raw
                elif a_implied_raw is not None:
                    h_implied = 1.0 - a_implied_raw
                else:
                    h_implied = None

                if h_implied is not None:
                    edge = model_prob - h_implied

                    # --- NEW: Archive ALL Predictions for CLV Analysis ---
                    from algomlb.db.models import ModelPredictionORM

                    archive = ModelPredictionORM(
                        game_id=game_id,
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
                        selection_implied_raw = (
                            h_implied_raw if edge > 0 else a_implied_raw
                        )

                        if selection_implied_raw:
                            final_odds = 1.0 / selection_implied_raw
                        else:
                            final_odds = (
                                1.0 / (1.0 - h_implied) if h_implied < 1 else 0
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
                            game_id=game_id,
                        )
                        self.session.add(bet)
                        placed_count += 1
            except Exception as e:
                print(f"Failed to place bet for game {game.game_id}: {e}")

        self.session.commit()
        return placed_count

    # MLB's score endpoint reports 0-0 for a game that hasn't started (or is
    # mid-inning) just as readily as for a real final, so "both scores
    # present" is never on its own proof a game is over -- and never for a
    # game still IN_PROGRESS. This narrow allowance exists only for the
    # observed bug where status gets stuck on SCHEDULED despite a real final
    # score: a long-stale SCHEDULED row with a decisive (non-tied) score.
    _STALE_SCHEDULED_GRACE = datetime.timedelta(hours=6)

    def settle_bets(self):
        """Check results for PENDING and PLACED bets and calculate P&L.

        Also voids bets on games that will never produce a result under this
        game_id: POSTPONED/CANCELLED games left unresolved well past their
        original start time (MLB reschedules these as a new game_id, so the
        original bet has nothing left to settle against). Left unhandled,
        these sit as PENDING forever and skew recap/P&L stats.
        """
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
        now = datetime.datetime.now(datetime.UTC)
        for bet in pending_bets:
            game = (
                self.session.query(GameResultORM)
                .filter(GameResultORM.game_id == bet.game_id)
                .first()
            )
            if not game:
                continue

            is_completed = game.status == GameStatus.COMPLETED
            if not is_completed and game.status == GameStatus.SCHEDULED:
                stale = (
                    game.game_datetime is not None
                    and (now - game.game_datetime) > self._STALE_SCHEDULED_GRACE
                )
                decisive_score = (
                    game.home_score is not None
                    and game.away_score is not None
                    and game.home_score != game.away_score
                )
                is_completed = stale and decisive_score

            if is_completed:
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
                continue

            if (
                game.status in (GameStatus.POSTPONED, GameStatus.CANCELLED)
                and game.game_datetime
                and (now - game.game_datetime) > datetime.timedelta(days=2)
            ):
                bet.pnl = 0.0
                bet.status = TransactionStatus.CANCELLED
                settled_count += 1

        self.session.commit()
        return settled_count
