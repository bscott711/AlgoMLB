import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from algomlb.db.session import Base
from algomlb.domain import GameStatus, TransactionStatus, GameType


class LiveOddsORM(Base):
    """Volatile time-series table for active games/odds."""

    __tablename__ = "live_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    odds_game_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(50), nullable=False)
    away_team: Mapped[str] = mapped_column(String(50), nullable=False)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    game_result_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )
    sportsbook: Mapped[str] = mapped_column(String(50), nullable=False)
    market_type: Mapped[str] = mapped_column(String(50), nullable=False)
    outcome: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class GameResultORM(Base):
    """Storage for completed/settled games and historical results."""

    __tablename__ = "game_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(
        String(50), nullable=False, unique=True, index=True
    )  # MLB Game PK
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    game_datetime: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    home_team: Mapped[str] = mapped_column(String(50), nullable=False)
    away_team: Mapped[str] = mapped_column(String(50), nullable=False)
    home_pitcher: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    away_pitcher: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    home_pitcher_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_pitcher_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_team_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_team_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[GameStatus] = mapped_column(
        Enum(GameStatus), nullable=False, default=GameStatus.SCHEDULED
    )
    game_type: Mapped[Optional[GameType]] = mapped_column(
        Enum(GameType), nullable=True, default=GameType.REGULAR_SEASON
    )
    ballpark_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Environmental Context
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Travel / Rest Fatigue
    home_rest_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_rest_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class BankrollLedgerORM(Base):
    """Persistent state for the paper bankroll."""

    __tablename__ = "bankroll_ledger"

    transaction_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    stake: Mapped[float] = mapped_column(Float, nullable=False)
    odds: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus), nullable=False, default=TransactionStatus.PENDING
    )
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    game_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )


class PitchEventORM(Base):
    """Statcast pitch-level data; one row per pitch."""

    __tablename__ = "pitch_events"
    __table_args__ = (
        UniqueConstraint(
            "game_id",
            "at_bat_number",
            "pitch_number",
            name="uq_pitch_events_game_ab_pitch",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at_bat_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    game_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    batter_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    release_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    release_spin_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pfx_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pfx_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plate_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plate_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    launch_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    launch_angle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Added Statcast Features
    pitch_type: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    stand: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)
    p_throws: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    events: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    release_extension: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    effective_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pitch_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    inning: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    zone: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bb_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)


class HistoricalOddsORM(Base):
    """Storage for historical opening and closing odds."""

    __tablename__ = "historical_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    market_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # h2h, spreads, totals
    odds_type: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # opening, closing
    home_price: Mapped[int] = mapped_column(Integer, nullable=False)  # American odds
    away_price: Mapped[int] = mapped_column(Integer, nullable=False)
    spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snapshot_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    fetched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.now
    )


class BallparkORM(Base):
    """Structural data for MLB ballparks from Kaggle mlb-ballparks dataset."""

    __tablename__ = "ballparks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ballpark: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    city: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    state: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    left_field: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    center_field: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    right_field: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    min_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hr_park_effects: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extra_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_temp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    roof: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    daytime: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hp_bearing_deg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class UmpireScorecardORM(Base):
    """Umpire accuracy and bias data from umpscorecards.us API."""

    __tablename__ = "umpire_scorecards"
    __table_args__ = (UniqueConstraint("game_pk", name="uq_umpire_scorecards_game_pk"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    game_type: Mapped[Optional[GameType]] = mapped_column(Enum(GameType), nullable=True)
    umpire_name: Mapped[str] = mapped_column(String(100), nullable=False)
    home_team: Mapped[str] = mapped_column(String(5), nullable=False)
    away_team: Mapped[str] = mapped_column(String(5), nullable=False)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Accuracy Metrics
    called_pitches: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    called_correct: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    called_wrong: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    x_overall_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accuracy_above_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_x_correct_calls: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    x_correct_calls: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    correct_calls_above_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Consistency & Favor
    consistency: Mapped[float] = mapped_column(Float, nullable=False)
    favoritism_home: Mapped[float] = mapped_column(Float, nullable=False)

    # Per-Side Impact
    home_batter_impact: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_pitcher_impact: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_batter_impact: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_pitcher_impact: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_run_impact: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_runs: Mapped[float] = mapped_column(Float, nullable=False)
    actual_runs: Mapped[float] = mapped_column(Float, nullable=False)

    # Challenge / ABS Metrics
    n_overturned: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_challenged: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    challenge_success_rate: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    n_overturned_home: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_challenged_home: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_overturned_away: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_challenged_away: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # ABS Zone Data
    abs_away_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    abs_away_b: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    abs_away_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    abs_away_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    abs_home_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    abs_home_b: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    abs_home_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    abs_home_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata Flags
    fully_valid: Mapped[Optional[bool]] = mapped_column(nullable=True)
    num_pitches_no_data: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class RetrosheetEventORM(Base):
    """Granular play-by-play event record from Retrosheet files."""

    __tablename__ = "retrosheet_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # gid
    play_number: Mapped[int] = mapped_column(Integer, nullable=False)  # pn
    event_text: Mapped[str] = mapped_column(String(255))  # event
    inning: Mapped[int] = mapped_column(Integer, nullable=False)
    top_bot: Mapped[int] = mapped_column(Integer, nullable=False)
    vis_home: Mapped[int] = mapped_column(Integer)
    site: Mapped[str] = mapped_column(String(10))
    bat_team: Mapped[str] = mapped_column(String(5))  # batteam
    pit_team: Mapped[str] = mapped_column(String(5))  # pitteam
    score_v: Mapped[int] = mapped_column(Integer, default=0)
    score_h: Mapped[int] = mapped_column(Integer, default=0)

    # Participants
    batter_id: Mapped[str] = mapped_column(String(20))  # batter
    pitcher_id: Mapped[str] = mapped_column(String(20))  # pitcher
    lp: Mapped[int] = mapped_column(Integer)  # lineup position
    bat_f: Mapped[int] = mapped_column(Integer)  # fielding pos of batter
    batter_hand: Mapped[Optional[str]] = mapped_column(String(1))  # bathand
    pitcher_hand: Mapped[Optional[str]] = mapped_column(String(1))  # pithand

    # Counts & Sequence
    balls: Mapped[int] = mapped_column(Integer, default=0)
    strikes: Mapped[int] = mapped_column(Integer, default=0)
    count_text: Mapped[Optional[str]] = mapped_column(String(5))  # count
    pitches: Mapped[Optional[str]] = mapped_column(String(50))
    nump: Mapped[Optional[int]] = mapped_column(Integer)

    # Outcomes (The 14 flags)
    pa_flag: Mapped[int] = mapped_column(Integer, default=0)
    ab_flag: Mapped[int] = mapped_column(Integer, default=0)
    single: Mapped[int] = mapped_column(Integer, default=0)
    double_flag: Mapped[int] = mapped_column(Integer, default=0)  # double is a keyword
    triple: Mapped[int] = mapped_column(Integer, default=0)
    hr: Mapped[int] = mapped_column(Integer, default=0)
    sh: Mapped[int] = mapped_column(Integer, default=0)
    sf: Mapped[int] = mapped_column(Integer, default=0)
    hbp: Mapped[int] = mapped_column(Integer, default=0)
    walk: Mapped[int] = mapped_column(Integer, default=0)
    k: Mapped[int] = mapped_column(Integer, default=0)
    xi: Mapped[int] = mapped_column(Integer, default=0)
    roe: Mapped[int] = mapped_column(Integer, default=0)
    fc: Mapped[int] = mapped_column(Integer, default=0)
    othout: Mapped[int] = mapped_column(Integer, default=0)
    noout: Mapped[int] = mapped_column(Integer, default=0)

    # BIP Details
    bip: Mapped[int] = mapped_column(Integer, default=0)
    bunt: Mapped[int] = mapped_column(Integer, default=0)
    ground: Mapped[int] = mapped_column(Integer, default=0)
    fly: Mapped[int] = mapped_column(Integer, default=0)
    line_flag: Mapped[int] = mapped_column(Integer, default=0)
    iw: Mapped[int] = mapped_column(Integer, default=0)
    gdp: Mapped[int] = mapped_column(Integer, default=0)
    othdp: Mapped[int] = mapped_column(Integer, default=0)
    tp: Mapped[int] = mapped_column(Integer, default=0)
    fle: Mapped[int] = mapped_column(Integer, default=0)
    wp: Mapped[int] = mapped_column(Integer, default=0)
    pb: Mapped[int] = mapped_column(Integer, default=0)
    bk: Mapped[int] = mapped_column(Integer, default=0)
    oa: Mapped[int] = mapped_column(Integer, default=0)
    di: Mapped[int] = mapped_column(Integer, default=0)

    # Baserunning
    sb2: Mapped[int] = mapped_column(Integer, default=0)
    sb3: Mapped[int] = mapped_column(Integer, default=0)
    sbh: Mapped[int] = mapped_column(Integer, default=0)
    cs2: Mapped[int] = mapped_column(Integer, default=0)
    cs3: Mapped[int] = mapped_column(Integer, default=0)
    csh: Mapped[int] = mapped_column(Integer, default=0)
    pko1: Mapped[int] = mapped_column(Integer, default=0)
    pko2: Mapped[int] = mapped_column(Integer, default=0)
    pko3: Mapped[int] = mapped_column(Integer, default=0)
    k_safe: Mapped[int] = mapped_column(Integer, default=0)

    # Errors
    e1: Mapped[int] = mapped_column(Integer, default=0)
    e2: Mapped[int] = mapped_column(Integer, default=0)
    e3: Mapped[int] = mapped_column(Integer, default=0)
    e4: Mapped[int] = mapped_column(Integer, default=0)
    e5: Mapped[int] = mapped_column(Integer, default=0)
    e6: Mapped[int] = mapped_column(Integer, default=0)
    e7: Mapped[int] = mapped_column(Integer, default=0)
    e8: Mapped[int] = mapped_column(Integer, default=0)
    e9: Mapped[int] = mapped_column(Integer, default=0)

    # State Pre/Post
    outs_pre: Mapped[int] = mapped_column(Integer, default=0)
    outs_post: Mapped[int] = mapped_column(Integer, default=0)
    br1_pre: Mapped[Optional[str]] = mapped_column(String(20))
    br2_pre: Mapped[Optional[str]] = mapped_column(String(20))
    br3_pre: Mapped[Optional[str]] = mapped_column(String(20))
    br1_post: Mapped[Optional[str]] = mapped_column(String(20))
    br2_post: Mapped[Optional[str]] = mapped_column(String(20))
    br3_post: Mapped[Optional[str]] = mapped_column(String(20))

    # LOB & Responsible Pitchers
    lob_id1: Mapped[Optional[str]] = mapped_column(String(20))
    lob_id2: Mapped[Optional[str]] = mapped_column(String(20))
    lob_id3: Mapped[Optional[str]] = mapped_column(String(20))
    pr1_pre: Mapped[Optional[str]] = mapped_column(String(20))
    pr2_pre: Mapped[Optional[str]] = mapped_column(String(20))
    pr3_pre: Mapped[Optional[str]] = mapped_column(String(20))
    pr1_post: Mapped[Optional[str]] = mapped_column(String(20))
    pr2_post: Mapped[Optional[str]] = mapped_column(String(20))
    pr3_post: Mapped[Optional[str]] = mapped_column(String(20))

    # Run / RBI Scoring
    run_b: Mapped[Optional[str]] = mapped_column(String(20))
    run1: Mapped[Optional[str]] = mapped_column(String(20))
    run2: Mapped[Optional[str]] = mapped_column(String(20))
    run3: Mapped[Optional[str]] = mapped_column(String(20))
    prun_b: Mapped[Optional[str]] = mapped_column(String(20))
    prun1: Mapped[Optional[str]] = mapped_column(String(20))
    prun2: Mapped[Optional[str]] = mapped_column(String(20))
    prun3: Mapped[Optional[str]] = mapped_column(String(20))
    ur_b: Mapped[int] = mapped_column(Integer, default=0)
    ur1: Mapped[int] = mapped_column(Integer, default=0)
    ur2: Mapped[int] = mapped_column(Integer, default=0)
    ur3: Mapped[int] = mapped_column(Integer, default=0)
    rbi_b: Mapped[int] = mapped_column(Integer, default=0)
    rbi1: Mapped[int] = mapped_column(Integer, default=0)
    rbi2: Mapped[int] = mapped_column(Integer, default=0)
    rbi3: Mapped[int] = mapped_column(Integer, default=0)
    runs: Mapped[int] = mapped_column(Integer, default=0)
    rbi: Mapped[int] = mapped_column(Integer, default=0)
    er: Mapped[int] = mapped_column(Integer, default=0)
    tur: Mapped[int] = mapped_column(Integer, default=0)

    # Defense (Fielders & PO/A)
    f2: Mapped[Optional[str]] = mapped_column(String(20))
    f3: Mapped[Optional[str]] = mapped_column(String(20))
    f4: Mapped[Optional[str]] = mapped_column(String(20))
    f5: Mapped[Optional[str]] = mapped_column(String(20))
    f6: Mapped[Optional[str]] = mapped_column(String(20))
    f7: Mapped[Optional[str]] = mapped_column(String(20))
    f8: Mapped[Optional[str]] = mapped_column(String(20))
    f9: Mapped[Optional[str]] = mapped_column(String(20))
    po1: Mapped[int] = mapped_column(Integer, default=0)
    po2: Mapped[int] = mapped_column(Integer, default=0)
    po3: Mapped[int] = mapped_column(Integer, default=0)
    po4: Mapped[int] = mapped_column(Integer, default=0)
    po5: Mapped[int] = mapped_column(Integer, default=0)
    po6: Mapped[int] = mapped_column(Integer, default=0)
    po7: Mapped[int] = mapped_column(Integer, default=0)
    po8: Mapped[int] = mapped_column(Integer, default=0)
    po9: Mapped[int] = mapped_column(Integer, default=0)
    a1: Mapped[int] = mapped_column(Integer, default=0)
    a2: Mapped[int] = mapped_column(Integer, default=0)
    a3: Mapped[int] = mapped_column(Integer, default=0)
    a4: Mapped[int] = mapped_column(Integer, default=0)
    a5: Mapped[int] = mapped_column(Integer, default=0)
    a6: Mapped[int] = mapped_column(Integer, default=0)
    a7: Mapped[int] = mapped_column(Integer, default=0)
    a8: Mapped[int] = mapped_column(Integer, default=0)
    a9: Mapped[int] = mapped_column(Integer, default=0)
    fseq: Mapped[Optional[str]] = mapped_column(String(20))
    firstf: Mapped[int] = mapped_column(Integer, default=0)
    loc: Mapped[Optional[str]] = mapped_column(String(10))
    hittype: Mapped[Optional[str]] = mapped_column(String(5))
    dpopp: Mapped[int] = mapped_column(Integer, default=0)
    pivot: Mapped[int] = mapped_column(Integer, default=0)

    # Umpires
    umphome: Mapped[Optional[str]] = mapped_column(String(50))
    ump1b: Mapped[Optional[str]] = mapped_column(String(50))
    ump2b: Mapped[Optional[str]] = mapped_column(String(50))
    ump3b: Mapped[Optional[str]] = mapped_column(String(50))
    umplf: Mapped[Optional[str]] = mapped_column(String(50))
    umprf: Mapped[Optional[str]] = mapped_column(String(50))

    # File Info
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    gametype: Mapped[Optional[GameType]] = mapped_column(Enum(GameType), nullable=True)
    pbp: Mapped[Optional[str]] = mapped_column(String(10))


class HistoricalDataORM(Base):
    """Aggregated season/daily player statistics from pybaseball/FanGraphs."""

    __tablename__ = "historical_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)


class PlayerRollingFeaturesORM(Base):
    """Rolling window features for ML models (e.g. L7 ER, L14 wOBA)"""

    __tablename__ = "player_rolling_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    feature_name: Mapped[str] = mapped_column(String(50), nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)


class OpenMeteoWeatherProgressionORM(Base):
    """
    Environmental arc from 1st pitch (T0) through the 9th inning (T4)
    extracted from Open-Meteo Archive (ERA5) and Historical Forecast APIs.
    """

    __tablename__ = "openmeteo_weather_progression"

    game_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("game_results.game_id"), primary_key=True
    )

    # T0-T4 Raw Hourly Slice: Temperature (F), Wind Speed (MPH), Wind Dir (DEG)
    temp_t0_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t0: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t0: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))

    temp_t1_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t1: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t1: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))

    temp_t2_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t2: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t2: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))

    temp_t3_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t3: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t3: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))

    temp_t4_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t4: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t4: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))

    # Supplemental T0 (First Pitch) Actuals
    humidity_t0: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    precip_t0_mm: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    pressure_t0_hpa: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    cloud_cover_t0_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))

    # Derived Progression Aggregates
    temp_delta_game: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))  # T3 - T0
    temp_min_game: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_max_game: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_variance_deg: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    headwind_t0_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    headwind_t3_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    headwind_delta_game: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    crosswind_t0_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_shift_gt_45deg: Mapped[Optional[bool]] = mapped_column(Boolean)
    temp_drop_gt_10f: Mapped[Optional[bool]] = mapped_column(Boolean)
    precip_any_game: Mapped[Optional[bool]] = mapped_column(Boolean)

    # T-24h Opening Forecast
    forecast_temp_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_wind_speed_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_wind_dir_deg: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_headwind_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_crosswind_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_precip_prob: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_cloud_cover_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_source: Mapped[Optional[str]] = mapped_column(String(50))

    # Market Surprise Deltas (Actual T0 - Forecast)
    delta_temp_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    delta_wind_speed_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    delta_headwind_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    delta_precip_mm: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))

    era5_model_used: Mapped[Optional[str]] = mapped_column(String(50))
    fetched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class OpenMeteoDailyForecastORM(Base):
    """
    Season-long daily forecast snapshots (2021+) captured at T-24h
    from Open-Meteo Historical Forecast API.
    """

    __tablename__ = "openmeteo_daily_forecasts"

    game_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("game_results.game_id"), primary_key=True
    )
    temp_max_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    temp_min_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    precip_sum_mm: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    wind_speed_max_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    weather_code: Mapped[Optional[int]] = mapped_column(Integer)
    precip_prob_max_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    uv_index_max: Mapped[Optional[float]] = mapped_column(Numeric(4, 1))
    sunshine_duration_sec: Mapped[Optional[float]] = mapped_column(Float)
    fetched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PlayerTransactionORM(Base):
    """Historical player transactions and IL stints."""

    __tablename__ = "player_transactions"

    transaction_id: Mapped[str] = mapped_column(String, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    player_name: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )
    team_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    transaction_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    effective_date: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    resolution_date: Mapped[Optional[datetime.date]] = mapped_column(
        Date, nullable=True
    )
    type_desc: Mapped[str] = mapped_column(String, nullable=False)
    il_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    injury_body_part: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    injury_descriptor: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    raw_description: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    from sqlalchemy import Computed

    days_on_il: Mapped[Optional[int]] = mapped_column(
        Integer,
        Computed("resolution_date - effective_date", persisted=True),
        nullable=True,
    )
