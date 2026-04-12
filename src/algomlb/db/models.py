import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from algomlb.db.session import Base
from algomlb.domain import (
    GameStatus,
    TransactionStatus,
    GameType,
    PlayerRole,
    BaselineQuality,
    SurfaceType,
    RoofType,
)


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
        String(20), nullable=True, default=GameType.REGULAR_SEASON
    )
    doubleheader_num: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )
    ballpark_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("ballparks.id"), nullable=True
    )

    # Environmental Context
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Travel / Rest Fatigue
    home_rest_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_rest_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_travel_distance_km: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    away_travel_distance_km: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )


class GameLineupORM(Base):
    """Starting lineup for each game. One row per starter (9 per team side)."""

    __tablename__ = "game_lineups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    team_side: Mapped[str] = mapped_column(
        String(4), nullable=False
    )  # 'home' or 'away'
    batting_order: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 1-9
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    player_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    position: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "game_pk", "team_side", "batting_order", name="uq_game_lineup_slot"
        ),
        Index("ix_lineup_game_date", "game_date"),
    )


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
    game_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    batter_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    at_bat_number: Mapped[int] = mapped_column(Integer, nullable=True)
    pitch_number: Mapped[int] = mapped_column(Integer, nullable=True)
    inning: Mapped[int] = mapped_column(Integer, nullable=True)
    zone: Mapped[int] = mapped_column(Integer, nullable=True)

    # Pitch Metrics
    pitch_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    release_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    release_spin_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    release_extension: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    effective_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pfx_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pfx_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plate_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plate_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Batted Ball Metrics
    launch_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    launch_angle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Event Data
    events: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)

    # Participant Metadata
    stand: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    p_throws: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)

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
    left_center: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    center_field: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    right_center: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    right_field: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Section Heights
    lf_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lc_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cf_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rc_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rf_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    min_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_wall_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hr_park_effects: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extra_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_temp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    surface_type: Mapped[Optional[SurfaceType]] = mapped_column(
        Enum(SurfaceType), nullable=True
    )
    roof_type: Mapped[Optional[RoofType]] = mapped_column(Enum(RoofType), nullable=True)
    daytime: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hp_bearing_deg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # GPS Coordinates for orientation & self-maintenance
    hp_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hp_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


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


class PlayerRollingFeaturesORM(Base):
    """
    Gold Layer: Pre-materialized rolling features for ML training/inference.
    Calculated from StatcastPlayerGameLog using role-specific windows.
    """

    __tablename__ = "player_rolling_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    season: Mapped[int] = mapped_column(SmallInteger, nullable=False, index=True)
    role: Mapped[PlayerRole] = mapped_column(Enum(PlayerRole), nullable=False)

    # Window Metadata
    window_games: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    n_games_used: Mapped[Optional[int]] = mapped_column(SmallInteger)
    days_since_last_game: Mapped[Optional[int]] = mapped_column(SmallInteger)
    baseline_quality: Mapped[BaselineQuality] = mapped_column(
        Enum(BaselineQuality), nullable=False, default=BaselineQuality.COLD_START
    )
    shrinkage_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    computed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # --- PITCHER Rolling Features ---
    roll_pitches: Mapped[Optional[float]] = mapped_column(Float)
    roll_strikes_pct: Mapped[Optional[float]] = mapped_column(Float)
    roll_whiff_pct: Mapped[Optional[float]] = mapped_column(Float)
    roll_k_pct: Mapped[Optional[float]] = mapped_column(Float)
    roll_bb_pct: Mapped[Optional[float]] = mapped_column(Float)
    roll_avg_release_speed: Mapped[Optional[float]] = mapped_column(Float)
    roll_avg_pfx_x: Mapped[Optional[float]] = mapped_column(Float)
    roll_avg_pfx_z: Mapped[Optional[float]] = mapped_column(Float)
    roll_avg_pitcher_xwoba: Mapped[Optional[float]] = mapped_column(Float)
    roll_pitcher_xwoba_shrunk: Mapped[Optional[float]] = mapped_column(Float)

    # Momentum & Trends (EMA)
    ema_pitcher_xwoba_3g: Mapped[Optional[float]] = mapped_column(Float)
    ema_pitcher_xwoba_7g: Mapped[Optional[float]] = mapped_column(Float)
    ema_edge_pct_3g: Mapped[Optional[float]] = mapped_column(Float)
    ema_velo_degradation_3g: Mapped[Optional[float]] = mapped_column(Float)

    # Volatility (Consistency)
    std_pitcher_xwoba_15g: Mapped[Optional[float]] = mapped_column(Float)
    std_edge_pct_15g: Mapped[Optional[float]] = mapped_column(Float)
    std_release_pos_z_15g: Mapped[Optional[float]] = mapped_column(Float)

    # Fatigue & Stuff Stability
    fatigue_index_7d: Mapped[Optional[float]] = mapped_column(Float)
    fatigue_index_14d: Mapped[Optional[float]] = mapped_column(Float)
    delta_spin_rate_3g: Mapped[Optional[float]] = mapped_column(Float)
    delta_extension_3g: Mapped[Optional[float]] = mapped_column(Float)
    delta_fb_velo_3g: Mapped[Optional[float]] = mapped_column(Float)

    # --- BATTER Rolling Features ---
    roll_pas: Mapped[Optional[float]] = mapped_column(Float)
    roll_hits_per_pa: Mapped[Optional[float]] = mapped_column(Float)
    roll_k_pct_batter: Mapped[Optional[float]] = mapped_column(Float)
    roll_bb_pct_batter: Mapped[Optional[float]] = mapped_column(Float)
    roll_barrel_pct: Mapped[Optional[float]] = mapped_column(Float)
    roll_avg_launch_speed: Mapped[Optional[float]] = mapped_column(Float)
    roll_avg_launch_angle: Mapped[Optional[float]] = mapped_column(Float)
    roll_avg_batter_xwoba: Mapped[Optional[float]] = mapped_column(Float)
    roll_batter_xwoba_shrunk: Mapped[Optional[float]] = mapped_column(Float)

    # Momentum & Trends (EMA)
    ema_batter_xwoba_3g: Mapped[Optional[float]] = mapped_column(Float)
    ema_batter_xwoba_7g: Mapped[Optional[float]] = mapped_column(Float)
    ema_bat_speed_3g: Mapped[Optional[float]] = mapped_column(Float)
    ema_attack_angle_3g: Mapped[Optional[float]] = mapped_column(Float)
    ema_chase_pct_3g: Mapped[Optional[float]] = mapped_column(Float)
    ema_iz_whiff_pct_3g: Mapped[Optional[float]] = mapped_column(Float)

    # Volatility (Consistency)
    std_batter_xwoba_15g: Mapped[Optional[float]] = mapped_column(Float)
    std_launch_angle_15g: Mapped[Optional[float]] = mapped_column(Float)

    # --- SHARED/MATCHUP Features ---
    seasonal_xwoba_vs_rh: Mapped[Optional[float]] = mapped_column(Float)
    seasonal_xwoba_vs_lh: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint(
            "player_id", "game_date", "role", name="uq_player_rolling_features"
        ),
        Index("ix_prf_date_role", "game_date", "role"),
    )


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


class GumboPitchORM(Base):
    """Canonical wall-clock timestamps for every pitch/event via MLB GUMBO feed."""

    __tablename__ = "gumbo_pitches"

    game_pk: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    at_bat_number: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitch_number: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    play_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )
    start_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    end_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class StatcastRawORM(Base):
    """Raw Statcast pitch-level ingestion buffer (Source of Truth)."""

    __tablename__ = "statcast_raw"

    # Identity
    game_pk: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    game_type: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    at_bat_number: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitch_number: Mapped[int] = mapped_column(SmallInteger, primary_key=True)

    # Game context
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(3), nullable=False)
    away_team: Mapped[str] = mapped_column(String(3), nullable=False)
    inning: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    inning_topbot: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)

    # Players
    batter: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pitcher: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    player_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Batted ball
    hc_x: Mapped[Optional[float]] = mapped_column(Numeric(7, 2), nullable=True)
    hc_y: Mapped[Optional[float]] = mapped_column(Numeric(7, 2), nullable=True)
    bb_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    events: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Launch metrics
    launch_speed: Mapped[Optional[float]] = mapped_column(Numeric(5, 1), nullable=True)
    launch_angle: Mapped[Optional[float]] = mapped_column(Numeric(5, 1), nullable=True)
    launch_speed_angle: Mapped[Optional[int]] = mapped_column(
        SmallInteger, nullable=True
    )
    estimated_ba_using_speedangle: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 3), nullable=True
    )
    estimated_woba_using_speedangle: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 3), nullable=True
    )

    # Pitch mechanics
    pitch_type: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    release_speed: Mapped[Optional[float]] = mapped_column(Numeric(5, 1), nullable=True)
    release_spin_rate: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 1), nullable=True
    )
    pfx_x: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)
    pfx_z: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)
    plate_x: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)
    plate_z: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)

    bat_speed: Mapped[Optional[float]] = mapped_column(Numeric(5, 2), nullable=True)
    delta_home_win_exp: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 3), nullable=True
    )
    delta_run_exp: Mapped[Optional[float]] = mapped_column(Numeric(5, 3), nullable=True)
    des: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pitch_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    post_away_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    post_bat_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    post_fld_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    post_home_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    swing_length: Mapped[Optional[float]] = mapped_column(Numeric(5, 2), nullable=True)
    hit_distance_sc: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 1), nullable=True
    )
    release_pos_x: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    release_pos_z: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    zone: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stand: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    p_throws: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    hit_location: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    balls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    strikes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    game_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    on_3b: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    on_2b: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    on_1b: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    outs_when_up: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vx0: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    vy0: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    vz0: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    ax: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    ay: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    az: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    sz_top: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    sz_bot: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    effective_speed: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    release_extension: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    fielder_2: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fielder_3: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fielder_4: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fielder_5: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fielder_6: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fielder_7: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fielder_8: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fielder_9: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    release_pos_y: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    woba_value: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    woba_denom: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    babip_value: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    iso_value: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bat_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fld_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    if_fielding_alignment: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    of_fielding_alignment: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    spin_axis: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_slg_using_speedangle: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    delta_pitcher_run_exp: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    hyper_speed: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    home_score_diff: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bat_score_diff: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_win_exp: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    bat_win_exp: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    age_pit_legacy: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    age_bat_legacy: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    age_pit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    age_bat: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_thruorder_pitcher: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    n_priorpa_thisgame_player_at_bat: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    pitcher_days_since_prev_game: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    batter_days_since_prev_game: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    pitcher_days_until_next_game: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    batter_days_until_next_game: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    api_break_z_with_gravity: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    api_break_x_arm: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    api_break_x_batter_in: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    arm_angle: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    attack_angle: Mapped[Optional[float]] = mapped_column(Numeric(7, 3), nullable=True)
    attack_direction: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    swing_path_tilt: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    intercept_ball_minus_batter_pos_x_inches: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )
    intercept_ball_minus_batter_pos_y_inches: Mapped[Optional[float]] = mapped_column(
        Numeric(7, 3), nullable=True
    )

    # Metadata
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class StatcastQuantFeatures(Base):
    """
    ML-ready quantitative features derived from StatcastRaw.
    """

    __tablename__ = "statcast_quant_features"

    game_pk: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    at_bat_number: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitch_number: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    batter: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pitcher: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Metrics
    xba_raw: Mapped[Optional[float]] = mapped_column(Float)
    xba_calibrated: Mapped[Optional[float]] = mapped_column(Float)
    xwoba_raw: Mapped[Optional[float]] = mapped_column(Float)
    xwoba_calibrated: Mapped[Optional[float]] = mapped_column(Float)
    launch_quality: Mapped[Optional[int]] = mapped_column(SmallInteger)
    pfx_x_std: Mapped[Optional[float]] = mapped_column(Float)
    pfx_z_std: Mapped[Optional[float]] = mapped_column(Float)
    release_speed_std: Mapped[Optional[float]] = mapped_column(Float)
    spray_angle_deg: Mapped[Optional[float]] = mapped_column(Float)
    hit_x_ft: Mapped[Optional[float]] = mapped_column(Float)
    hit_y_ft: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata
    calibrated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    baseline_window_days: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["game_pk", "at_bat_number", "pitch_number"],
            [
                "statcast_raw.game_pk",
                "statcast_raw.at_bat_number",
                "statcast_raw.pitch_number",
            ],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "game_pk",
            "at_bat_number",
            "pitch_number",
            name="uq_statcast_quant_features_pk",
        ),
    )


class StatcastPlayerGameLog(Base):
    """
    Silver Medallion Layer: Game-level summary of player performance metrics.
    Acts as the source for Bayesian shrinkage and rolling feature generation.
    """

    __tablename__ = "statcast_player_game_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    role: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # 'PITCHER' or 'BATTER'

    # --- Pitcher Metrics ---
    pitches: Mapped[Optional[int]] = mapped_column(SmallInteger)
    strikes: Mapped[Optional[int]] = mapped_column(SmallInteger)
    whiffs: Mapped[Optional[int]] = mapped_column(SmallInteger)
    k: Mapped[Optional[int]] = mapped_column(SmallInteger)
    bb: Mapped[Optional[int]] = mapped_column(SmallInteger)
    fb_speed: Mapped[Optional[float]] = mapped_column(Float)
    avg_release_speed: Mapped[Optional[float]] = mapped_column(Float)
    avg_pfx_x: Mapped[Optional[float]] = mapped_column(Float)
    avg_pfx_z: Mapped[Optional[float]] = mapped_column(Float)
    avg_pitcher_xwoba: Mapped[Optional[float]] = mapped_column(Float)
    avg_release_extension: Mapped[Optional[float]] = mapped_column(Float)
    avg_spin_rate: Mapped[Optional[float]] = mapped_column(Float)
    avg_spin_axis: Mapped[Optional[float]] = mapped_column(Float)
    std_arm_angle: Mapped[Optional[float]] = mapped_column(Float)
    std_release_pos_z: Mapped[Optional[float]] = mapped_column(Float)
    edge_pct: Mapped[Optional[float]] = mapped_column(Float)
    fastball_velo_degradation: Mapped[Optional[float]] = mapped_column(Float)
    hard_hits_allowed: Mapped[Optional[int]] = mapped_column(SmallInteger)

    # --- Batter Metrics ---
    pas: Mapped[Optional[int]] = mapped_column(SmallInteger)
    abs: Mapped[Optional[int]] = mapped_column(SmallInteger)
    hits: Mapped[Optional[int]] = mapped_column(SmallInteger)
    batter_k: Mapped[Optional[int]] = mapped_column(SmallInteger)
    batter_bb: Mapped[Optional[int]] = mapped_column(SmallInteger)
    barrels: Mapped[Optional[int]] = mapped_column(SmallInteger)
    avg_launch_speed: Mapped[Optional[float]] = mapped_column(Float)
    avg_launch_angle: Mapped[Optional[float]] = mapped_column(Float)
    avg_batter_xwoba: Mapped[Optional[float]] = mapped_column(Float)
    avg_bat_speed: Mapped[Optional[float]] = mapped_column(Float)
    avg_swing_length: Mapped[Optional[float]] = mapped_column(Float)
    avg_attack_angle: Mapped[Optional[float]] = mapped_column(Float)
    std_launch_angle: Mapped[Optional[float]] = mapped_column(Float)
    in_zone_whiff_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    chase_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    sweet_spots: Mapped[Optional[int]] = mapped_column(SmallInteger)
    hard_hits: Mapped[Optional[int]] = mapped_column(SmallInteger)
    pull_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    center_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    oppo_count: Mapped[Optional[int]] = mapped_column(SmallInteger)

    # --- Platoon Splits ---
    xwoba_vs_rh: Mapped[Optional[float]] = mapped_column(Float)
    pa_vs_rh: Mapped[Optional[int]] = mapped_column(SmallInteger)
    xwoba_vs_lh: Mapped[Optional[float]] = mapped_column(Float)
    pa_vs_lh: Mapped[Optional[int]] = mapped_column(SmallInteger)

    # --- Metadata ---
    summarized_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "game_pk", "player_id", "role", name="uq_statcast_player_game_log"
        ),
    )


class StatcastProcessRegistry(Base):
    """
    Tracks the high-water mark for incremental processing of Statcast data.
    """

    __tablename__ = "statcast_process_registry"

    target_table: Mapped[str] = mapped_column(String(50), primary_key=True)
    last_processed_ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class StatcastBattedBallORM(Base):
    """
    Analytics table for Batted Ball Flight Decoupling.
    Stores environmental residuals and raw contact quality.
    """

    __tablename__ = "statcast_batted_balls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    batter_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    venue_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    stand: Mapped[str] = mapped_column(String(1), nullable=False)

    # Raw Statcast Batted Ball Data
    launch_speed: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    launch_angle: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    hc_x: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    hc_y: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    spray_angle: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    hit_distance_sc: Mapped[Optional[float]] = mapped_column(Numeric(6, 1))
    bb_type: Mapped[Optional[str]] = mapped_column(String(20))
    events: Mapped[Optional[str]] = mapped_column(String(40))
    is_rhb: Mapped[int] = mapped_column(Integer, nullable=False)

    # Environmental Inputs (Game Time)
    temperature_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    pressure_hpa: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    relative_humidity: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_direction_deg: Mapped[Optional[float]] = mapped_column(Numeric(6, 1))
    precipitation_mm_hr: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))

    # Derived Physics Features
    cf_bearing_deg: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    air_density_ratio: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    tailwind_component: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))

    # Decoupler Analytics (Populated by ML pipeline)
    baseline_distance: Mapped[Optional[float]] = mapped_column(Numeric(6, 1))
    total_delta: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    delta_density: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    delta_wind: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    delta_precip: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    environmental_factor: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    spin_contact_factor: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )


class TeamManagerORM(Base):
    """
    Team manager lookup table — one row per (team_id, season).
    Sourced from MLB StatsAPI coaching roster (jobId=MNGR).
    Used by manager_hook_events for hook-behavior modeling.
    """

    __tablename__ = "team_managers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    team_abbr: Mapped[str] = mapped_column(String(5), nullable=False)
    team_name: Mapped[str] = mapped_column(String(100), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    manager_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    manager_name: Mapped[str] = mapped_column(String(100), nullable=False)
    jersey_number: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    effective_start_date: Mapped[Optional[datetime.date]] = mapped_column(
        Date, nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "team_id",
            "season",
            "manager_id",
            name="uq_team_managers_team_season_mgr",
        ),
    )


class GameManagerRegistryORM(Base):
    """
    Identity resolution and manager state for every team-game.
    Maps Retrosheet IDs to canonical MLB game_pk.
    One row per team per game (2 rows per game_pk).
    """

    __tablename__ = "game_manager_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    retrosheet_game_id: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )

    team_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    opponent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    manager_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    home_away: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # 'home' or 'away'
    doubleheader_num: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    game_type: Mapped[str] = mapped_column(String(10), nullable=False)  # 'R', 'P', etc.

    # Manager context at game time (derived)
    manager_stint_start: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    manager_tenure_day: Mapped[int] = mapped_column(Integer, nullable=False)
    days_since_manager_change: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("game_pk", "team_id", name="uq_registry_game_team"),
        UniqueConstraint(
            "retrosheet_game_id", "team_id", name="uq_registry_retro_team"
        ),
    )


class ManagerHookEventORM(Base):
    """
    Every pitcher removal observed in historical games.
    Refined schema captures deterministic state for simulation priors.
    Derived from retrosheet_events + GameManagerRegistry mapping.
    """

    __tablename__ = "manager_hook_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Game identity
    game_pk: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Team & manager
    team_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    opponent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    manager_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Pitcher being removed
    pitcher_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    is_starter: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Game state at moment of hook
    inning: Mapped[int] = mapped_column(Integer, nullable=False)
    outs_at_hook: Mapped[int] = mapped_column(Integer, nullable=False)
    pitches_thrown: Mapped[int] = mapped_column(Integer, nullable=False)
    tto_at_hook: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # times through order

    # Contextual features
    score_diff_at_hook: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # pitcher's team - opponent
    base_state_at_hook: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # 0-7 bitmask
    leverage_index_at_hook: Mapped[float] = mapped_column(Float, nullable=False)

    # Manager context
    manager_tenure_day: Mapped[int] = mapped_column(Integer, nullable=False)
    days_since_manager_change: Mapped[int] = mapped_column(Integer, nullable=False)

    # Simulation dependencies
    bullpen_availability_snapshot_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    hook_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    removed_before_next_batter: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "game_pk",
            "pitcher_id",
            name="uq_manager_hook_game_pitcher",
        ),
    )


class ManagerHookProfileORM(Base):
    """
    Rolling aggregated manager hook tendencies.
    One row per (manager_id, season) with cumulative rates.
    Provides a stable prior for the simulator before game-specific context.
    """

    __tablename__ = "manager_hook_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    manager_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    manager_name: Mapped[str] = mapped_column(String(100), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Sample size
    total_hooks: Mapped[int] = mapped_column(Integer, nullable=False)
    total_sp_starts: Mapped[int] = mapped_column(Integer, nullable=False)

    # Aggregate tendencies
    avg_sp_pitch_count: Mapped[float] = mapped_column(Float, nullable=False)
    avg_ip_per_start: Mapped[float] = mapped_column(Float, nullable=False)
    avg_hook_inning: Mapped[float] = mapped_column(Float, nullable=False)

    # Hook profiles (rates)
    pull_before_3rd_tto_pct: Mapped[float] = mapped_column(Float, nullable=False)
    pull_with_lead_pct: Mapped[float] = mapped_column(Float, nullable=False)
    pull_when_over_90_pitches_pct: Mapped[float] = mapped_column(Float, nullable=False)
    quick_hook_high_leverage_pct: Mapped[float] = mapped_column(Float, nullable=False)
    bullpen_protective_pct: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "manager_id",
            "season",
            name="uq_manager_hook_profile_mgr_season",
        ),
    )


class TeamEloHistoryORM(Base):
    """
    Team-level Elo ratings computed from game results only.
    No market odds are used — this is a pure baseball outcomes prior.
    Two rows per game (one home, one away).
    """

    __tablename__ = "team_elo_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    game_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    team_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    elo_pre: Mapped[float] = mapped_column(Float, nullable=False)
    elo_post: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "game_pk",
            "team_id",
            "is_home",
            name="uq_team_elo_history_row",
        ),
    )


class UraniumEvalHistoryORM(Base):
    """
    Walk-forward evaluation summary for Uranium models.
    One row per (model_target, model_version, fold_date).
    """

    __tablename__ = "uranium_eval_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_target: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    train_start_year: Mapped[int] = mapped_column(Integer, nullable=False)
    train_end_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fold_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)

    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    auc: Mapped[Optional[float]] = mapped_column(Float)
    log_loss_val: Mapped[Optional[float]] = mapped_column(Float)
    brier: Mapped[Optional[float]] = mapped_column(Float)
    ece: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "model_target",
            "model_version",
            "fold_date",
            name="uq_uranium_eval_target_version_fold",
        ),
    )


class UraniumCalibrationBinsORM(Base):
    """
    Binned calibration results for Uranium reliability curves.
    Typically 20 bins per (model_target, model_version, fold_date).
    """

    __tablename__ = "uranium_calibration_bins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_target: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    fold_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)

    bin_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bin_start: Mapped[float] = mapped_column(Float, nullable=False)
    bin_end: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_prob_mean: Mapped[float] = mapped_column(Float, nullable=False)
    actual_prob_mean: Mapped[float] = mapped_column(Float, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "model_target",
            "model_version",
            "fold_date",
            "bin_index",
            name="uq_uranium_calibration_bin_target_version_fold",
        ),
    )


class UraniumShapGlobalORM(Base):
    """
    Global SHAP feature importance for Uranium models.
    Tracks feature impact trends across temporal folds.
    """

    __tablename__ = "uranium_shap_global"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_target: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    fold_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)

    feature_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    mean_abs_shap: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "model_target",
            "model_version",
            "fold_date",
            "feature_name",
            name="uq_uranium_shap_global_target_version_fold",
        ),
    )


class HistoricalDataORM(Base):
    """
    Silver Layer: Legacy aggregate player/team statistics.
    Deprecated in favor of pitch-level Statcast data but retained for backward compatibility.
    """

    __tablename__ = "historical_player_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)


class UraniumSimulatedPlayerPropsORM(Base):
    """
    Monte Carlo simulation results for individual player props.
    Stores the refined probability distribution for each betting market.
    """

    __tablename__ = "uranium_simulated_player_props"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Prop Market (e.g., 'K', 'H', 'HR', 'RBI', 'R', 'TB', 'HRR')
    stat_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Distribution Metrics
    mean: Mapped[float] = mapped_column(Float, nullable=False)
    median: Mapped[float] = mapped_column(Float, nullable=False)

    # Cumulative Probabilities (P >= X)
    prob_over_0_5: Mapped[Optional[float]] = mapped_column(Float)
    prob_over_1_5: Mapped[Optional[float]] = mapped_column(Float)
    prob_over_2_5: Mapped[Optional[float]] = mapped_column(Float)
    prob_over_3_5: Mapped[Optional[float]] = mapped_column(Float)
    prob_over_4_5: Mapped[Optional[float]] = mapped_column(Float)

    # Percentiles for risk assessment
    p10: Mapped[Optional[float]] = mapped_column(Float)
    p90: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata
    trials: Mapped[int] = mapped_column(Integer, default=10000)
    simulated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "game_pk",
            "player_id",
            "stat_type",
            name="uq_simulated_player_prop_game_player_stat",
        ),
    )
