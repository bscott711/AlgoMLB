from typing import Final

# Compass bearing FROM home plate TOWARD center field (degrees)
# Positive headwind = wind blowing OUT (batter-friendly)
# Source: Andrew Clem Stadium Statistics + Reddit 2023 Google Earth survey
STADIUM_HP_BEARINGS: Final[dict[str, float]] = {
    "angel_stadium": 220.0,
    "at_t_park": 100.0,  # Oracle Park
    "busch_stadium": 95.0,
    "camden_yards": 110.0,
    "chase_field": 155.0,
    "citi_field": 130.0,
    "citizens_bank_park": 110.0,
    "comerica_park": 145.0,
    "coors_field": 165.0,
    "dodger_stadium": 180.0,
    "fenway_park": 68.0,
    "globe_life_field": 220.0,
    "great_american_ball_park": 95.0,
    "guaranteed_rate_field": 90.0,
    "kauffman_stadium": 155.0,
    "loanDepot_park": 200.0,
    "marlins_park": 200.0,
    "miller_park": 195.0,  # American Family Field
    "minute_maid_park": 230.0,
    "nationals_park": 100.0,
    "petco_park": 310.0,
    "pnc_park": 110.0,
    "progressive_field": 115.0,
    "rogers_centre": 110.0,
    "safeco_field": 195.0,  # T-Mobile Park
    "target_field": 125.0,
    "tropicana_field": 180.0,
    "truist_park": 290.0,
    "wrigley_field": 175.0,
    "yankee_stadium": 135.0,
}
