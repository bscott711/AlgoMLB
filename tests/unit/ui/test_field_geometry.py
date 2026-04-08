import pytest
import numpy as np
from algomlb.ui.components import field_equations

STADIUM_FUNCTIONS = [
    field_equations.arizona_diamondbacks,
    field_equations.atlanta_braves,
    field_equations.baltimore_orioles,
    field_equations.boston_red_sox,
    field_equations.chicago_cubs,
    field_equations.chicago_white_sox,
    field_equations.cincinnati_reds,
    field_equations.cleveland_guardians,
    field_equations.colorado_rockies,
    field_equations.detroit_tigers,
    field_equations.houston_astros,
    field_equations.kansas_city_royals,
    field_equations.los_angeles_angels,
    field_equations.los_angeles_dodgers,
    field_equations.miami_marlins,
    field_equations.milwaukee_brewers,
    field_equations.minnesota_twins,
    field_equations.new_york_mets,
    field_equations.new_york_yankees,
    field_equations.philadelphia_phillies,
    field_equations.pittsburgh_pirates,
    field_equations.san_diego_padres,
    field_equations.san_francisco_giants,
    field_equations.seattle_mariners,
    field_equations.st_louis_cardinals,
    field_equations.tampa_bay_rays,
    field_equations.texas_rangers,
    field_equations.toronto_blue_jays,
    field_equations.washington_nationals,
    field_equations.athletics,
]


@pytest.mark.parametrize("func", STADIUM_FUNCTIONS)
def test_stadium_polar_sweep(func):
    """Exhaustive sweep from 0 to 89 degrees (stadium models often exclude exactly 90)."""
    for theta in range(0, 90):
        r = func(float(theta))
        assert isinstance(r, (float, int, np.floating))
        assert r > 200
        assert not np.isnan(r)


def test_get_stadium_points_dispatch():
    """Verify the entry point correctly dispatches to stadium functions."""
    # Known stadium
    points = field_equations.get_stadium_points("Dodger Stadium")
    assert len(points) > 0
    # Each point is (r, theta)
    assert len(points[0]) == 2

    # Fallback case
    fallback = [330, 375, 400, 375, 330]
    points_fb = field_equations.get_stadium_points(
        "Unknown Park", fallback_dims=fallback
    )
    assert len(points_fb) > 0


def test_get_fallback_points_logic():
    """Verify the segmented linear fallback for generic stadiums."""
    dims = [330, 375, 400, 375, 330]
    points = field_equations._get_fallback_points(dims)
    assert len(points) > 0
    # Center field point should be near index 45 or in the list
    assert any(p[0] == 400 for p in points)
