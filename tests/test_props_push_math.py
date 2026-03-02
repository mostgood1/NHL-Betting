import math

from nhl_betting.models.props import (
    ev_two_way_decimal,
    is_integer_line,
    poisson_over_under_push_probs,
)


def test_is_integer_line_basic():
    assert is_integer_line(2.0)
    assert is_integer_line(2)
    assert not is_integer_line(2.5)


def test_poisson_over_under_push_probs_integer_line_sums_to_one():
    lam = 2.0
    line = 2.0
    p_over, p_under, p_push = poisson_over_under_push_probs(lam, line)
    assert 0.0 <= p_over <= 1.0
    assert 0.0 <= p_under <= 1.0
    assert 0.0 <= p_push <= 1.0
    assert math.isclose(p_over + p_under + p_push, 1.0, abs_tol=1e-10)
    # Under-win must exclude pushes on integer lines.
    assert math.isclose(p_under, 1.0 - p_over - p_push, abs_tol=1e-10)


def test_poisson_over_under_push_probs_half_line_has_no_push():
    lam = 2.0
    line = 2.5
    p_over, p_under, p_push = poisson_over_under_push_probs(lam, line)
    assert math.isclose(p_push, 0.0, abs_tol=1e-12)
    assert math.isclose(p_over + p_under, 1.0, abs_tol=1e-10)


def test_ev_two_way_decimal_with_push():
    # Even odds: win profit = +1.0, loss profit = -1.0, push = 0.0
    ev = ev_two_way_decimal(prob_win=0.4, dec_odds=2.0, prob_push=0.1)
    assert math.isclose(ev, -0.1, abs_tol=1e-12)
