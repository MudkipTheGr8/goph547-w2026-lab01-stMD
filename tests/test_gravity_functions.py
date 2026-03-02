# GOPH 547 Lab 01: forward modelling of corrected gravity data and the concepts of gravity potential and gravity effect.
# Matthew Davidson UCID: 30182729
# February 04, 2025
# Description:
# This file tests both functions from gravity.py (gravity_potential_point() and gravity_effect_point()) to make sure they are working.

import numpy as np
from goph547lab01.gravity import gravity_potential_point, gravity_effect_point

def test_gravity_potential_point():
    """Tests the gravity_potential_point function by comparing its output to expected values for known inputs.
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Asserts that the computed gravitational potential matches the expected values for the test cases.
    """
    # Test case 1: Simple case
    x = [0, 0, 0]
    xm = [0, 0, 1]
    m = 1.0
    U = gravity_potential_point(x, xm, m)
    expected_U = 6.67e-11 * m / 1.0
    assert np.isclose(U, expected_U), f"Expected {expected_U}, got {U}"

    # Test case 2: Different position
    x = [1, 1, 1]
    xm = [0, 0, 0]
    m = 2.0
    r = np.sqrt(3)
    U = gravity_potential_point(x, xm, m)
    expected_U = 6.67e-11 * m / r
    assert np.isclose(U, expected_U), f"Expected {expected_U}, got {U}"