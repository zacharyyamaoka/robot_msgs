"""The stock robot viewer's strips (viz.rerun robot=): each y range holds the motion and the
limits within reach, and little more.

Found on the UR5e tour workspace-dev-029 (2026-11-24): wrist_2's qd peaks at 179.7 °/s against
its 180 °/s limit, and its strip spanned -236 ... 436 °/s: the 8 % air went on past the limit the
motion had already reached, then the half-span snap rounded it out to 400. Rerun drew one tick
("0") in it, and the other strips' wider tick labels shifted that strip's playhead by ~10 px.
Written before the fix.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_viz_rerun_robot import ur5e

from robot_msgs.viz import rerun as viz_rerun

OVERSHOOT = 0.25  # past a limit the motion reaches: at most this share of the strip (snap + air)


def test_a_motion_that_reaches_its_limit_gets_round_numbers_around_the_limit_not_past_it():
    low, high = viz_rerun._strip_range(np.array([-100.0, 0.0, 179.7]), -180.0, 180.0)
    assert -230.0 < low <= -180.0 and 180.0 <= high < 230.0, (low, high)


def test_a_motion_far_from_its_limits_is_padded_and_snapped_as_before():
    low, high = viz_rerun._strip_range(np.array([200.0, 330.0]), -360.0, 360.0)
    assert low < 200.0 and 360.0 <= high, (
        low,
        high,
    )  # +360 is within one motion span: the strip shows it
    assert low > 50.0, (low, high)


@pytest.mark.parametrize("task", ["workspace-dev-029", "workspace-dev-003"])
def test_no_strip_runs_far_past_a_limit_its_motion_reaches(task):
    traj, tool, data = ur5e(task)
    plan = viz_rerun._robot_plan(traj, task, data["urdf"], tool)
    past = []
    for short, tab, _, _ in viz_rerun._FIELDS:
        if short not in plan.signals:
            continue
        for index, joint in enumerate(plan.joints):
            shown = plan.signals[short][:, index] * plan.scale[index]
            low, high = (bound[index] * plan.scale[index] for bound in plan.limits[short])
            vmin, vmax = float(np.nanmin(shown)), float(np.nanmax(shown))
            span = max(vmax - vmin, 1e-9)
            y_low, y_high = viz_rerun._strip_range(shown, low, high)
            assert y_low <= vmin and vmax <= y_high, (tab, joint)
            width = y_high - y_low
            if (
                np.isfinite(high)
                and high - vmax <= span
                and y_high - max(vmax, high) > OVERSHOOT * width
            ):
                past.append(f"{tab} {joint}: {y_high:.0f} for a {high:.0f} limit")
            if (
                np.isfinite(low)
                and vmin - low <= span
                and min(vmin, low) - y_low > OVERSHOOT * width
            ):
                past.append(f"{tab} {joint}: {y_low:.0f} for a {low:.0f} limit")
    assert not past, past
