"""Unit tests for the nautical direction conventions in ``utils.directions``."""

from __future__ import annotations

import unittest

import numpy as np

from utils.directions import (
    in_nautical_sector,
    math_sector_to_nautical_clockwise,
    math_to_nautical,
    nautical_sector_to_wedge_angles,
    nautical_sector_wedge_coords,
    normalize_nautical,
)


class TestNormalizeNautical(unittest.TestCase):
    def test_wraps_negative_into_0_360(self) -> None:
        self.assertEqual(normalize_nautical(-10.0), 350.0)

    def test_wraps_over_360(self) -> None:
        self.assertEqual(normalize_nautical(370.0), 10.0)

    def test_identity_inside_range(self) -> None:
        self.assertEqual(normalize_nautical(180.0), 180.0)

    def test_vectorized(self) -> None:
        out = normalize_nautical(np.array([-10.0, 0.0, 360.0, 725.0]))
        np.testing.assert_allclose(out, [350.0, 0.0, 0.0, 5.0])


class TestMathToNautical(unittest.TestCase):
    def test_east_is_90(self) -> None:
        # Math 0deg (East) -> nautical 90deg.
        self.assertEqual(math_to_nautical(0.0), 90.0)

    def test_north_is_0(self) -> None:
        # Math 90deg (North) -> nautical 0deg.
        self.assertEqual(math_to_nautical(90.0), 0.0)

    def test_wraps(self) -> None:
        self.assertEqual(math_to_nautical(-90.0), 180.0)


class TestMathSectorToNauticalClockwise(unittest.TestCase):
    def test_left_less_than_right_no_wrap(self) -> None:
        # A math wedge from 0 to 90 (East to North) maps to nautical [0, 90].
        left, right = math_sector_to_nautical_clockwise(0.0, 90.0)
        self.assertLess(left, right)
        self.assertEqual([left, right], [0.0, 90.0])

    def test_sector_crossing_north_has_negative_left(self) -> None:
        # North is math 90deg (90 - 0 = 90 nautical... i.e. nautical 0 <-> math
        # 90). A math wedge straddling 90 (here [45, 135]) straddles North.
        left, right = math_sector_to_nautical_clockwise(45.0, 135.0)
        self.assertLess(left, right)
        self.assertLess(left, 0.0)
        self.assertEqual([left, right], [-45.0, 45.0])


class TestInNauticalSector(unittest.TestCase):
    def test_no_wrap_sector(self) -> None:
        self.assertTrue(in_nautical_sector(150.0, 100.0, 200.0))
        self.assertFalse(in_nautical_sector(50.0, 100.0, 200.0))

    def test_boundaries_are_inclusive(self) -> None:
        self.assertTrue(in_nautical_sector(100.0, 100.0, 200.0))
        self.assertTrue(in_nautical_sector(200.0, 100.0, 200.0))

    def test_sector_crossing_north(self) -> None:
        # [-10, 50] means 350deg through North to 50deg.
        self.assertTrue(in_nautical_sector(0.0, -10.0, 50.0))
        self.assertTrue(in_nautical_sector(355.0, -10.0, 50.0))
        self.assertTrue(in_nautical_sector(25.0, -10.0, 50.0))
        self.assertFalse(in_nautical_sector(180.0, -10.0, 50.0))

    def test_vectorized(self) -> None:
        dirs = np.array([90.0, 150.0, 250.0])
        out = in_nautical_sector(dirs, 100.0, 200.0)
        np.testing.assert_array_equal(out, [False, True, False])


class TestNauticalSectorToWedgeAngles(unittest.TestCase):
    def test_theta1_less_than_theta2(self) -> None:
        theta1, theta2 = nautical_sector_to_wedge_angles(100.0, 200.0)
        self.assertLess(theta1, theta2)

    def test_matches_known_conversion(self) -> None:
        # Nautical [0, 90] (North to East, clockwise) -> math wedge [0, 90]
        # (East to North, CCW) since North(nautical 0)=math 90, East(nautical 90)=math 0.
        theta1, theta2 = nautical_sector_to_wedge_angles(0.0, 90.0)
        self.assertEqual((theta1, theta2), (0.0, 90.0))

    def test_wrap_around_adds_360(self) -> None:
        theta1, theta2 = nautical_sector_to_wedge_angles(350.0, 10.0)
        self.assertLess(theta1, theta2)
        self.assertGreaterEqual(theta2 - theta1, 0.0)


class TestNauticalSectorWedgeCoords(unittest.TestCase):
    def test_closed_polygon_starts_and_ends_at_origin(self) -> None:
        coords = nautical_sector_wedge_coords(10.0, 20.0, 100.0, 200.0, 0.5)
        self.assertEqual(coords[0], (10.0, 20.0))
        self.assertEqual(coords[-1], (10.0, 20.0))
        self.assertGreater(len(coords), 3)

    def test_explicit_point_count(self) -> None:
        coords = nautical_sector_wedge_coords(0.0, 0.0, 0.0, 90.0, 1.0, n_pts=5)
        # origin + 5 arc points + closing origin
        self.assertEqual(len(coords), 7)


if __name__ == "__main__":
    unittest.main()
