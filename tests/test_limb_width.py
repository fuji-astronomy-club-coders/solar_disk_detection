"""Regression tests for limb width configuration and calculation."""

import unittest
from unittest.mock import patch

import numpy as np

from min2.MIN2ver2 import MIN2Config, MIN2_ignore_sunspots, load_config


class TestLimbWidthConfiguration(unittest.TestCase):
    """Validate absolute and radius-relative limb width settings."""

    def test_integer_pixel_width_is_supported(self) -> None:
        config = MIN2Config(limb_wigth=24)
        self.assertEqual(config.limb_wigth, 24)

    def test_fraction_string_is_supported(self) -> None:
        config = MIN2Config(limb_wigth="1/20")
        self.assertEqual(config.limb_wigth, "1/20")
        self.assertEqual(load_config({"limb_wigth": "1/20"}).limb_wigth, "1/20")

    def test_invalid_fraction_strings_are_rejected(self) -> None:
        invalid_values = ("1", "1/0", "0/0", "-1/2", "1/-2", "abc/2", "1/2/3")
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                MIN2Config(limb_wigth=value)

    def test_integer_width_does_not_use_ratio_state(self) -> None:
        image = np.zeros((32, 32), dtype=np.uint8)
        spots = [
            [20.0, 0.0, 0, 0],
            [-20.0, 0.0, 0, 0],
            [0.0, 20.0, 0, 0],
            [0.0, 5.0, 0, 0],
        ]
        with (
            patch("min2.MIN2ver2.cut_and_sampling", return_value=spots),
            patch(
                "min2.MIN2ver2.fit_circle",
                side_effect=[(0.0, 0.0, 10.0), (0.0, 0.0, 8.0)],
            ),
            patch("min2.MIN2ver2.retry_edge_on_spot") as retry,
        ):
            MIN2_ignore_sunspots(
                img_inst=image,
                n=2,
                light_threshold=0,
                limb_wigth=24,
                iter_cycles=1,
            )
        retry.assert_not_called()

    def test_fraction_uses_current_solar_radius(self) -> None:
        image = np.zeros((32, 32), dtype=np.uint8)
        spots = [
            [20.0, 0.0, 0, 0],
            [-20.0, 0.0, 0, 0],
            [0.0, 20.0, 0, 0],
            [0.0, 5.0, 0, 0],
        ]
        with (
            patch("min2.MIN2ver2.cut_and_sampling", return_value=spots),
            patch(
                "min2.MIN2ver2.fit_circle",
                side_effect=[
                    (0.0, 0.0, 10.0),
                    (0.0, 0.0, 8.0),
                    (0.0, 0.0, 8.0),
                ],
            ),
            patch("min2.MIN2ver2.retry_edge_on_spot", return_value=None) as retry,
        ):
            MIN2_ignore_sunspots(
                img_inst=image,
                n=2,
                light_threshold=0,
                limb_wigth="1/4",
                iter_cycles=1,
            )
        retry.assert_called_once()


if __name__ == "__main__":
    unittest.main()
