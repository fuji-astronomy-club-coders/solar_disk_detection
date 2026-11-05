"""Tests for interactive edge-point review configuration and flow."""

import unittest
from unittest.mock import patch

import numpy as np

from min2.MIN2ver2 import MIN2_ignore_sunspots, load_config


class InteractiveReviewTests(unittest.TestCase):
    """Validate interactive review settings and integration points."""

    def test_interactive_review_is_disabled_by_default(self) -> None:
        self.assertFalse(load_config().interactive_review)
        self.assertTrue(load_config({"interactive_review": True}).interactive_review)

    def test_non_boolean_interactive_review_setting_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_config({"interactive_review": "true"})

    def test_trial_circle_review_removal_refits_the_circle(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        spots = [
            [52.0, 32.0, 100.0, 0.0],
            [32.0, 52.0, 100.0, 1.0],
            [12.0, 32.0, -100.0, 0.0],
            [32.0, 12.0, -100.0, 1.0],
        ]
        reviewed_spots = spots[:3]

        with (
            patch("min2.MIN2ver2.cut_and_sampling", return_value=spots),
            patch(
                "min2.MIN2ver2.fit_circle",
                side_effect=[[32.0, 32.0, 20.0], [31.0, 31.0, 19.0]],
            ) as fit_circle,
            patch(
                "min2.MIN2ver2._review_points_interactively",
                return_value=reviewed_spots,
            ) as review,
        ):
            result = MIN2_ignore_sunspots(
                img_inst=image,
                n=2,
                light_threshold=0,
                iter_cycles=0,
                interactive_review=True,
            )

        review.assert_called_once()
        self.assertEqual(fit_circle.call_count, 2)
        np.testing.assert_array_equal(fit_circle.call_args_list[1].args[0], reviewed_spots)
        self.assertEqual(result, ((31.0, 31.0), 19.0))

    def test_each_iteration_opens_a_review_before_automatic_filtering(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        spots = [
            [52.0, 32.0, 100.0, 0.0],
            [32.0, 52.0, 100.0, 1.0],
            [12.0, 32.0, -100.0, 0.0],
            [32.0, 12.0, -100.0, 1.0],
        ]

        with (
            patch("min2.MIN2ver2.cut_and_sampling", return_value=spots),
            patch("min2.MIN2ver2.fit_circle", return_value=[32.0, 32.0, 20.0]),
            patch(
                "min2.MIN2ver2._review_points_interactively",
                side_effect=lambda _image, points, _circle, _title: points,
            ) as review,
        ):
            MIN2_ignore_sunspots(
                img_inst=image,
                n=2,
                light_threshold=0,
                iter_cycles=1,
                interactive_review=True,
            )

        self.assertEqual(review.call_count, 2)
        self.assertEqual(review.call_args_list[0].args[3], "Trial circle: review sampled edge points")
        self.assertEqual(review.call_args_list[1].args[3], "Iteration 1: review retained edge points")


if __name__ == "__main__":
    unittest.main()
