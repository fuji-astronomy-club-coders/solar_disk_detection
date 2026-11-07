import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from min2.MIN2ver2 import MIN2Config, show_circle, show_circle_simple


class MIN2PlotPointStyleTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_defaults_preserve_display_specific_styles(self) -> None:
        config = MIN2Config()
        self.assertIsNone(config.plot_point_size)
        self.assertIsNone(config.plot_point_color)

    def test_custom_style_is_validated(self) -> None:
        config = MIN2Config(plot_point_size=72, plot_point_color="#00aaff")
        self.assertEqual(config.plot_point_size, 72)
        self.assertEqual(config.plot_point_color, "#00aaff")

    def test_rejects_invalid_point_sizes(self) -> None:
        for value in (0, -1, True, float("inf"), float("nan"), "72"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MIN2Config(plot_point_size=value)  # type: ignore[arg-type]

    def test_rejects_invalid_point_colors(self) -> None:
        for value in ("not-a-color", 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MIN2Config(plot_point_color=value)  # type: ignore[arg-type]

    def test_detailed_plot_uses_configured_size_and_color(self) -> None:
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        with patch("min2.MIN2ver2.plt.show"):
            show_circle(
                image,
                spots=[[2.0, 3.0, 1.0, 0.0]],
                n=2,
                point_size=72,
                point_color="#00aaff",
            )

        collection = plt.gcf().axes[0].collections[0]
        np.testing.assert_array_equal(collection.get_sizes(), [72])
        np.testing.assert_allclose(collection.get_facecolors()[0], to_rgba("#00aaff"))

    def test_simple_plot_uses_configured_size_and_color(self) -> None:
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        with patch("min2.MIN2ver2.plt.show"):
            show_circle_simple(
                image,
                spots=[[2.0, 3.0, 1.0, 0.0]],
                n=2,
                markersize=123,
                point_color="#00aaff",
            )

        collection = plt.gcf().axes[0].collections[0]
        np.testing.assert_array_equal(collection.get_sizes(), [123])
        np.testing.assert_allclose(collection.get_facecolors()[0], to_rgba("#00aaff"))


if __name__ == "__main__":
    unittest.main()
