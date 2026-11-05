"""Tests for the shared image loader."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from astropy.io import fits

from min2.image_loader import load_image


class ImageLoaderTests(unittest.TestCase):
    def test_load_fits_and_replace_nan(self) -> None:
        source = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solar.fits"
            fits.PrimaryHDU(data=source).writeto(path)

            actual = load_image(path)

        np.testing.assert_array_equal(
            actual,
            np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32),
        )

    def test_load_tiff(self) -> None:
        source = np.array([[0, 1024], [2048, 4095]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solar.tiff"
            tifffile.imwrite(path, source)

            actual = load_image(path)

        np.testing.assert_array_equal(actual, source)


if __name__ == "__main__":
    unittest.main()
