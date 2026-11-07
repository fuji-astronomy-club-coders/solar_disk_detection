"""Tests for optional per-image MIN2 result sidecar generation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from min2.MIN2ver2 import MIN2Config, MIN2_ignore_sunspots, load_config


class Min2ResultFileTests(unittest.TestCase):
    """Check serialization, opt-in behavior, and save-result configuration."""

    def setUp(self) -> None:
        self.image = np.zeros((24, 40), dtype=np.uint16)

    def test_result_file_is_written_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sun_001.tiff"
            image_path.write_bytes(b"image")
            expected = ((19.25, 11.75), 9.5)
            def fake_core(**kwargs: object) -> tuple[tuple[float, float], float]:
                details = kwargs["result_details"]
                assert isinstance(details, dict)
                details.update(
                    {
                        "effective_configuration": {
                            "n": 10,
                            "light_threshold": 50,
                            "effective_light_threshold": 12800,
                            "limb_wigth": 24,
                            "iter_cycles": 2,
                            "show": False,
                            "debug": False,
                            "show_simple": False,
                            "interactive_review": False,
                            "image_dtype": "uint16",
                        },
                        "sampled_edge_points": [[1.0, 2.0, 3.0, 0.0]],
                        "initial_retained_points": [[1.0, 2.0, 3.0, 0.0]],
                        "final_retained_points": [[1.0, 2.0, 3.0, 0.0]],
                        "excluded_points": [
                            {
                                "iteration": 1,
                                "stage": "automatic_sunspot_filter",
                                "reason": "sunspot_edge_point",
                                "point": [4.0, 5.0, 6.0, 0.0],
                                "retry_point": [7.0, 8.0, 9.0, 0.0],
                            }
                        ],
                        "iterations": [
                            {"iteration": 1, "termination_reason": "circle_refitted"}
                        ],
                    }
                )
                return expected

            with patch(
                "min2.MIN2ver2._MIN2_ignore_sunspots_core",
                side_effect=fake_core,
            ) as core:
                result = MIN2_ignore_sunspots(
                    img_inst=self.image,
                    img_path=image_path,
                    config={"save_result": True},
                )

            sidecar = image_path.with_name("sun_001_min2.json")
            self.assertTrue(sidecar.is_file())
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(result, expected)
            self.assertEqual(payload["format"], "solar_disk_detection.min2_result")
            self.assertEqual(payload["format_version"], 2)
            self.assertEqual(payload["source_image"], {
                "name": "sun_001.tiff",
                "shape": [24, 40],
                "dtype": "uint16",
                "size_bytes": 5,
                "sha256": hashlib.sha256(b"image").hexdigest(),
            })
            self.assertEqual(payload["solar_geometry"], {
                "center_x_px": 19.25,
                "center_y_px": 11.75,
                "radius_px": 9.5,
            })
            self.assertEqual(payload["min2_version"], "MIN2 v2.5.2")
            self.assertEqual(payload["configuration"]["requested"]["save_result"], True)
            self.assertIn("effective", payload["configuration"])
            self.assertEqual(
                payload["detection_trace"]["excluded_points"][0]["point"],
                [4.0, 5.0, 6.0, 0.0],
            )
            self.assertEqual(
                payload["detection_trace"]["excluded_points"][0]["retry_point"],
                [7.0, 8.0, 9.0, 0.0],
            )
            self.assertEqual(
                payload["detection_trace"]["sampled_edge_points"],
                [[1.0, 2.0, 3.0, 0.0]],
            )
            self.assertEqual(
                payload["detection_trace"]["iterations"][0]["iteration"],
                1,
            )
            core.assert_called_once()

    def test_saving_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sun.tiff"
            image_path.write_bytes(b"image")
            with patch(
                "min2.MIN2ver2._MIN2_ignore_sunspots_core",
                return_value=((20.0, 12.0), 9.0),
            ):
                MIN2_ignore_sunspots(img_inst=self.image, img_path=image_path)
            self.assertFalse(image_path.with_name("sun_min2.json").exists())

    def test_enabled_saving_requires_an_existing_image_path(self) -> None:
        with patch("min2.MIN2ver2._MIN2_ignore_sunspots_core") as core:
            with self.assertRaises(ValueError):
                MIN2_ignore_sunspots(
                    img_inst=self.image,
                    config={"save_result": True},
                )
            core.assert_not_called()

    def test_save_result_configuration_requires_boolean(self) -> None:
        with self.assertRaises(ValueError):
            MIN2Config(save_result="true")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            load_config({"save_result": 1})


if __name__ == "__main__":
    unittest.main()
