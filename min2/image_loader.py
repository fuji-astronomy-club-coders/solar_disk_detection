"""画像ファイルを NumPy 配列として読み込む。"""

from pathlib import Path

import cv2
import numpy as np
import tifffile
from astropy.io import fits


def load_image(image_path: str | Path) -> np.ndarray:
    """TIFF/FITS と一般的なラスター画像を読み込む。

    NaN は 0 に置換し、画素値の dtype は可能な限り維持する。
    """
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"存在しない画像ファイルです: {path}")

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    elif suffix in {".fits", ".fit", ".fts"}:
        image = fits.getdata(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"画像の読み込みに失敗しました: {path}")

    return np.asarray(np.nan_to_num(image, nan=0.0))
