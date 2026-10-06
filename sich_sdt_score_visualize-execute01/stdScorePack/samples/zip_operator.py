"""Operations on sample data stored in a zip file, and a function that allows those operations to be adapted for a standard directory."""

import zipfile
from pathlib import Path

import cv2
import numpy as np

"""
ZIPファイル操作用のユーティリティ関数
"""
__version__ = 2.1  # 非zip用完全置換関数の作製


def load_image_from_path_cv2(dirpath: str, image_name: str) -> np.ndarray | None:
    """ファイルシステムから直接画像を読み込んでOpenCV（NumPy配列）形式で返す。

    Args:
        image_path (str): 対象とする画像ファイルのパス。

    Returns
    -------
        np.ndarray: デコードされた画像データ（BGR形式のNumPy配列）。
    """
    image_path = Path(dirpath) / image_name
    # 16bit画像を正しく読み込むためにIMREAD_UNCHANGEDを使用
    return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)


def get_image_names_from_dir(directory_path: str, extensions: list) -> list[str]:
    """指定したディレクトリ（およびサブディレクトリ）から指定された拡張子に一致する画像ファイル名の一覧を取得する。

    Args:
    -------
        directory_path (str): 検索対象とするディレクトリのパス。
        extensions (list, optional): 抽出対象とする画像拡張子のリスト。デフォルトは ['.tiff']。

    Returns
    -------
        list[str]: 条件に一致した画像ファイルのフルパスリスト。
    """
    image_extensions = tuple(ext.lower() for ext in extensions)

    image_paths = []

    # pathlibを使用してディレクトリ内を再帰的に検索[cite: 1]
    for path in Path(directory_path).rglob("*"):
        # ファイルかつ拡張子が一致するもののみを抽出
        if path.is_file() and path.suffix.lower() in image_extensions:
            image_paths.append(str(path))

    return image_paths


def load_image_from_zip_cv2(zip_path: str, image_name: str) -> np.ndarray:
    """ZIPファイルから直接画像を読み込んでOpenCV（NumPy配列）形式で返す。

    Args:
        zip_path (str): 対象とするZIPファイルのパス。
        image_name (str): ZIP内にある画像ファイルのパス（名前）。

    Returns
    -------
        np.ndarray: デコードされた画像データ（BGR形式のNumPy配列）。
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        # バイナリデータを取得
        img_bytes = zf.read(image_name)

        # バイナリデータを一時的に1次元のNumPy配列（バイトデータ配列）に変換
        nparr = np.frombuffer(img_bytes, np.uint8)

        # 画像をデコードする
        img = cv2.imdecode(
            nparr, cv2.IMREAD_UNCHANGED
        )  # 16bit画像を正しく読み込むためにIMREAD_UNCHANGEDを使用

    return np.array(img)


def get_image_names_from_zip(zip_path: str, extensions: list) -> list[str]:
    """ZIPファイル内から指定された拡張子に一致する画像ファイル名の一覧を取得する。

    Args:
        zip_path (str): 対象とするZIPファイルのパス。
        extensions (list, optional): 抽出対象とする画像拡張子のリスト。デフォルトは ['.tiff']。

    Returns
    -------
        list[str]: 条件に一致した画像ファイル名（パス）のリスト。
    """
    image_extensions = tuple(ext.lower() for ext in extensions)

    image_names = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # ZIP内のすべてのファイルパスを取得
        for file_name in zf.namelist():
            # フォルダのエントリ（末尾が / ）は除外
            if file_name.endswith("/"):
                continue

            # 拡張子が一致するものを抽出
            if file_name.lower().endswith(image_extensions):
                image_names.append(file_name)

    return image_names
