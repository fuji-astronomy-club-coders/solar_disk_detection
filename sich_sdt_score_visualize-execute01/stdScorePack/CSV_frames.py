"""csv形式の画像の保存を読み込み"""

from pathlib import Path

import numpy as np

__versio__ = 1.2  # saveにdelimiterとfmtを追加


def save_images_to_csv(
    file_path: str, images_3d: np.ndarray, delimiter: str = ",", fmt: str = "%d"
) -> None:
    """
    3次元の画像配列（枚数, 縦, 横）を、形状情報をヘッダーに含めてCSVに保存する。

    Parameters
    ----------
    file_path : str
        保存先のCSVファイルのパス
    images_3d : np.ndarray
        保存する3次元のNumPy配列 (モノクロ画像を想定)
    """
    if images_3d.ndim != 3:
        raise ValueError(f"エラー: 3次元配列を指定してください。現在の次元数: {images_3d.ndim}")

    # 1. 2次元に変形 (枚数, 縦 * 横)
    num_images = images_3d.shape[0]
    images_2d = images_3d.reshape(num_images, -1)

    # 2. 形状情報をヘッダー文字列にする (例: "5,28,28")
    shape_header = f"{images_3d.shape[0]},{images_3d.shape[1]},{images_3d.shape[2]}"

    # 3. CSVに保存（コメントとして # を先頭に付ける）
    np.savetxt(
        file_path,
        images_2d,
        delimiter=delimiter,
        fmt=fmt,
        header=shape_header,
        comments="# ",
    )


def load_images_from_csv(file_path: str | Path, dtype: np.dtype[np.uint] = np.uint8) -> np.ndarray:  # pyright: ignore[reportArgumentType]
    """
    形状情報がヘッダーに記述されたCSVから、3次元の画像配列を復元して読み込む。

    Parameters
    ----------
    file_path : str|Path
        読み込むCSVファイルのパス
    dtype : data-type, optional
        読み込む配列のデータ型（デフォルトは np.uint8）

    Returns
    -------
    np.ndarray
        復元された3次元のNumPy配列
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"エラー: ファイルが見つかりません ({file_path})")

    # 1. ヘッダー行だけを読み込み、元のshapeを抽出
    with file_path.open(mode="r", encoding="utf-8") as f:
        first_line = f.readline()

    if not first_line.startswith("#"):
        raise ValueError("エラー: CSVの1行目に形状情報(# コメント)が見つかりません。")

    try:
        # '# 5,28,28\n' -> (5, 28, 28) に変換
        shape_str = first_line.replace("#", "").strip()
        original_shape = tuple(map(int, shape_str.split(",")))
    except Exception as e:
        raise ValueError("エラー: 形状情報の読み取りに失敗しました。") from e

    # 2. データ本体を読み込む (先頭の # 行は自動でスキップされる)
    loaded_2d = np.loadtxt(file_path, delimiter=",", dtype=dtype)

    # 3. 3次元に復元して返す
    loaded_3d = loaded_2d.reshape(original_shape)

    return loaded_3d


if __name__ == "__main__":
    csv_file = "test_dataset.csv"

    # 1. テストデータの作成 (3枚の10x10画像)
    print("--- テストデータ作成 ---")
    original_data = np.random.randint(0, 256, (3, 10, 10), dtype=np.uint8)
    print(f"元の形状: {original_data.shape}")

    # 2. 保存処理
    save_images_to_csv(csv_file, original_data)
    print(f"'{csv_file}' に保存しました。")

    # 3. 読み込み処理
    restored_data = load_images_from_csv(csv_file)
    print(f"'{csv_file}' から復元しました。")
    print(f"復元後の形状: {restored_data.shape}")

    # 4. データが完全に一致するか確認
    if np.array_equal(original_data, restored_data):
        print("✅ データは完全に一致しました！")
    else:
        print("❌ データが一致しません。")
