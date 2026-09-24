import os
import cv2
import tqdm
import sys
import numpy as np
import glob
import json
from pathlib import Path

from CSV_frames import save_images_to_csv
from MIN2ver2 import MIN2_ignore_sunspots
from samples.zip_operator import get_image_names_from_dir, load_image_from_path_cv2

# 全体
OUTPUT_MODE = "VIDEO"  # CSV(0),VIDEO(2),WITH(1)
# パラメータ std score
INPUT_DIR = "./sun_images"  # 処理対象の画像フォルダ
CROP_H = 800  # 抽出する画像サイズ(縦幅)
CROP_W = 800  # 抽出する画像サイズ(横幅)
OUT_DIR_CSV = "./output_pixels"  # CSV保存先フォルダ

# パラメータ colormap
DEBUG = True  # True: デバッグ情報を表示
OUTPUT_DIR = r"C:\Users\2025005585\Desktop\python"  # 出力画像の保存先ルートフォルダ
OUTPUT_NAME = "output_test_sample"  # 保存用フォルダ名（旧：動画ファイル名）
MEAN_STD_OUTPUT_DIR = (
    r"C:\Users\2025005585\Desktop\python"  # 平均値と標準偏差の出力画像の保存先フォルダ
)
MEAN_IMAGE_NAME = "mean_image"  # 平均値の出力画像のファイル名
STD_IMAGE_NAME = "std_image"  # 標準偏差の出力画像のファイル名
IMAGE_EXT = ".png"  # 画像の拡張子


if os.environ.get("RUN_BY_SUBPROCESS") == "true":
    print("このスクリプトは subprocess から実行されています。")
    # 標準入力から流れてきた文字列を一括で読み込む
    input_data = sys.stdin.read()

    # JSON文字列をPythonの辞書オブジェクトに復元
    locals().update(json.loads(input_data))


# 関数
def create_colormap():
    # 256要素を持つLUTを作成(偏差値0~100に対応する色を設定)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    # 偏差値50未満は濃い青から白へ段階的に変化
    lut[0:5] = [100, 0, 0]
    lut[5:10] = [115, 25, 25]
    lut[10:15] = [130, 50, 50]
    lut[15:20] = [145, 75, 75]
    lut[20:25] = [160, 100, 100]
    lut[25:30] = [175, 125, 125]
    lut[30:35] = [190, 150, 150]
    lut[35:40] = [205, 175, 175]
    lut[40:45] = [220, 200, 200]
    lut[45:50] = [235, 225, 225]
    # 偏差値50以上は白から濃い赤へ段階的に変化
    lut[50:55] = [255, 255, 255]
    lut[55:60] = [225, 225, 240]
    lut[60:65] = [195, 195, 225]
    lut[65:70] = [165, 165, 210]
    lut[70:75] = [135, 135, 195]
    lut[75:80] = [105, 105, 180]
    lut[80:85] = [75, 75, 165]
    lut[85:90] = [45, 45, 150]
    lut[90:95] = [15, 15, 135]
    lut[95:256] = [0, 0, 120]
    return lut


def normalize_image(image):
    """
    画像を50〜100に正規化する
    image:
        meanやstdなどの2次元画像
    Returns:
        50〜100のuint8画像
    """

    img_min = image.min()
    img_max = image.max()

    # 全画素が同じ値の場合
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - img_min) / (img_max - img_min) * 50 + 50

    return normalized.astype(np.uint8)


def save_statistics_image(image, filename):
    """
    meanやstdを正規化してカラーマップ画像として保存
    """

    # 0〜100に正規化
    normalized_image = normalize_image(image)

    # LUT適用のため3チャンネル化
    three_channel_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

    # カラーマップ適用
    color_mapped_image = cv2.LUT(three_channel_image, create_colormap())

    # 保存
    cv2.imwrite(filename, color_mapped_image)


def crop_and_pad(
    img: np.ndarray, cx: int, cy: int, crop_h: int, crop_w: int
) -> np.ndarray:
    # 切り抜きたい理想の範囲（画面外にはみ出す可能性あり）
    h, w = img.shape
    crop_h = int(crop_h / 2)
    crop_w = int(crop_w / 2)
    y1, y2 = cy - crop_h, cy + crop_h
    x1, x2 = cx - crop_w, cx + crop_w

    # 画面外にはみ出している量（余白の計算）
    top = max(0, -y1)
    bottom = max(0, y2 - h)
    left = max(0, -x1)
    right = max(0, x2 - w)

    # 画面内に収まる安全な範囲だけでまずは切りぬく
    crop_y1, crop_y2 = max(0, y1), min(h, y2)
    crop_x1, crop_x2 = max(0, x1), min(w, x2)
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # はみ出していた部分を黒色（0）で埋めて、常にsize x size にする
    padded = cv2.copyMakeBorder(
        cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    return padded


def extract_sun_mini(
    dir_path: str, h_size: int, w_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """フォルダ内の太陽画像から太陽中心を算出し、指定サイズで切りぬいた画像配列を返します。
    画面端にかかる場合は、足りない部分を黒く塗りつぶします。

    Args:
        folder(str):対象の画像が保存されているフォルダのパス
        h_size(int):切りぬく長方形の縦幅
        w_size(int):切りぬく長方形の横幅

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - 切りぬかれた画像の3次元配列（N,h_size,w_size)
            - 各画像の中心座標配列（N,2）
    """

    print(f"---画像の読み込みと切り抜き処理を開始:{dir_path}---")
    # 画像ファイルのみ1000枚取得
    image_names = get_image_names_from_dir(dir_path)
    frames = []
    min2_centers = []
    # tqdmによる進捗表示
    for name in tqdm.tqdm(image_names, desc="Processing images"):
        # 16bit(下位12bit)画像を輝度値(1ch)のまま正しく読み込む
        img = load_image_from_path_cv2(dir_path, name)
        if img is None:
            continue
        try:
            cx, cy, r = MIN2_ignore_sunspots(img, show=False, debug=False)
        except Exception:
            continue

        cx = int(cx)
        cy = int(cy)

        padded = crop_and_pad(img, cx, cy, h_size, w_size)

        frames.append(padded)
        min2_centers.append([cx, cy])

    return np.array(frames), np.array(min2_centers)


def calculate_hensachi(frames: np.ndarray):
    """平均画像・標準偏差画像・偏差値画像を計算する。"""

    # 平均画像
    mean = np.mean(frames, axis=0)

    # 標準偏差画像
    std = np.std(frames, axis=0)

    # 偏差値画像
    hensachi = np.where(std == 0, 50, 50 + 10 * (frames - mean) / std)

    return mean, std, hensachi


# --- 実行とCSV保存（1フレームずつピクセル保存） ---
if __name__ == "__main__":
    # 保存先フォルダの作成
    os.makedirs(OUT_DIR_CSV, exist_ok=True)

    print(f"\n--- 画像ファイルの読み込み開始: {INPUT_DIR} ---")

    frames, centers = extract_sun_mini(INPUT_DIR, h_size=CROP_H, w_size=CROP_W)
    mean, std, hensachi = calculate_hensachi(frames)

    # 平均値・標準偏差の確認(デバッグ表示)
    if DEBUG:
        print("====平均値・標準偏差データ情報====")
        print("framesサイズ:", frames.shape)
        print("meanサイズ:", mean.shape)
        print("stdサイズ:", std.shape)
        print("================================")

    # 1フレームごと、全ピクセルをCSVに保存
    if OUTPUT_MODE in ("CSV", "WITH", 0, 1):
        if len(frames) > 0:
            # 偏差値画像をCSVとして保存
            print("\n--- 偏差値CSV保存処理を開始 ---")

            HENSACHI_DIR = "./output_hensachi"
            os.makedirs(HENSACHI_DIR, exist_ok=True)

            cropped_hensachi = []
            for i, frame in enumerate(tqdm.tqdm(hensachi, desc="Saving Hensachi CSVs")):
                cropped_hensachi.append(
                    crop_and_pad(frame, centers[i][0], centers[i][1], CROP_H, CROP_W)
                )
            cropped_hensachi = np.array(cropped_hensachi)

            save_images_to_csv(
                f"{HENSACHI_DIR}/cropped_hensachi_{Path(INPUT_DIR).name}_allframe.csv",
                cropped_hensachi,
                delimiter=",",
                fmt="%.2f",
            )

            # 読み込み方法
            # loaded_data = np.load('images_archive.npz')
            # 保存時につけた名前「my_images」で取り出す（3次元のまま復元されます）
            # retrieved_images = loaded_data['my_images']

            print("偏差値CSVの保存が完了しました。")

            # 1ピクセルずつの個別アクセス（例：1枚目の座標x=10, y=20の明るさ）
            print("\n--- サンプルピクセルの確認 ---")
            print(f"個別ピクセル明るさ: {frames[0, 20, 10]}")

        else:
            print("有効なフレームが抽出されなかったため、保存処理をスキップしました。")

    """
    以下colormap
    """
    if OUTPUT_MODE in ("WITH", "VIDEO", 1, 2):
        data = hensachi

        # データ形状からフレーム数・画像サイズの取得
        n_frames, height, width,_ = data.shape

        # 入力データの確認(デバッグ表示)
        if DEBUG:
            print("====偏差値入力データ情報====")
            print("データサイズ:", data.shape)
            print("最小値:", data.min())
            print("最大値:", data.max())

            print("==========================")

        # カラーマップ作成
        colormap_lut = create_colormap()

        # =================== フォルダ作成と画像保存 ===================
        # 動画名(OUTPUT_NAME)と同じ名前のフォルダを作成
        frame_output_dir = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        os.makedirs(frame_output_dir, exist_ok=True)
        Path(frame_output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- 画像フレームの保存処理を開始: {frame_output_dir} ---")

        # 1フレームずつ取り出し、LUTを適用して画像として保存
        for i in tqdm.tqdm(range(n_frames), desc="Saving image frames"):
            frame = data[i]
            # 偏差値を0〜100に収めてuint8 型に変換
            clipped_frame = np.clip(frame, 0, 100).astype(np.uint8)
            # LUTを適用するため、グレースケール画像を3チャンネル(BGR)画像へ変換
            three_channel_frame = cv2.cvtColor(
                clipped_frame, cv2.COLOR_GRAY2BGR
            )
            # LUTを適用し、偏差値を対応する色へ変換
            color_mapped_frame = cv2.LUT(three_channel_frame, colormap_lut)

            # 画像ファイル名を設定 (例: frame_0000.png, frame_0001.png ...)
            frame_filename = f"frame_{i:04d}{IMAGE_EXT}"
            frame_filepath = os.path.join(frame_output_dir, frame_filename)

            # 画像保存
            cv2.imwrite(frame_filepath, color_mapped_frame)

        print("画像フレームの保存が完了しました")

        # =============平均と標準偏差の画像作成用=============
        # 平均値画像を保存
        save_statistics_image(
            mean,
            os.path.join(MEAN_STD_OUTPUT_DIR, MEAN_IMAGE_NAME + IMAGE_EXT),
        )

        # 標準偏差画像を保存
        save_statistics_image(
            std,
            os.path.join(MEAN_STD_OUTPUT_DIR, STD_IMAGE_NAME + IMAGE_EXT),
        )

        print("平均値画像と標準偏差画像の出力が完了しました")