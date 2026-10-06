"""偏差値動画にて確認された対角線状の青と赤のピクセル群が検出円の誤差に起因することを検証するために誤差を再現した動画を生成する実行ファイル"""

from pathlib import Path

import cv2
import numpy as np
import yaml
from stdScorePack.MIN2ver2 import MIN2_ignore_sunspots as min2v221
from stdScorePack.std_score_visualize import crop_and_pad

if __name__ == "__main__":
    # ===設定の読み込み===
    config_name = "DummyDiskError-config.yaml"
    config_path = Path(__file__).parent / config_name

    with config_path.open(mode="r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        sampleConfig = config["sampleConfig"]
        min2Config = config["min2Config"]
        demoConfig = config["demoConfig"]

    # --- 設定値のチェック ---
    # - demoConfig
    modes=["percent","absolute"]
    if demoConfig["mode"] not in modes:
        raise ValueError("modes is unknown value")

    # ===画像の読み込み===
    imagePath = Path(sampleConfig["imagePath"])
    if not imagePath.exists():
        raise FileNotFoundError(f"存在しないファイルです:{imagePath}")
    image = (cv2.imread(imagePath), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"ファイルの読み込みに失敗しました:{imagePath}")
    else:
        image=np.array(image)
    if image.shape[2] != 1:
        raise ValueError(f"画像のチャネル数が1ではありません チャネル数:{image.shape[2]}")

    # 太陽円盤を検出
    cx,cy,r = min2v221(
        readed_img=image, img_name=imagePath.name, img_path=str(imagePath), **min2Config
    )
    intcx=np.round(cx)
    intcy=np.round(cy)

    #最大誤差(pix)を計算
    if demoConfig["mode"]=="percent":
        errorRange = [np.round(r)*v for v in demoConfig["errorSizes"]]
    else:
        errorRange=[np.round(v) for v in demoConfig["errorSizes"]]


    # ===フレームデータの作成===
    # ---中心座標のリストを作成---
    # 誤差にウェイトをつけるならここ
    # ---中心座標をもとに全フレームを生成---
    # ===動画出力===
    # ---統計量計算---
    # ---可視描画---
    # ---動画出力---

