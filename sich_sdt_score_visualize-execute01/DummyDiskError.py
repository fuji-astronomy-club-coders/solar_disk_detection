"""偏差値動画にて確認された対角線状の青と赤のピクセル群が検出円の誤差に起因することを検証するために誤差を再現した動画を生成する実行ファイル"""

import yaml
from stdScorePack.std_score_visualize import crop_and_pad
from stdScorePack.MIN2ver2 import MIN2_ignore_sunspots as min2v221
from pathlib import Path


if  __name__ == __main__:
    #---設定の読み込み---
    config_name ="DummyDiskError-config.yaml"
    config_path = Path(__path__).parent() / config_name

    with open(file=config_path, mode='r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        sampleConfig=config["sampleConfig"]
        min2Config=config["min2Config"]
    #---画像の読み込み---
    imagePath = Path(sampleConfig["imagePath"])
    if not imagePath.exists():
        raise FileNotFoundError(f"存在しないファイルです:{sampleimagePath}")
    image = (cv2.imread(imagePath), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"ファイルの読み込みに失敗しました:{sampleimagePath}")
    elif image.shape[2] != 1:
        raise ValueError(f"画像のチャネル数が1ではありません チャネル数:{image.shape[2]}")
    #---画像処理・解析---
    (cx,cy,r) = min2v221(image,img_name=imagePath.name,img_path=imagePath,**min2Config)

    croph,cropw=sampleConfig["cropHW"]
    cropped=crop_and_pad(img=image,cx=cx,cy=cy,crop_h=croph,crop_w=cropw)
