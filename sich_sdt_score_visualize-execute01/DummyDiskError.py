"""偏差値動画にて確認された対角線状の青と赤のピクセル群が検出円の誤差に起因することを検証するために誤差を再現した動画を生成する実行ファイル"""

import yaml
from pathlib import Path


if  __name__ == __main__:
    #---設定の読み込み---
    config_name ="DummyDiskError-config.yaml"
    config_path = Path(__path__).parent() / config_name

    with open(file=config_path, mode='r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
