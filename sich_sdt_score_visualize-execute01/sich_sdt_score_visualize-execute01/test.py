#TODO:実行時に処理がどこまで進んだかわかりずらいので、loggingかprintによる進捗表示を。特にfor文等、長くなりそうな処理はtqdm moduleによる進捗表示を。
#TODO:画像の標準出力はコンソールが汚れるので避けるべし。

import os
import cv2
import tqdm
import sys
import numpy as np

from samples.zip_operater import get_image_names_from_zip, load_image_from_zip_cv2
zip_path = "samples/2025-07-20-PL1.zip"
image_names = get_image_names_from_zip(zip_path)
frames = []
for name in image_names:
    img = load_image_from_zip_cv2(zip_path, name)
    frames.append(img)
frames = np.array(frames)

#TODO:可読性が下がるので、asを使うimportは改行してください 

#---パラメータ宣言（頻繁に変更する設定をここに集約）---
INPUT_DIR = "./sun_images"   #処理対象の画像フォルダ
CROP_H = 800           #　抽出する画像サイズ(縦幅)
CROP_W = 800           #　抽出する画像サイズ(横幅)
OUT_DIR = "./output_pixels"  #  CSV保存先フォルダ

def extract_sun_mini(zip_path:str, h_size:int,w_size:int) -> np.ndarray:
    #NOTE:docstringを追加
    """フォルダ内の太陽画像から太陽重心を算出し、指定サイズで切りぬいた画像配列を返します。
    画面端にかかる場合は、足りない部分を黒く塗りつぶします。

    Args:
        folder(str):対象の画像が保存されているフォルダのパス
        h_size(int):切りぬく長方形の縦幅
        w_size(int):切りぬく長方形の横幅

    Returns:
        np.ndarray:切りぬかれた画像の3次元配列（N,h_size,w_size)
    """
    print(f"---画像の読み込みと切り抜き処理を開始:{zip_path}---")
    # 画像ファイルのみ1000枚取得
    #BUG:厳密に1000枚とは限らないので、フォルダ内の画像すべてを読み込む
    image_names = get_image_names_from_zip(zip_path)
    frames = []
    half_h = h_size//2
    half_w = w_size//2
    #tqdmによる進捗表示
    for name in tqdm.tqdm(image_names, desc="Processing images"):
        #FIXME:撮影は基本16bit(下位12bit)で行うので、`cv2.IMRED_GRAYSCALE`はアカン
        #16bit(下位12bit)画像を輝度値(1ch)のまま正しく読み込む
        img = load_image_from_zip_cv2(zip_path, name)
        if img is None: 
            continue
        #二値化処理
        _, thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)

        # 太陽の重心(cx, cy)を計算
        M = cv2.moments (thresh)
        if M["m00"] == 0: 
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        
        #切り抜きたい理想の範囲（画面外にはみ出す可能性あり）
        h,w = img.shape
        y1,y2 =  cy - half_h, cy + half_h
        x1,x2 = cx - half_w,cx + half_w

        #画面外にはみ出している量（余白の計算）
        top =max(0,-y1)
        bottom =max(0,y2 - h)
        left = max(0,-x1)
        right =max(0,x2 - w)

        #画面内に収まる安全な範囲だけでまずは切りぬく
        crop_y1,crop_y2 = max(0,y1),min(h,y2)
        crop_x1,crop_x2 = max(0,x1),min(w,x2)
        cropped = img[crop_y1:crop_y2,crop_x1:crop_x2]

        #はみ出していた部分を黒色（0）で埋めて、常にsize x size にする 
        padded = cv2.copyMakeBorder(cropped,top,bottom,left,right,cv2.BORDER_CONSTANT,value = 0)

        frames.append(padded)
        
    return np.array(frames)

# --- 実行とCSV保存（1フレームずつピクセル保存） ---
#FIXME:`if __name__ == "__main__"` がうまく使えていない。二人で相談してif文内に入れるものを決めてください。＃
if __name__ == "__main__":

    # コマンドライン引数からZIPのパスを取得（未指定ならデフォルト）
    if len(sys.argv) > 1:
        target_zip = sys.argv[1]
    else:
        target_zip = "samples/2025-07-20-PL1.zip"

    # 保存先フォルダの作成（タイポ os.makediirs を修正）
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # ZIPを展開せずに一時フォルダを使って安全に読み込む
    print(f"\n--- ZIPファイルの読み込み開始: {target_zip} ---")

    frames = extract_sun_mini(
        target_zip,
        h_size=CROP_H,
        w_size=CROP_W
    )
    
    # 1フレームごと、全ピクセルをCSVに保存
    if len(frames) > 0:
        print(f"¥n--- CSV保存処理を開始:{OUT_DIR}---")
        for i, frame in enumerate(tqdm.tqdm(frames,desc="Saving CSVs")):
            np.savetxt(f"{OUT_DIR}/frame_{i+1:03d}.csv", frame, delimiter=",", fmt="%d")

            # 偏差値画像をCSVとして保存
    print("\n--- 偏差値CSV保存処理を開始 ---")

    HENSACHI_DIR = "./output_hensachi"
    os.makedirs(HENSACHI_DIR, exist_ok=True)

    for i, frame in enumerate(tqdm.tqdm(hensachi, desc="Saving Hensachi CSVs")):
        np.savetxt(
            f"{HENSACHI_DIR}/hensachi_{i+1:03d}.csv",
            frame,
            delimiter=",",
            fmt="%.2f"
        )

        print("偏差値CSVの保存が完了しました。")
        
        # 1ピクセルずつの個別アクセス（例：1枚目の座標x=10, y=20の明るさ）
        print("\n--- サンプルピクセルの確認 ---")
        print(f"個別ピクセル明るさ: {frames[0, 20, 10]}")
    else:
        print("有効なフレームが抽出されなかったため、保存処理をスキップしました。")
#平均を計算
mean=np.mean(frames,axis=0)
print("平均画像の画素値")
print(mean)
#標準偏差を計算
std=np.std(frames,axis=0)
print("標準偏差画像")
print(std)
#偏差値を計算
#0除算を防ぐため、偏差値を50としました。
hensachi = np.where(
    std == 0,
    50,
    50 + 10 * (frames - mean) / std
)

"""
#偏差値画像を1枚ずつ表示する。
for i in range(len(hensachi)):
    print(f"{i+1}枚目の偏差値画像")
    print(hensachi[i])
"""
