# solar_disk_detection-MIN2

太陽画像から最小二乗法を用いて太陽の近似円（中心座標および半径）を検出するPythonライブラリです。黒点（サンスポット）などのノイズ領域を識別・除外しながら、高精度な円検出を行います。

## 特徴

- **最小二乗法による高精度な円検出**: 輝度変化（微分値）からサンプリングされた輪郭点（エッジ）群をもとに近似円を算出します。
- **黒点の影響排除**: 検出された輪郭点の中から黒点に起因する誤検知点を識別し、再探査および排除を行うイテレーション機能を備えています。
- **柔軟な画像入力**: 8bit/16bit画像、カラー（BGR/BGRA）、グレースケール画像の自動判定・変換に対応しています。
- **豊富な視覚化ツール**: 検出結果の円およびエッジ点のプロットに加え、各エッジ付近の輝度・微分プロファイルを表示可能です。

## プロジェクト構造

```text
solar_disk_detection-MIN2_v2.4.0/
├── min2/
│   ├── __init__.py        # バージョン情報の定義
│   └── MIN2ver2.py        # 円検出・黒点除外処理コアロジック
├── generate_dependencies.py
├── pyproject.toml         # プロジェクト設定および依存関係定義
├── LICENSE                # ライセンス情報
└── README.md              # 本ドキュメント

```

## インストール・依存関係

本プロジェクトは Python 3.10 以上に対応しています。

### 必要パッケージ

* OpenCV (`opencv-python`)
* NumPy
* Matplotlib

### 設手順

```bash
# リポジトリのクローンまたは展開後、ディレクトに入 we
cd solar_disk_detection-MIN2_v2.4.0

# 依存パッケージのインストール
pip install -e .

```

---

## 基本的な使用方法

### 1. Pythonコードからの実行

メイン関数 `MIN2_ignore_sunspots` を呼び出して太陽の中心座標と半径を取得します。

```python
import cv2
from min2.MIN2ver2 import MIN2_ignore_sunspots

# 画像パスを指定して実行する場合
img_path = "path/to/sun_image.png"

# 中心座標 (cx, cy) と半径 r を取得
(cx, cy), r = MIN2_ignore_sunspots(
    img_inst="PATH",
    img_path=img_path,
    n=10,
    light_threshold=50,
    show=True,        # 検出結果の画像を表示
    show_simple=False  # エッジ部の微分グラフを含む詳細表示
)

print(f"中心座標: ({cx:.2f}, {cy:.2f}), 半径: {r:.2f}")

```

NumPy配列として読み込み済みの画像を直接渡すことも可能です。

```python
image = cv2.imread("path/to/sun_image.png", cv2.IMREAD_UNCHANGED)

(cx, cy), r = MIN2_ignore_sunspots(
    img_inst=image,
    n=10,
    light_threshold=50,
    show=False
)

```

### 2. コマンドライン（CLI）からの実行

`MIN2ver2.py` を直接実行して処理を行えます。

* **ファイル選択ダイアログを表示して実行**:
```bash
python -m min2.MIN2ver2

```


* **画像ファイルを直接指定して実行**:
```bash
python -m min2.MIN2ver2 path/to/sun_image.png

```


* **フォルダ内の全画像を対象に一括処理する場合**:
```bash
python -m min2.MIN2ver2 path/to/folder --dir

```



---

## 主要関数と変数・パラメータの説明

### `MIN2_ignore_sunspots(...)`

黒点（サンスポット）の影響を除外しながら、太陽の最終的な近似円（中心と半径）を検出するメイン関数です。

#### 引数

| 引数名 | 型 | デフォルト値 | 説明 |
| --- | --- | --- | --- |
| `img_inst` | `np.ndarray | str` | `"PATH"` | 入力画像オブジェクト（NumPy配列）または読み込み指示（`"PATH"` の場合は `img_path` を読み込み）。 |
| `n` | `int` | `10` | 画像格子の分割数（2以上）。画像上に配置する走査線の本数を指定します。 |
| `light_threshold` | `int` | `50` | 太陽像とみなす輝度のしきい値。16bit画像の場合は自動的に256倍されます。 |
| `limb_wigth` | `int` | `24` | 太陽の縁の幅の基準値。黒点判定の閾値計算に使用します。 |
| `iter_cycles` | `int` | `2` | 黒点排除の最大反復回数。`0` の場合は排除を行いません。 |
| `show` | `bool` | `False` | 検出結果・処理結果の画像描画を行うかどうか。 |
| `debug` | `bool` | `False` | ログ出力や途中経過の描写（`show=True` 時）を行うデバッグモード。 |
| `img_path` | `Path | str | None` | `""` | 処理対象画像のファイルパス（`img_inst="PATH"` 時は必須）。 |
| `show_simple` | `bool` | `False` | `True` に設定すると、各エッジプロファイルを省いたシンプルな画像のみを表示します。 |

#### 戻り値

* **`tuple[tuple[float, float], float]`**: `((cx, cy), r)`
* `cx`: 近似円の中心 x 座標
* `cy`: 近似円の中心 y 座標
* `r`: 近似円の半径



---

### サンプリング・円フィッティング内部関数

* **`cut_and_sampling(img_inst, sun_threshold, n)`**
* 画像を縦横 `n` 等分する分割線上で走査し、微分値（輝度変化）が大きい箇所を太陽のエッジ点としてサンプリングします。
* **戻り値**: サンプリングされた点 `[x, y, grad_val, direc]` のリスト。
* `direc`: 走査方向（`DIR_X_LINE`: 横線走査 / `DIR_Y_LINE`: 縦線走査）




* **`fit_circle(spots, show, img_inst, n)`**
* エッジ点の座標群から最小二乗法（`np.linalg.lstsq`）を用いて近似円の中心と半径を計算します。
* **戻り値**: `[cx, cy, r]`


* **`retry_edge_on_spot(spot, cir_stat, img_inst, sun_threshold)`**
* 黒点等により本来のエッジより内側で誤検知された点について、近似円の外側領域のみを再走査して正しい縁の点を再探査します。



---

## モジュール定数・グローバル変数

* **`version`**: ライブラリのバージョン文字列（例: `"MIN2 v2.4.0"`）。
* **`DIR_X_LINE` (`0`)**: 横方向走査線で検出したエッジ点を示すフラグ。
* **`DIR_Y_LINE` (`1`)**: 縦方向走査線で検出したエッジ点を示すフラグ。

---

## 旧コード（バージョン 2.3.5 以前）との互換性について

バージョン 2.3.5 以前の旧コードでは、グローバル変数 (`img`, `divnum`, `height`, `width`) や `"GLOBAL"` パラメータに依存した実装が行われていました。

本バージョン（v2.4.0）では以下の改善および互換性処理が行われています。

1. **グローバル変数依存の脱却**:
* 内部処理はすべて関数引数経由での受け渡しに統一され、マルチスレッド環境や複数画像の連続処理時の安全性が向上しました。


2. **旧コード互換用の維持**:
* `img_inst="GLOBAL"` を使用する外部コードとの互換性を保つため、`img` や `divnum` などのグローバル変数は実行時に内部で自動更新される仕様を維持しています。


3. **黒点リトライ座標および描画処理の修正**:
* 黒点探査時の範囲計算や `iter_cycles=0` 指定時の挙動修正など、旧バージョンで発生していた境界値での不具合が修正されています。



---

## ライセンス

本プロジェクトは LICENSE ファイルに記載されたライセンス条項のもとで公開されています。
