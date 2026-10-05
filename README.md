# 🎥 Camera AI - YOLO11n Real-time Object Detection

> **3行サマリー**
> C++とOpenCVのDNNモジュールで、YOLO11n(ONNX形式)によるリアルタイム物体検出を行うデスクトップアプリです。
> Webカメラの映像から人・車などのオブジェクトを検出し、バウンディングボックスとFPSを画面表示しつつ、検出結果をTCP経由でPython側へ送信します。
> AIとのペアプログラミングで開発し、コードレビューを通じて改善点を洗い出しています(詳細は[Known Limitations](#-known-limitations)参照)。

---

## 📌 Overview

Webカメラの映像に対して、YOLO11nモデル(ONNX形式)をOpenCVのDNNモジュールでリアルタイム推論し、検出結果(バウンディングボックス・クラス名・信頼度)を画面に描画します。同時に、検出件数・推論時間・検出クラスをJSON形式でTCP経由でPythonサーバーへ送信し、他のシステム(ダッシュボードなど)と連携できる構成になっています。

YOLOの生出力(`[1, 84, 8400]`形式)を、ライブラリの高レベルAPIに頼らず、座標変換・NMS(Non-Maximum Suppression)まで自分で実装している点が特徴です。

---

## ✨ Features

### 1. YOLO11n Real-time Inference

- OpenCVの`cv::dnn`モジュールで、ONNX形式のYOLO11nモデルを読み込み、推論を実行
- COCOデータセット(80クラス)に対応

### 2. 生出力の手動パース

- YOLOの出力(`[1, 84, 8400]`)を`transpose`し、1候補=1行の形に整形
- 各候補のスコア(`colRange(4, 84)`)から、`cv::minMaxLoc`で最も確率の高いクラスと信頼度を取得
- 信頼度が閾値(`confidenceThreshold = 0.25`)未満の候補は除外

### 3. 座標変換とバウンディングボックスの補正

- モデルの入力サイズ(640×640)と、実際のカメラ映像サイズの比率(`xFactor`, `yFactor`)を使って、検出座標を元映像のスケールに変換
- `box &= cv::Rect(0, 0, frame.cols, frame.rows)`で、画面外にはみ出たボックスを切り詰め

### 4. NMS(Non-Maximum Suppression)

- `cv::dnn::NMSBoxes`で、同一オブジェクトに対する重複したバウンディングボックスを間引き

### 5. TCP通信によるPython連携

- 検出結果(検出数・推論時間・検出クラスのリスト)をJSON形式で、TCP経由でPythonサーバー(`127.0.0.1:5001`)へ送信
- 送信間隔は0.5秒に制限し、通信負荷を抑制
- 接続が切れた場合、3秒おきに自動で再接続を試行
- 送信失敗(`SOCKET_ERROR`)時は、ソケットを閉じて次回接続時にやり直す設計

### 6. リアルタイム表示

- 検出したオブジェクトにバウンディングボックスとラベル(クラス名+信頼度)を描画
- FPS・推論時間(ms)・検出オブジェクト数・TCP接続状態を画面左上にオーバーレイ表示

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| C++ | メイン実装言語 |
| OpenCV(`cv::dnn`) | YOLOモデルの読み込み・推論、画像処理、描画 |
| ONNX | YOLO11nモデルのフォーマット |
| Winsock2(`ws2_32`) | Python側とのTCP通信 |
| g++ / MinGW | コンパイル |

---

## 🚀 Setup

### Requirements

- Windows
- OpenCV(DNNモジュールを含むビルド)
- g++(MinGW等)
- `models/yolo11n.onnx`(リポジトリには含まれません。下の「Model」を参照)

### Model

モデルファイルはリポジトリに含めていません。Ultralytics で、ONNX形式に書き出して用意します。

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=onnx imgsz=640
```

書き出された `yolo11n.onnx` を、プロジェクトの `models/` フォルダ(なければ作成)に置きます。OpenCVのバージョンによっては読み込めないことがあります。その場合は、`opset=12` などを指定して書き出し直してください。

### Build

```bash
g++ src/main.cpp -o CameraAI.exe $(pkg-config --cflags --libs opencv5) -lws2_32
```

> `$(...)` を使うため、MSYS2 などの bash 系シェルで実行します。`opencv5` は pkg-config のパッケージ名です。環境によって `opencv4` などの場合があるので、`pkg-config --list-all` で opencv で始まる名前を確認して、合わせてください。

### Run

```bash
./CameraAI.exe
```

起動すると、Webカメラ(`cv::VideoCapture(0)`)を開き、Pythonサーバー(`127.0.0.1:5001`)への接続を試みます。`Q`または`ESC`キーで終了します。

> Pythonサーバー側が起動していなくても、C++側は単体で動作(検出・表示)します。TCP送信は、接続できている場合のみ行われます。

---

## 🔒 Privacy & Security Notes

- カメラ映像は、画面表示と推論に使うだけで、ファイルへの保存はしません。
- Pythonへ送るのは、検出数・推論時間・検出クラス名のみです。映像そのものは送りません。
- 送信先は `127.0.0.1`(同じPC内)です。TCP通信には認証・暗号化がないため、ローカルでの利用を前提としています。

---

## 🧠 Design Considerations

### なぜ生出力を自分でパースしているか

YOLOのラッパーライブラリに頼らず、OpenCVの`cv::dnn`で生のONNX出力を扱うことで、モデルの出力形式・NMSの仕組みを実際に理解することを目的としています。座標変換、スコアの抽出、NMSまでを自分で実装することで、ブラックボックスにせずに扱えるようにしています。

### なぜTCPでPythonと連携しているか

C++側は推論と映像処理に専念し、データの蓄積・可視化はPython側に任せることで、役割を分離しています。これは、[C System Monitor Ver.4](https://github.com/M4meShibA/C-System-Monitor-Ver.4)と同じ設計思想です。

---

## ⚠️ Known Limitations

開発中のコードレビューを通じて、以下の点を認識しています。今後の改善候補です。

- **80クラス固定の前提**: `scores`の範囲(`colRange(4, 84)`)は、COCOデータセット(80クラス)を前提にしています。独自データセットでクラス数の異なるモデルに差し替える場合は、この範囲を動的に変更する必要があります
- **JSON文字列の手組み**: 検出したクラス名を`"`で囲んでJSONに組み込んでいますが、エスケープ処理は行っていません。現状はCOCOの固定クラス名のみを扱うため問題は起きませんが、外部からクラス名が渡される設計に拡張する場合は注意が必要です
- **推論がメインループをブロック**: `net.forward()`の実行中、映像取得・画面更新・キー入力の受付が止まります。現在のCPU推論では大きな問題にはなっていませんが、フレームレートを重視する場合は、推論を別スレッド化し、最新のフレームのみを処理する設計への変更が有効です
- **出力サイズの前提が固定**: `output.size[1]`, `output.size[2]`を直接使っており、モデルのバージョンが変わって出力形状が変化した場合に気づきにくい構造です。起動時に出力サイズをログ表示する等の対策を検討しています
- **`WSAStartup`が再接続のたびに呼ばれる**: `WSACleanup`と対になっているため致命的ではありませんが、プログラム起動時に1回だけ呼ぶ形の方が素直です

---

## 🎯 Purpose

このプロジェクトでは、以下を実践的に学習・実装することを目的としています。

- C++でのOpenCV(`cv::dnn`)を用いた推論パイプラインの構築
- YOLOモデルの生出力を、自分の手でパース・後処理する経験
- C++とPythonを、TCP通信で疎結合に連携させる設計

[C System Monitor Ver.4](https://github.com/M4meShibA/C-System-Monitor-Ver.4)(C言語・Windows API・TCP通信)からの発展として、より計算負荷の高い画像処理・AI推論の領域に取り組んだプロジェクトです。

---

## 📄 License

MIT License(`LICENSE` を参照)。

ただし、`yolo11n.onnx` などのモデルファイルは、このリポジトリに含まれていません。モデルを使うときは、Ultralytics が定めるライセンスに従ってください(最新の条件は公式サイトで確認してください)。

---

## 👨‍💻 Author

Portfolio Project — Camera AI (YOLO11n Real-time Object Detection)

`C++` / `OpenCV` / `YOLO` / `ONNX` / `TCP` / `Winsock2`
