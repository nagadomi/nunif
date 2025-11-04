# 動画用インペイントモデルの学習

## 1. 学習用データセットの作成

ソースとなる動画ファイルから、学習データを生成します。

ソース動画には通常の2D動画を使用します。動画は1080pの解像度を推奨します。生成時の処理で1080pまたは720pにリサイズされるため、1080p未満の動画では拡大による画質劣化が発生します。

### 学習データ生成コマンド

動画1つごとに以下のコマンドを実行し、学習データを生成します。
まずは短い動画で動作確認することを推奨します。

```
python create_training_data.py video_inpaint --dataset-dir path/to/video.mp4 --prefix sintel -o ./data/video_inpaint_v1/train/ --model-type Any_B
```

コマンド実行後、`./data/video_inpaint_v1/train/`内に各データが格納されたディレクトリが生成されます。1ディレクトリ=1データです。ディレクトリ名は`{prefix}_{model_type}_{FILE_HASH}_{SEQUENCE_NO}`となります。動画ファイルと`--prefix`と`--model-type`の組み合わせが衝突しないように注意してください。

<img width="813" height="630" alt="data-ouput" src="https://github.com/user-attachments/assets/4cad2728-9611-479d-8937-a4b77ca0a813" />


各ディレクトリには画像シーケンスが保存されます。`_C.png`がRGBフレーム、`_M.png`がインペイントマスクです。

<img width="828" height="538" alt="video-sequence-data" src="https://github.com/user-attachments/assets/451ebf7a-73c6-4461-89ff-68e427f89bc6" />

学習用データディレクトリには`train/`と`eval/`が必要です。
`train/`は学習用、`eval/`は評価用です。この例では`./data/video_inpaint_v1/train`に出力します。

### コマンドオプション一覧

`python create_training_data.py video_inpaint -h`で全オプションを表示できます。一部主要オプションは以下の通りです。

| オプション            | 説明
|----------------------|--------------------------------------------------------------
|`video_inpaint`       | 動画用データ生成スクリプトの呼び出し（固定）
|`--dataset-dir`       | 入力動画ファイルのパス
|`-o`                  | 出力先ディレクトリ
|`--prefix`            | データのプレフィックス（動画ごとに設定推奨）
|`--model-type`        | 使用する深度推定モデル名（iw3準拠）
|`--resolution`        | 深度画像の解像度（iw3準拠）
|`--max-fps`           | サンプリング最大FPS（デフォルト: 30）
|`--seq`               | 1データあたりのフレーム数（デフォルト: 16、学習時に12フレームをランダムクロップ）
|`--skip-first`        | 冒頭でスキップするフレーム数（デフォルト: 0）
|`--skip-interval`     | データごとのスキップ間隔（デフォルト: 16）
|`--batch-size`        | 深度推定モデルのバッチサイズ（デフォルト: 2）
|`--size`              | 画像サイズ（デフォルト: 512、学習時にはここから256x256タイルにクロップ）
|`--num-samples`       | ランダムクロップの回数（デフォルト: 1）

#### 入出力関連
- 必須: `--dataset-dir`, `-o`, `--prefix`

#### 深度推定モデル関連
- `--model-type`, `--resolution`  
  同じ動画に対して異なる深度推定モデルで複数バリエーションを作成可能

#### データ拡張・調整
- `--skip-first`, `--skip-interval`, `--num-samples`  
  長い動画なら`--skip-interval`を大きく、短い動画やデータ拡張時は`--num-samples`を増やすなど、データ数を調整  
  `--skip-first`で開始位置を調節(別コマンドとサンプリング位置を変更、冒頭のスキップなど)

### ソース動画の選定

多様なジャンルの動画を推奨します。インペイント範囲は主に背景のため、被写体(前景)の影響は少ないと思います。

- 屋外（都市、森林、海、空 など）
- 屋内（室内、工場等）

動画例：
- 都市散策
- サバイバル・ハイキング
- 工場見学、オフィスツアー、ジム
- ビーチ

### バッチ処理

複数動画や異なる深度推定モデルの組み合わせで生成する場合は、シェルスクリプトやBATファイルの利用を推奨します。エラー時はプレフィクスを条件に該当データを削除し、再処理してください。

例
```bash
#!/bin/bash -e

OUTPUT_DIR=./data/video_inpaint_v1/train/

python create_training_data.py video_inpaint --dataset-dir /data/videos/shibuya.webm -o ${OUTPUT_DIR} --prefix shibuy1 --model-type Any_B
python create_training_data.py video_inpaint --dataset-dir /data/videos/shibuya.webm -o ${OUTPUT_DIR} --prefix shibuy2 --model-type Any_L --resolution 518 --skip-first 16
python create_training_data.py video_inpaint --dataset-dir /data/videos/shibuya.webm -o ${OUTPUT_DIR} --prefix shibuy3 --model-type ZoeD_Any_N

python create_training_data.py video_inpaint --dataset-dir /data/videos/kamakura.webm -o ${OUTPUT_DIR} --prefix kamakura1 --model-type Any_B
python create_training_data.py video_inpaint --dataset-dir /data/videos/kamakura.webm -o ${OUTPUT_DIR} --prefix kamakura2 --model-type Distill_Any_S --skip-first 16
python create_training_data.py video_inpaint --dataset-dir /data/videos/kamakura.webm -o ${OUTPUT_DIR} --prefix akamkura3 --model-type ZoeD_Any_N

# ...
```

### 評価用データ（`eval/`）の作成

評価用データは、学習時の進捗確認や性能評価に用います。

**理想**：`train/`で使用していない動画から`eval/`用データを作成する  
**簡易法**：`train/`に作成済みの一部データを`eval/`フォルダへ移動 (切り取り&貼り付け)

簡易方法でのデータ数は200程度が目安です。

## 2. モデル学習

作成したデータセットを用い、動画インペイントモデルの学習を開始します（画像インペイントと同じのコマンドですが、`--video`オプションを付与）。

```
python train.py inpaint -i ./data/video_inpaint_v1 --model-dir models/video_inpaint_v1/ --video --backward-step 8 --save-eval-step 1
```

- `-i`: データセットディレクトリ
- `--model-dir`: モデル出力先ディレクトリ（`eval/`配下に学習進捗サンプルを保存）
- `--save-eval-step`: 評価データごとの保存間隔（1: 全保存、20: 20データごと保存）

動画モデル学習時は`--batch-size 1`固定です（12フレーム/データ）。`--backward-step`はGradient Accumulationの回数です。Mini-batchの代わりにGradient Accumulationを使用します。

デフォルトでは200エポックで周期的学習率スケジューラを使用します（40エポック毎に学習率リセット）。リセット直後は一時的に精度が下がります。

途中停止後は`--resume`を追加することで中断箇所から再開可能です。

```
python train.py inpaint -i ./data/video_inpaint_v1 --model-dir models/video_inpaint_v1/ --video --backward-step 8 --save-eval-step 1 --resume
```

2回目以降の訓練やファインチューニングでは、周期的学習率からSchedule-Free AdamW(固定学習率)に切替えます。前回の学習結果を初期値として使用するために`--resume`、 スケジューラーを初期化するために`--reset-state`を指定します。
必要に応じてこのコマンドを複数回繰り返します。経験的には、3回目くらいまでは結果が改善します（600 epoch程度)。

```
python train.py inpaint -i ./data/video_inpaint_v1 --model-dir models/video_inpaint_v1/ --video --backward-step 8 --save-eval-step 1 --optimizer adamw_schedulefree --resume --reset-state
```

また学習済みモデルのチェックポイントファイルを指定して開始することもできます。この場合、`--model-dir`からではなく`--checkpoint-file`からモデルの初期ウェイトが読み込まれます。
```
python train.py inpaint -i ./data/video_inpaint_v1 --model-dir models/video_inpaint_v1/ --video --backward-step 8 --save-eval-step 1 --optimizer adamw_schedulefree --checkpoint-file iw3/pretrained_models/hub/checkpoints/iw3_light_video_inpaint_v1_20250919.pth
```

## 3. 学習済みモデルの利用

現時点では特別な実行時オプションはありません。

`iw3/inpaint_utils.py`の`VIDEO_MODEL_URL`を、`--model-dir`配下に保存された`inpaint.light_video_inpaint_v1.pth`へのフルパスに書き換えてください。

## 4. コード構成

- `create_training_data.py`：`iw3/training/inpaint/create_training_data_video.py`を呼び出し
- `train.py`：`iw3/training/inpaint/trainer.py` および `iw3/training/inpaint/dataset_video.py`を利用

各種詳細はコードをご参照ください。不明点はAIサポート等をご活用ください。
