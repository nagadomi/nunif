# 画像用インペイントモデルの学習

## 1. 学習用データセットの作成

ソースとなる画像ファイルから、学習データを生成します。通常の2D画像を使用します。

### 画像データセットの準備

元となる画像データセットは、以下のディレクトリ構造です。

```
image_dataset/
├── train/
└── eval/
```

`train/`は学習用の画像データディレクトリです。`eval/`は評価用(検証用)の画像ディレクトリです。

画像は短い方の辺が1080px以上あることが推奨されます。それより小さい場合は拡大されます。(これは後で変更するかもしれません)

### データセット生成コマンド

以下のコマンドを実行し、学習データを作成します。

```
python create_training_data.py inpaint --dataset-dir /path/to/image_dataset --prefix image1 -o ./data/inpaint_v1/ --model-type Any_B
```

コマンド実行後、`./data/inpaint_v1/train/`、`./data/inpaint_v1/eval/`内にに各データが生成されます。ファイル名は`{prefix}_{model_type}_{SEQUENCE_NO}_(M|C).png`になります。`_C.png`がRGB画像、`_M.png`がインペイントマスクです。

複数のディレクトリや異なる`--model-type`で実行する場合は、`--prefix`と`--model-type`の組み合わせが衝突しないように注意してください。

### コマンドオプション一覧

`python create_training_data.py inpaint -h`で全オプションを表示できます。一部主要オプションは以下の通りです。

| オプション            | 説明
|----------------------|--------------------------------------------------------------
|`inpaint`             | 画像用データ作成スクリプトの呼び出し（固定値）
|`--dataset-dir`       | 入力画像データセットのパス
|`-o`                  | 出力先ディレクトリ
|`--prefix`            | データのプレフィックス（必須）
|`--model-type`        | 使用する深度推定モデル名（iw3準拠）
|`--resolution`        | 深度画像の解像度（iw3準拠）
|`--size`              | 画像サイズ（デフォルト: 512、学習時にはここから256x256タイルにクロップ）
|`--num-samples`       | ランダムクロップの回数（デフォルト: 2）。

### バッチ処理

複数画像データセットや異なる深度推定モデルの組み合わせで生成する場合は、シェルスクリプトやBATファイルの利用を推奨します。エラー時はプレフィクスを条件に該当データを削除し、再処理してください。

```bash
#!/bin/bash -e

OUTPUT_DIR=./data/inpaint_v1/

python create_training_data.py inpaint --dataset-dir /data/Flickr2K -o ${OUTPUT_DIR} --prefix flicker1 --model-type Any_B
python create_training_data.py inpaint --dataset-dir /data/Flickr2K -o ${OUTPUT_DIR} --prefix flicker2 --model-type ZoeD_Any_N
# ...
```

## 2. モデル学習

静止したデータセットを用い、画像インペイントモデルの学習を開始します。

```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_v1/
```

- `-i`: データセットディレクトリ
- `--model-dir`: モデル出力先ディレクトリ（`eval/`配下に学習進捗サンプルを保存）
- `--save-eval-step`: 評価データごとの保存間隔（デフォルト: 20、1: 全保存、20: 20バッチごと保存）

デフォルトでは200エポックで周期的学習率スケジューラを使用します（40エポック毎に学習率リセット）。リセット直後は一時的に精度が下がります。

途中停止後は`--resume`を追加することで中断箇所から再開可能です。

2回目以降の訓練やファインチューニングでは、周期的学習率からSchedule-Free AdamW(固定学習率)に切替えます。前回の学習結果を初期値として使用するために`--resume`、 スケジューラーを初期化するために`--reset-state`を指定します。
必要に応じてこのコマンドを複数回繰り返します。経験的には、3回目くらいまでは結果が改善します（600 epoch程度)。

```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_v1/ --optimizer adamw_schedulefree --resume --reset-state
```

また学習済みモデルのチェックポイントファイルを指定して開始することもできます。この場合、`--model-dir`からではなく`--checkpoint-file`からモデルの初期ウェイトが読み込まれます。
```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_v1/ --optimizer adamw_schedulefree --checkpoint-file iw3/pretrained_models/hub/checkpoints/iw3_light_inpaint_v1_20250919.pth
```

### GANを使った学習

`--discriminator`オプションを指定するとGANを使った学習が有効になります。敵対的損失を使うとチェックボードアーティファクトを軽減できます。

GANを使ったファインチューニングは以下のようなコマンドになります。

```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_gan_v1/ --optimizer adamw_schedulefree --max-epoch 30 --discriminator l3cffce --save-epoch --disable-hard-example --checkpoint-file models/inpaint_v1/inpaint.light_inpaint_v1.pth
```

現在、`--max-epoch`が多いと次第に敵対的損失の効果がなくなっていくため適当なところで止めてください。`--save-epoch`を指定すると、各epochのチェックポイントファイルが`{model_dir}/{epoch}`ディレクトリに保存されます。

## 3. 学習済みモデルの利用

現時点では特別な実行時オプションはありません。

`iw3/inpaint_utils.py`の`IMAGE_MODEL_URL`を、`--model-dir`配下に保存された`inpaint.light_inpaint_v1.pth`へのフルパスに書き換えてください。

## 4. コード構成

- `create_training_data.py`：`iw3/training/inpaint/create_training_data.py`を呼び出し
- `train.py`：`iw3/training/inpaint/trainer.py` および `iw3/training/inpaint/dataset.py`を利用

各種詳細はコードをご参照ください。不明点はAIサポート等をご活用ください。
