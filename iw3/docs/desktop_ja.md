# iw3 desktop streaming

(警告: これは非常に実験的なツールです。
 今のところMeta QuestとPICO 4で動作することが分かっています。VisionProでは動作しません。
 またLinux/Waylandで非常に遅い。)

iw3.desktopはPCのデスクトップ画面を3D変換してWiFi経由でストリーミング配信するツールです。
Meta Quest上のブラウザからサイドバイサイド3Dとして視聴できます。

PC上に表示されたあらゆる画像と動画をリアルタイムで視聴できます。

PCを操作するには、PCのキーボートとマウスを使います。オーディオもPCから使用します。ディスプレイだけMeta Questを使います。

GUIウィンドウや文字の深度推定結果は多分よくないので、3D酔いに注意してください。
基本的には画像や動画を全画面再生する使い方を想定しています。

## セキュリティに関する注意

iw3.desktopは、デフォルトではパスワードなしのHTTPサーバーを起動します。
同じネットワーク内であれば他のPCからアクセスされる可能性があることに注意してください。

ベーシック認証のパスワードを`--passward`オプションで設定できます。

## PC上でのサーバー起動

以下のコマンドで起動します。

```
python -m iw3.desktop
```

Windows上でnunif-windows-packageを使っている場合は、`nunif-prompt.bat`を起動してコンソール上からコマンドを入力してください。

起動に成功すると以下のようなのメッセージが表示されます。

```
Open http://192.168.11.6:1303
Estimated FPS = 30.24, Streaming FPS = 0.00
```
(`192.168.11.6`のアドレス部分は、ネットワーク環境によって異なります)

(New) または以下のコマンドでGUIが起動します。

```
python -m iw3.desktop.gui
```

![iw3-desktop-gui](https://github.com/user-attachments/assets/18175b2a-a027-42ce-ae5c-a9ee7ae178e5)

(Windowsでnunif-windows-packageを使っている場合は、`iw3-desktop-gui.bat`を実行してください。それがない場合は`update-installer.bat`と`update.bat`を実行すると現れます。)

ファイヤウォールのダイアログが表示された場合は許可してください。

URLをPCのブラウザで開いて動画が再生できるかチェックしてください。ウェブページはGoogle ChromeとMeta Quest 2 Browserで動作確認しています。Firefoxでは動作しません。

LAN内のPCアドレスが正しく検出されてない場合は、`--bind-addr`オプションで指定できます。
```
python -m iw3.desktop --bind-addr 192.168.1.2
```

## Meta Quest上での視聴

Meta Quest 2で動作確認しています。

Meta Quest上で以下の手順により動画を3D再生できます。

1. `Browser`を起動する
2. サーバーのURLを入力する
3. (オプション) URLをお気に入りに登録する
4. 動画を再生する
5. Browserの右上のアイコンから**ブラウザ**を全画面にする
6. 動画の右下のアイコンから**動画**を全画面にする
7. Browserの下のフロントメニューのスクリーンアイコンから`ディスプレイモード > 3Dサイドバイサイド`に設定する
8. (オプション) カーブウィンドウに設定する

動画とブラウザを両方とも全画面にしなければディスプレイモードが変更できないことに注意してください。

その後はPCのキーボードとマウスを使って表示されているスクリーンを操作できます。

## オプション

### PICO 4用のオプション

ユーザーからの報告によるとPICO 4のブラウザは動画をFull SBSで表示します。

`--full-sbs`オプションでストリーミング動画をFull SBSに変更できます。

```
python -m iw3.desktop --full-sbs
```
デフォルトはHalf SBSです。Meta QuestのブラウザはHalf SBSにしか対応していません。

### 解像度

`--stream-height`オプションで画面の縦サイズを指定できます。デフォルトは1080pxです。

```
python -m iw3.desktop --stream-height 720
```

### FPS

`--stream-fps`オプションでストリーミングのFPSを指定できます。デフォルトは15FPSです。

```
python -m iw3.desktop --stream-fps 30
```

`Estimated FPS`が指定されたFPSより著しく低い場合、PCの性能は指定されたFPSを処理するのに十分ではありません。

`--batch-size 1`で処理している理由でFPSは動画変換時よりかなり低くなります。

また、おそらくブラウザの制限により`Streaming FPS = 30`より高いFPSは達成できません。

### MJPEG 設定

`--stream-quality`でJPEG品質を指定できます。 (0-100)
```
python -m iw3.desktop --stream-quality 80
```
デフォルトは90です。低い値を指定すると、ネットワークトラフィックが削減されます。

### ステレオ設定

GUI/CLIと同じオプションが指定できます。

```
python -m iw3.desktop --depth-model ZoeD_Any_N --divergence 2 --convergence 0.5 --resolution 518
```

デフォルトは、`--depth-model Any_V2_S --divergence 1 --convergence 1`です。

### ネットワーク

`--bind-addr`と`--port`オプションでHTTPサーバーを起動するアドレスを指定できます。
```
python -m iw3.desktop --port 7860
```

サーバーをインターネットに公開する場合 (オススメしません)
```
python -m iw3.desktop --bind-addr 0.0.0.0 --port 7860
```

### 認証

`--user`と`--password`オプションでHTTPベーシック認証を設定できます。

```
python -m iw3.desktop --password iw3
```
```
python -m iw3.desktop --user admin --password 1234
```
