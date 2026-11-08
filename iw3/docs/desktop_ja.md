# iw3-desktop: PCデスクトップ3Dストリーミングツール

## 概要

`iw3-desktop`は、PCのデスクトップ画面をリアルタイムで3D変換し、WiFi経由でストリーミング配信するツールです。Meta QuestなどのVRデバイスのブラウザからサイドバイサイド3Dとして視聴できます。

PC上に表示されるあらゆる画像や動画をリアルタイムでVR空間で視聴することが可能です。PCの操作には、引き続きPCのキーボードとマウスを使用し、音声もPCから出力されます。VRデバイスはディスプレイとして機能します。

GUIウィンドウやテキストの深度推定結果は完全ではないため、3D酔いに注意が必要です。本ツールは、基本的に画像や動画を全画面再生する用途を想定しています。

Webストリーミングに加えて、ローカルビューワーとしてのウィンドウ表示にも対応しています。これは、ウィンドウまたは仮想モニターをキャプチャーする機能を持つ3DモニターやVRデスクトップソフトウェア（Virtual DesktopやBigscreenなど）の配信機能を活用するものです。

## 既知の問題

*   Meta QuestおよびPICO 4での動作は確認済みですが、VisionProでは動作しません。
*   Linuxでは表示サーバーによって画面取得方法を選択する必要があります（下記「画面取得方法」参照）。
    *   **Wayland**: `--screenshot obs` または `--screenshot kwcapture` (KDE Plasmaのみ) を使用します。
    *   **X11**: `--screenshot mss` を使用します。

## セキュリティに関する注意

`iw3-desktop`は、デフォルトでパスワードなしのHTTPサーバーを起動します。同一ネットワーク内の他のPCからアクセスされる可能性があるため、ご注意ください。

ベーシック認証のパスワードは`--password`オプションで設定できます。

## GUIの使用方法

Windowsで`nunif-windows-package`を使用している場合、`iw3-desktop-gui.bat`を実行することでGUIが起動します。もし`iw3-desktop-gui.bat`が存在しない場合は、`update-installer.bat`と`update.bat`を実行してください。

コマンドラインから起動する場合は、以下のコマンドを使用します。

```bash
python -m iw3.desktop.gui
```

![iw3-desktop-gui](https://github.com/user-attachments/assets/18175b2a-a027-42ce-ae5c-a9ee7ae178e5)

## CLIの使用方法

以下のコマンドでサーバーを起動します。

```bash
python -m iw3.desktop
```
(Windowsで`nunif-windows-package`を使用している場合、`nunif-prompt.bat`を起動し、コンソールから上記のコマンドを入力してください。)

サーバーが正常に起動すると、以下のようなメッセージが表示されます。

```
Open http://192.168.11.6:1303
Estimated FPS = 30.24, Streaming FPS = 0.00
```
(`192.168.11.6`のアドレスは、ネットワーク環境によって異なります。)

ファイアウォールのダイアログが表示された場合は、アクセスを許可してください。

PCのブラウザで表示されたURLを開き、動画が再生されることを確認してください。ウェブページはGoogle ChromeとMeta Quest 2 Browserで動作確認済みです。Firefoxでは動作しません。

LAN内のPCアドレスが正しく検出されない場合は、`--bind-addr`オプションでIPアドレスを指定できます。

```bash
python -m iw3.desktop --bind-addr 192.168.1.2
```

## Meta Questでの視聴方法

Meta Quest 2での動作を確認しています。以下の手順で3D動画を再生できます。

1.  `Browser`を起動します。
2.  サーバーのURLを入力します。
3.  (オプション) URLをお気に入りに登録します。
4.  動画を再生します。
5.  Browser右上のアイコンから**ブラウザ**を全画面表示にします。
6.  動画右下のアイコンから**動画**を全画面表示にします。
7.  Browser下部のフロントメニューのスクリーンアイコンから`ディスプレイモード > 3Dサイドバイサイド`に設定します。
8.  (オプション) カーブウィンドウに設定します。

ディスプレイモードは、動画とブラウザの両方が全画面表示になっている場合にのみ変更できる点にご注意ください。

その後はPCのキーボードとマウスを使用して、表示されているスクリーンを操作できます。

## オプション

### PICO 4用オプション

ユーザーからの報告によると、PICO 4のブラウザは動画をFull SBSで表示します。

`--full-sbs`オプションを使用することで、ストリーミング動画をFull SBSに変更できます。

```bash
python -m iw3.desktop --full-sbs
```
デフォルトはHalf SBSです。Meta QuestのブラウザはHalf SBSのみに対応しています。

### 解像度 (動画解像度)

`--stream-height`オプションで画面の垂直方向の解像度を指定できます。デフォルトは1080pxです。

```bash
python -m iw3.desktop --stream-height 720
```

### FPS

`--stream-fps`オプションでストリーミングのフレームレート（FPS）を指定できます。デフォルトは15FPSです。

```bash
python -m iw3.desktop --stream-fps 30
```

`Estimated FPS`が指定されたFPSよりも著しく低い場合、PCの性能が指定されたFPSの処理に追いついていないことを意味します。

`--batch-size 1`での処理のため、FPSは動画変換時よりもかなり低くなります。また、ブラウザの制限により、`Streaming FPS = 30`を超えるFPSは達成できない可能性があります。

### MJPEG 設定

`--stream-quality`オプションでJPEGの品質を指定できます（0-100）。

```bash
python -m iw3.desktop --stream-quality 80
```
デフォルトは90です。低い値を指定すると、ネットワークトラフィックが削減されます。

### 画面取得方法

`--screenshot` で画面取得バックエンドを選択します。

| 方法 | 対応環境 | モニター/ウィンドウ指定 | 説明 |
| --- | --- | --- | --- |
| `pil` | 全OS | 不可 | `PIL.ImageGrab`。WaylandではXWayland経由 |
| `mss` | Windows, X11, macOS | モニター (ウィンドウ: Windows/X11) | LinuxはX11のみ対応 |
| `wc_mp` | Windows | モニター + ウィンドウ | `windows_capture` (別プロセス) |
| `wc_cuda` | Windows + NVIDIA | モニター + ウィンドウ | `wc_cuda` |
| `obs` | Linux Wayland | デスクトップポータル経由 | `libobs` (`linux-pipewire` プラグイン、別プロセス) |
| `kwcapture` | Linux Wayland (KDE Plasma) | モニター + ウィンドウ | [kwcapture](https://pypi.org/project/kwcapture/) (`org.kde.KWin.ScreenShot2`) |

#### Linux Wayland (`obs` / `kwcapture`)

*   **`obs`**: `libobs` の `linux-pipewire` プラグインと `xdg-desktop-portal` を使用します。モニターやウィンドウの選択は起動時のシステムのポータルダイアログで行います。
    *   必要なライブラリ: `libobs`, `obs-plugins` (`linux-pipewire.so`), `libglib-2.0`, `libwayland-client` (例: `apt install obs-plugins libobs0t64` または `obs-studio`)
*   **`kwcapture`**: KWinの `ScreenShot2` D-Busインターフェースを使用します (KDE Plasma専用)。
    *   必要なパッケージ: `pip install kwcapture`

### ステレオ設定

GUI/CLIと同じオプションを指定できます。

```bash
python -m iw3.desktop --depth-model ZoeD_Any_N --divergence 2 --convergence 0.5 --resolution 518
```

デフォルトは、`--depth-model Any_V2_S --divergence 1 --convergence 1`です。

### ネットワーク

`--bind-addr`と`--port`オプションでHTTPサーバーを起動するアドレスとポートを指定できます。

```bash
python -m iw3.desktop --port 7860
```

サーバーをインターネットに公開する場合（**非推奨**）：

```bash
python -m iw3.desktop --bind-addr 0.0.0.0 --port 7860
```

### 認証

`--user`と`--password`オプションでHTTPベーシック認証を設定できます。

```bash
python -m iw3.desktop --password iw3
```

```bash
python -m iw3.desktop --user admin --password 1234
```

### ローカルビューワー

`--local-viewer`オプションを指定します。

- `--uncap-fps`: ディスプレイのリフレッシュレート上限を超えて描画します。
- `--fit-window`: アスペクト比を無視してビューワーウィンドウ全体に画面を引き伸ばしてフィットさせます。

CLIから指定した場合でもGUIウィンドウが表示されるため、wxpythonとOpenGLが必要です（`requirements-gui.txt`からインストールされます）。

CUDAを使用する実行では、フレームをGPUメモリから直接ウィンドウへコピーします（CUDA-GL Interop）。
これはウィンドウの描画とCUDAが同じGPU上で行われる場合のみ利用できます。そのため
Waylandセッションでコンポジターが内蔵GPUで動作している場合、PRIMEノートパソコンの場合、
またはドライバがバッファを共有できない場合には、次のようなメッセージが1回表示されます。

```
Failed to initialize CUDA-GL Interop: cudaGraphicsGLRegisterBuffer failed: invalid OpenGL or DirectX context (219) [Intel ... / CUDA NVIDIA GeForce RTX 3080 Ti]. Falling back to the CPU transfer path.
```

これは情報表示です。フレームはシステムメモリ経由でコピーされ、速度は低下しますが正しく表示されます。
Wayland固有の要件ではなく、画面取得自体はCUDAを用いません。

