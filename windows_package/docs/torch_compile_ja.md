# Using torch.compile on Windows

この手順では、Windows環境で torch.compile を有効にする方法を説明します。
torch.compile を使用することで、一部のモデルで実行速度が向上する可能性がありますが、初回実行時に追加のコンパイル時間が発生します。

## GPU と CPU 共通の手順

インストールスクリプトは `torch_compile` フォルダ内にあります。存在しない場合は`update-installer.bat`を実行してください。

1. `enable_long_path.reg` を実行します
2. コンピューターを再起動します
3. `install_python_dev.bat` を実行します

### `enable_long_path.reg` – 長いファイルパスの有効化

Windows では、パスの最大長が 260 文字に制限されています。
`enable_long_path.reg` を実行してコンピューターを再起動することで、この制限を解除できます。

詳細は Microsoft のドキュメント [パスの最大長の制限](https://learn.microsoft.com/ja-jp/windows/win32/fileio/maximum-file-path-limitation?tabs=registry) を参照してください。

### `install_python_dev.bat` – Python の開発用ファイルをインストール

nunif-windows が使用している Embeddable Python には、開発用のヘッダーファイルやライブラリが含まれていません。
このスクリプトを実行することで、それらのファイルが追加インストールされます。

## GPU での使用

GPU 向けの torch.compile 対応は PyTorch/Triton 非公式 です。
以下のリポジトリにある Windows 用の Triton を使用します:
https://github.com/woct0rdho/triton-windows

インストールスクリプトは `torch_compile` フォルダ内にあります。

1. `install_triton_windows.bat` を実行します

その後、デバイスとして NVIDIA GPU を選択し、`torch.compile` チェックボックスを有効にできるか確認してください。

コンパイルキャッシュは次のディレクトリに保存されます:

`C:\Users\<username>\AppData\Local\Temp\torchinductor_*`

不要な場合は削除しても問題ありません。

## CPU での使用

CPU 用の torch.compile を使用するには、Visual Studio 2022 または 2019 のインストールが必要です。
ここでは Visual Studio 2022 Community Edition のインストール方法を示します。

ダウンロードリンク:
https://aka.ms/vs/17/release/vs_community.exe

1. 「Desktop development with C++」を選択します
2. 「言語パック」タブで English を選択します（必須）
3. インストールを実行します

インストール後、デバイスとして CPU を選択し、`torch.compile` チェックボックスを有効にできるか確認してください。

アップデートなどで動作しなくなった場合は、次のキャッシュを削除して再試行してください:

`C:\Users\<username>\AppData\Local\Temp\torchinductor_*`
