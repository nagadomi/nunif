# nunif windows package

Windowsユーザー向けのオンラインインストーラーです。
waifu2x-gui、iw3-guiの実行に必要なものをひとつのフォルダ内にインストールします。

waifu2x-guiは超解像ソフトウェアです。画像をいい感じに拡大します。

iw3-guiは実写のあらゆる画像・動画を3D画像・3D動画に変換するソフトウェアです。VRで本当に見たかった画像・動画をVRデバイスで3Dメディアとして見れるようになります。

# 前提条件

- Windows 7 以降. Windows 10 以降が推奨. Windows Server 2008 r2以降.
- Visual C++ 再頒布可能パッケージ https://aka.ms/vs/16/release/vc_redist.x64.exe

# ダウンロードとインストール

| 説明                                                                          | 画像
| ------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------- 
| 1. ZIPファイルをダウンロードして好きな場所に配置します。ここでは`Document\software`とします。| https://github.com/nagadomi/nunif/releases/download/0.0.0/nunif-windows.zip
| 2. ダウンロードしたZIPファイルを右クリックしてプロパティを表示します。        | ![zip menu ja](https://github.com/nagadomi/nunif/assets/287255/238f8f0c-b858-4ba8-a798-66e3bd02a43d)
| 3. セキュリティの**許可する**にチェックを付けて`適用`をクリックします。       | ![zip unlock ja](https://github.com/nagadomi/nunif/assets/287255/72cebc81-586a-4fff-b306-0e33ef7e04e6)
| 4. ZIPファイルを右クリックして**すべて展開**します。                          | ![zip extract ja](https://github.com/nagadomi/nunif/assets/287255/4a59cc8b-b974-422d-af98-4afd095bc649)
| 5. 展開されたファイルの中にある`install.bat`を実行します。                    | ![file list](https://github.com/nagadomi/nunif/assets/287255/27fae8f2-c8bc-497b-b554-fc5c804a7c3e)
| 6. 黒い画面が表示されインストールが行われます。                               | ![install cmd](https://github.com/nagadomi/nunif/assets/287255/7587f561-4eec-4568-b916-8ae3c6f143cb)
| 7. すべて成功すると`Successfly installed nunif`のあとに`続行するには何かキーを押してください`と表示されます。何かキーを押すと画面が閉じます。 | ![sucess ja](https://github.com/nagadomi/nunif/assets/287255/ffce086f-bddb-489a-a6eb-b4552f8f5226)


以上でインストールは完了です。

## waifu2x

`waifu2x-gui.bat`を実行します。最初の実行時は画面が出てくるまでに時間がかかることがあります。

画面の詳細は、[waifu2x GUI](../../waifu2x/docs/gui_ja.md)を参照してください。

## iw3

`iw3-gui.bat`を実行します。最初の実行時は画面が出てくるまでに時間がかかることがあります。

画面の詳細は、[iw3 GUI](../../iw3/docs/gui_ja.md)を参照してください。

## 更新

`update.bat`を実行するとソースコードをgithubの最新状態に更新します。
ソースコードに何らか変更を行っているとリセットされることがあるので注意してください。

何かおかしくなって再構成したい場合は、`python`,`git`,`nunif`フォルダを削除してから`update.bat`を実行してください。

`update-installer.bat`を実行すると`update.bat`自体を更新します。

更新は頻繁に行われているので、取得タイミングによっては何がバグっていることがあるかもしれません。
報告があれば修正します。

## GUI設定のリセット

GUIは入力状態が保存されています。リセットしたい場合は、`nunif\tmp`フォルダを削除してください。

## アンインストール

フォルダごと削除してください。

## Windows Defender SmartScreenにブロックされる

![smart screen1 ja](https://github.com/nagadomi/nunif/assets/287255/10426aba-a411-42ae-bdc6-9e77a48bf3a4)

![smart screen2 ja](https://github.com/nagadomi/nunif/assets/287255/3625f0e3-8189-4275-b5ad-fadc755d02fa)

Windowsによって警告が表示される場合は、**詳細**をクリックすると`実行`ボタンが表示されます。一度実行すると次からは警告は出なくなります。

インターネットからダウンロードしたバッチファイルにはこの警告が表示されるようです。
上記のzipファイルのセキュリティ・アンブロックはこの許可を事前に与える操作です。

バッチファイルはテキスト形式なのでメモ帳等で中身を見て安全性を確認できます。
`update.bat`(`install.bat`)は以下を行っています。

- Pythonの公式サイトから組み込み用Pythonをダウンロードして(フォルダ内に)展開
- Git for WindowsのGithubリリースから最小構成のGitをダウンロードして(フォルダ内に)展開
- Githubからnunif(このプログラム)のリポジトリを(フォルダ内に)取得
- Pythonの依存ライブラリを(フォルダ内に)インストール
- 学習済みモデルファイルを(フォルダ内に)ダウンロード

意図的にはフォルダ外にアクセスしません。
