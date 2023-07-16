from os import path
from . web.webgen.gen import load_locales as load_webgen_locales


WEBGEN_TERMS = [
    "artwork", "scan", "photo",
    "noise_reduction", "nr_none", "nr_low", "nr_medium", "nr_high", "nr_highest",
    "upscaling", "up_none",
]


def merge_en(webgen_locales, lang):
    t = webgen_locales["en"].copy()
    t.update(webgen_locales[lang])
    return t


def merge_locales(webgen_locales, locales, webgen_lang, lang):
    locale = merge_en(webgen_locales, webgen_lang)
    assert len([term for term in WEBGEN_TERMS if term not in locale]) == 0
    webgen_locale = {term: locale[term] for term in WEBGEN_TERMS}
    locales[lang].update(webgen_locale)


LOCALES = {
    "ja_JP": {
        "Input": "入力",
        "Output": "出力",
        "Choose a file": "ファイルを選択",
        "Choose a directory": "フォルダを選択",
        "Set the same directory": "同じフォルダを設定",
        "Resume": "処理を再開",
        "Skip processing of files that already exist": "すでに存在するファイルの処理をスキップする",

        "Error": "エラー",
        "Initializing": "初期化中",
        "Cancelled": "キャンセル済み",
        "Finished": "完了",
        "Confirm": "確認",
        "already exists. Overwrite?": "すでに存在します。上書きしますか？",

        "Superresolution": "超解像",
        "Model": "モデル",
        "Anime Style Art, Cliparts": "アニメスタイルアート、クリップアート",
        "Manga, Anime Screencaps, Anime Style Art for more clear results":
        "漫画, アニメキャプチャ, イラスト, よりくっきりした結果",
        "Photograph": "写真・実写",
        "Old version, Art model, fast": "旧バージョン, イラストモデル, 速い",
        "Old version, Art model, veryfast": "旧バージョン, イラストモデル, とても速い",
        "Old version, Photo model, veryfast": "旧バージョン, 写真モデル, とても速い",

        "Video Encoding": "動画圧縮",

        "Max FPS": "最大フレームレート",
        "Pixel Format": "ピクセルフォーマット",
        "CRF": "CRF(固定品質)",
        "Preset": "プリセット",
        "Tune": "チューニング",

        "Video Filter": "ビデオフィルタ",
        "Deinterlace": "デインターレース",
        "Rotate": "回転",
        "Left 90 (counterclockwise)": "左に90°回転 (半時計回り)",
        "Right 90 (clockwise)": "右に90°回転 (時計回り)",
        "Add grain noise": "ノイズを追加",

        "Processor": "プロセッサ",
        "Device": "デバイス",
        "Batch Size": "バッチサイズ",
        "Tile Size": "分割サイズ",
        "FP16 (fast)": "FP16 (速い)",
    },
    "en_US": {}
}
# Merge from webgen locales
_WEBGEN_LOCALES = load_webgen_locales(path.join(path.dirname(__file__), "web", "webgen", "locales"))
merge_locales(_WEBGEN_LOCALES, LOCALES, "ja", "ja_JP")
merge_locales(_WEBGEN_LOCALES, LOCALES, "en", "en_US")

# For Windows
LOCALES["Japanese_Japan"] = LOCALES["ja_JP"]
