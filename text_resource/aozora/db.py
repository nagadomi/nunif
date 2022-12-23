import csv
import os
from os import path
from dataclasses import dataclass
from enum import Enum
from glob import glob


DATA_DIR = path.join(path.dirname(__file__), "data")


@dataclass
class AozoraRecord():
    author_id: str
    author: str
    title_id: str
    title: str
    kana_type: str
    file_path: str
    file_size: int


class AozoraDB():
    KANA_TYPE_MODERN = "新字新仮名"

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.csv_file = path.join(data_dir, "list_person_all.csv")
        self.load()

    def load(self, modern_only=False):
        # 人物ID,著者名,作品ID,作品名,仮名遣い種別,翻訳者名等,入力者名,校正者名,状態,状態の開始日,底本名,出版社名,入力に使用した版,校正に使用した版

        self.data = []
        with open(self.csv_file, mode="r", encoding="cp932") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if modern_only and row["仮名遣い種別"] != AozoraDB.KANA_TYPE_MODERN:
                    continue
                file_path = self.find_file_path(row["人物ID"], row["作品ID"])
                if file_path:
                    item = AozoraRecord(author_id=row["人物ID"],
                                        author=row["著者名"],
                                        title_id=row["作品ID"],
                                        title=row["作品名"],
                                        kana_type=row["仮名遣い種別"],
                                        file_path=file_path,
                                        file_size=path.getsize(file_path))
                    self.data.append(item)
                else:
                    pass

    def find_file_path(self, author_id, title_id):
        author_dir = path.join(self.data_dir, author_id, "files")
        if not path.isdir(author_dir):
            return None
        title_id = str(int(title_id, 10))  # remove zero padding
        text_dirs = [path.join(author_dir, ent) 
                     for ent in os.listdir(author_dir)
                     if ent.startswith(title_id + "_")]
        if not text_dirs:
            return None
        if len(text_dirs) > 1:
            def get_update_id(file_path):
                basename = path.splitext(path.basename(file_path))[0]
                key = basename.split("_")[-1]
                if key == "txt":
                    update_id = 0
                elif key == "ruby":
                    update_id = 1
                else:
                    update_id = int(key) if key.isdigit() else 0
                return update_id
            tmp = [(f, get_update_id(f)) for f in text_dirs]
            tmp = sorted(tmp, key=lambda l: l[1], reverse=True)
            text_dirs = [l[0] for l in tmp]
        texts = [path.join(text_dirs[0], text) for text in os.listdir(text_dirs[0])]
        if len(texts) == 0:
            print("warning: no text", text_dirs[0], file=sys.stderr)
            return None
        elif len(texts) > 1:
            print("warning multi text", texts, file=sys.stderr)
        return texts[0]

    @staticmethod
    def filter_modern(items):
        return [item for item in items if item.kana_type == AozoraDB.KANA_TYPE_MODERN]

    @staticmethod
    def order_by_size(items):
        return list(sorted(items, key=lambda item: item.file_size, reverse=True))

    def find_by_title(self, keyword):
        return [item for item in self.data if keyword in item.title]

    def find_by_author(self, keyword, modern_only=True, size_order=True, limit=None):
        items = [item for item in self.data if keyword in item.author]
        if modern_only:
            items = self.filter_modern(items)
        if size_order:
            items = self.order_by_size(items)
        if isinstance(limit, int):
            items = items[:limit]
        return items

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    from pprint import pprint
    db = AozoraDB()
    print(len(db))
    pprint(db.find_by_author("芥川 竜之介"))
    pprint(db.find_by_title("ねこ"))
