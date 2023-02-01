import re
import math
from nunif.utils import text as T


def load_content(text_file):
    content = []
    with open(text_file, mode="r", encoding="cp932") as f:
        lines = list(f.readlines())
        start = False
        for i, line in enumerate(lines):
            if not start:
                if line.startswith("----") and i + 1 < len(lines) and lines[i + 1] == "\n":
                    start = True
                continue

            line = re.sub(r"^[\s　]+", "", line.strip())
            if not line:
                continue
            if line[0] in {"＊", "＃", "［"}:
                continue
            if all([c == "―" for c in line]):
                continue

            line = re.sub(r"※［＃.*］", "", line)
            line = re.sub(r"＊[１２３４５６７８９０]*［＃.*］", "", line)
            line = re.sub(r"［＃.*］", "", line)
            line = re.sub(r"＊[１２３４５６７８９０　]*", "", line)
            line = re.sub(r"《.*》", "", line)
            line = re.sub(r"｜", "", line)
            if line.startswith("底本："):
                break
            if line:
                content.append(line)
    return "\n".join(content)


def load_resource(text_file):
    content = load_content(text_file)
    return T.separate_speech_lines(content)


def load_speech_lines(text_file, remove_punct=False,
                      min_len=3, max_len=math.inf):
    speech_lines, non_speech_lines = load_resource(text_file)
    if remove_punct:
        speech_lines = T.remove_empty([T.remove_punct(line) for line in speech_lines])

    return T.filter_length(speech_lines, min_len=min_len, max_len=max_len)


def load_non_speech_lines(text_file, remove_punct=False,
                          min_len=3, max_len=math.inf):
    speech_lines, non_speech_lines = load_resource(text_file)
    if remove_punct:
        non_speech_lines = T.remove_empty([T.remove_punct(line) for line in non_speech_lines])

    return T.filter_length(non_speech_lines, min_len=min_len, max_len=max_len)


def _test_load_content():
    from .db import AozoraDB
    db = AozoraDB()
    item = db.find_by_title("吾輩は猫である")[0]
    print(load_content(item.file_path))


def _test_load_resource():
    from .db import AozoraDB
    from pprint import pprint
    db = AozoraDB()
    item = db.find_by_title("吾輩は猫である")[0]
    speech_lines = load_speech_lines(item.file_path, remove_punct=True, min_len=10)
    pprint(speech_lines)
    print(item.file_path)


if __name__ == "__main__":
    _test_load_resource()
