from os import path
import sys


assert sys.version_info > (3, 9)


class Char():
    jis1 = None
    hiragana = None
    hiragana_basic = None

    @classmethod
    @property
    def JIS1(cls):
        if cls.jis1 is None:
            with open(path.join(path.dirname(__file__), "jis1.txt"),
                      mode="r", encoding="utf-8") as f:
                cls.jis1 = set(f.read().strip())
        return cls.jis1

    @classmethod
    @property
    def HIRAGANA(cls):
        if cls.hiragana is None:
            with open(path.join(path.dirname(__file__), "hiragana.txt"),
                      mode="r", encoding="utf-8") as f:
                cls.hiragana = set(f.read().strip())
        return cls.hiragana

    @classmethod
    @property
    def HIRAGANA_BASIC(cls):
        if cls.hiragana_basic is None:
            with open(path.join(path.dirname(__file__), "hiragana_basic.txt"),
                      mode="r", encoding="utf-8") as f:
                cls.hiragana_basic = set(f.read().strip())
        return cls.hiragana_basic


if __name__ == "__main__":
    print(Char.JIS1)
    print(Char.HIRAGANA)
    print(Char.HIRAGANA_BASIC)
