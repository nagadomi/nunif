import re
import math


# （）,(),『』,「」,〝〟,“”,"",'',``
# 512 is the threshold for skipping over a forgotten closing parenthesis
SPEECH_BLOCK_PATTERNS = (
    r"（[^（）]{0,512}）", r"\([^\(\)]{0,512}\)",
    r"『[^『』]{0,512}』", r"「[^「」]{0,512}」",
    r"〝[^〝〟]{0,512}〟", r"“[^“”]{0,512}”",
    r'"[^"]{0,512}"', r"'[^']{0,512}'", r"`[^`]{0,512}`"
)
SPEECH_BRACKETS = set("（）()『』「」〝〟“”""'`")


def separate_speech_lines(text):
    if isinstance(text, (list, tuple)):
        text = "\n".join(text)
    speech_blocks = sum([[m[0] for m in re.finditer(regex, text, re.M)]
                         for regex in SPEECH_BLOCK_PATTERNS], [])

    non_speech_text = text
    for block in sorted(speech_blocks, key=lambda line: len(line), reverse=True):
        non_speech_text = non_speech_text.replace(block, "")
    non_speech_text = re.sub(r"[\r\n]+", "\n", non_speech_text)
    non_speech_blocks = [block for block in non_speech_text.split("\n")
                         if all(b not in block for b in SPEECH_BRACKETS)]

    speech_lines = sum([split_sentence(block[1:-1].strip(" \r\n\t　"))
                        for block in speech_blocks], [])
    non_speech_lines = sum([split_sentence(block.strip(" \t　"))
                            for block in non_speech_blocks], [])

    return speech_lines, non_speech_lines


def split_sentence(text):
    text = re.sub(r"([。\.\?？！]+)", r"\1\n", text)
    text = re.sub(r"[\r\n]+", "\n", text)
    lines = remove_empty([line.strip(" 　\t") for line in text.split("\n")])
    return lines


def remove_punct(line):
    return re.sub(r"[。、\.,\?？！]+", "", line)


def remove_empty(lines):
    return [line for line in lines if line.strip(" 　\t\r\n")]


def filter_length(lines, min_len=0, max_len=math.inf):
    return [line for line in lines if min_len <= len(line) <= max_len]
