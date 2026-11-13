import glob
import struct
import zlib
from datetime import datetime, timedelta, timezone
from os import path


def _resolve_git_dir(root: str | None = None) -> str | None:
    if root is None:
        git_dir = path.normpath(path.join(path.dirname(__file__), "..", "..", ".git"))
    else:
        git_dir = path.join(root, ".git")

    if path.isdir(git_dir):
        return git_dir
    if path.isfile(git_dir):
        with open(git_dir, mode="r", encoding="utf-8") as f:
            line = f.readline().strip()
        if line.startswith("gitdir: "):
            target = line[8:].strip()
            if not path.isabs(target):
                target = path.normpath(path.join(path.dirname(git_dir), target))
            if path.isdir(target):
                return target
    return None


def _resolve_head_sha(git_dir: str) -> str | None:
    head_path = path.join(git_dir, "HEAD")
    if not path.isfile(head_path):
        return None
    with open(head_path, mode="r", encoding="utf-8") as f:
        line = f.readline().strip()

    if line.startswith("ref: "):
        ref = line[5:].strip()
        ref_path = path.join(git_dir, *ref.split("/"))
        if path.isfile(ref_path):
            with open(ref_path, mode="r", encoding="utf-8") as rf:
                return rf.readline().strip()
        packed_refs = path.join(git_dir, "packed-refs")
        if path.isfile(packed_refs):
            with open(packed_refs, mode="r", encoding="utf-8") as pf:
                for pline in pf:
                    pline = pline.strip()
                    if not pline or pline.startswith(("#", "^")):
                        continue
                    parts = pline.split(" ", 1)
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
        return None

    if len(line) == 40:
        return line
    return None


def _read_loose_commit(git_dir: str, sha: str) -> bytes | None:
    obj_path = path.join(git_dir, "objects", sha[:2], sha[2:])
    if not path.isfile(obj_path):
        return None
    with open(obj_path, "rb") as f:
        data = zlib.decompress(f.read())
    header, _, body = data.partition(b"\x00")
    if not header.startswith(b"commit "):
        return None
    return body


def _read_packed_commit(git_dir: str, sha: str) -> bytes | None:
    target = bytes.fromhex(sha)
    first_byte = target[0]
    pack_dir = path.join(git_dir, "objects", "pack")
    for idx_path in sorted(glob.glob(path.join(pack_dir, "pack-*.idx"))):
        with open(idx_path, "rb") as f:
            if f.read(8) != b"\xfftOc\x00\x00\x00\x02":
                continue
            fanout = struct.unpack(">256I", f.read(256 * 4))
            total = fanout[255]
            start = fanout[first_byte - 1] if first_byte > 0 else 0
            end = fanout[first_byte]
            if start >= end:
                continue

            names_offset = 8 + 256 * 4
            f.seek(names_offset + start * 20)
            found_idx = -1
            for i in range(start, end):
                if f.read(20) == target:
                    found_idx = i
                    break
            if found_idx < 0:
                continue

            offsets_offset = names_offset + total * 20 + total * 4
            f.seek(offsets_offset + found_idx * 4)
            (offset,) = struct.unpack(">I", f.read(4))
            if offset & 0x80000000:
                large_idx = offset & 0x7FFFFFFF
                f.seek(offsets_offset + total * 4 + large_idx * 8)
                (offset,) = struct.unpack(">Q", f.read(8))

        pack_path = idx_path[:-4] + ".pack"
        if not path.isfile(pack_path):
            continue
        with open(pack_path, "rb") as pf:
            pf.seek(offset)
            byte = pf.read(1)[0]
            obj_type = (byte >> 4) & 0x07
            while byte & 0x80:
                byte = pf.read(1)[0]
            if obj_type != 1:
                return None
            decomp = zlib.decompressobj()
            chunks: list[bytes] = []
            while not decomp.eof:
                buf = pf.read(4096)
                if not buf:
                    break
                chunks.append(decomp.decompress(buf))
            return b"".join(chunks)
    return None


def _parse_commit_date(body: bytes) -> datetime | None:
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line:
            break
        if line.startswith("committer "):
            parts = line.rsplit(" ", 2)
            if len(parts) == 3:
                try:
                    ts = int(parts[1])
                    tz_str = parts[2]
                    sign = -1 if tz_str.startswith("-") else 1
                    tz_digits = tz_str.lstrip("+-")
                    hours = int(tz_digits[:2])
                    minutes = int(tz_digits[2:4])
                    tz = timezone(timedelta(hours=sign * hours, minutes=sign * minutes))
                    return datetime.fromtimestamp(ts, tz=tz)
                except ValueError:
                    return None
    return None


def get_last_commit_date(root: str | None = None) -> datetime | None:
    """Return the committer date of HEAD without invoking git subprocess."""
    git_dir = _resolve_git_dir(root)
    if git_dir is None:
        return None
    sha = _resolve_head_sha(git_dir)
    if sha is None:
        return None
    body = _read_loose_commit(git_dir, sha)
    if body is None:
        body = _read_packed_commit(git_dir, sha)
    if body is None:
        return None
    return _parse_commit_date(body)


def get_current_branch(root=None):
    if root is None:
        head_path = path.join(path.dirname(__file__), "..", "..", ".git", "HEAD")
    else:
        head_path = path.join(root, ".git", "HEAD")

    if path.exists(head_path):
        with open(head_path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line[0:5] == "ref: ":
                    branch_path = line[5:]
                    branch_name = branch_path.split("/")[-1]
                    return branch_name
    return None


def _test_get_current_branch():
    print(get_current_branch())
    print(get_current_branch(root="dummy_path"))
    print(get_last_commit_date())


if __name__ == "__main__":
    _test_get_current_branch()
