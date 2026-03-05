import os
import sys
from os import path
import hashlib
import json
from nunif.utils.home_dir import ensure_home_dir, is_nunif_home_set


def md5(s):
    return hashlib.md5((s + "iw3").encode()).hexdigest()


def get_cache_dir():
    if is_nunif_home_set():
        cache_dir = path.join(ensure_home_dir("iw3"), "scene_cache")
    else:
        cache_dir = path.join(path.dirname(__file__), "..", "tmp", "iw3_scene_cache")

    if not path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cache_path(input_video_path, max_fps):
    cache_dir = get_cache_dir()

    max_fps_key = str(max_fps)
    path_key = path.abspath(input_video_path)
    size_key = str(path.getsize(input_video_path))
    mtime_key = str(path.getmtime(input_video_path))
    param = f"{max_fps_key} {path_key} {size_key} {mtime_key}"
    cache_filename = md5(param) + ".json"

    cache_path = path.join(cache_dir, cache_filename)
    return cache_path


def save_cache(input_video_path, pts, max_fps, start_time, end_time):
    cache_path = get_cache_path(input_video_path, max_fps)
    data = {
        "pts": sorted(list(pts)),
        "max_fps": max_fps,
        "start_time": start_time,
        "end_time": end_time,
    }
    with open(cache_path, mode="w", encoding="utf-8") as f:
        json.dump(data, f)


def is_within_range(data, start_time, end_time):
    def to_sec(val, default):
        if val is None:
            return default
        try:
            parts = str(val).split(":")
            if len(parts) > 3:
                raise ValueError
            units = [1, 60, 3600]
            total_sec = sum(int(c) * u for c, u in zip(reversed(parts), units))
            return max(total_sec, 0)
        except (ValueError, TypeError):
            raise ValueError("time must be hh:mm:ss, mm:ss or ss format")

    data_start = to_sec(data.get("start_time"), 0)
    data_end = to_sec(data.get("end_time"), float("inf"))
    query_start = to_sec(start_time, 0)
    query_end = to_sec(end_time, float("inf"))
    return data_start <= query_start and query_end <= data_end


def try_load_cache(input_video_path, max_fps, start_time, end_time):
    cache_path = get_cache_path(input_video_path, max_fps=max_fps)
    if path.exists(cache_path):
        try:
            with open(cache_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            if not is_within_range(data, start_time, end_time):
                return None

            return set(data["pts"])
        except Exception as e:  # noqa
            print(f"{cache_path}: {e}", file=sys.stderr)
            return None
    else:
        return None


def purge_cache(input_video_path, max_fps):
    cache_path = get_cache_path(input_video_path, max_fps)
    if path.exists(cache_path):
        os.unlink(cache_path)


def list_cache_files():
    cache_dir = get_cache_dir()
    return (path.join(cache_dir, fn)
            for fn in os.listdir(cache_dir)
            if fn.endswith(".json"))


def purge_cache_all():
    for cache_path in list_cache_files():
        os.unlink(cache_path)


def _test():
    import unittest

    class TestSceneCache(unittest.TestCase):
        def setUp(self):
            purge_cache_all()
            self.video_path = "tmp/scene_cache_test.mp4"
            self.max_fps = 30
            self.pts = [1, 2, 3, 4, 5]
            with open(self.video_path, "w") as f:
                f.write("dummy data")

        def tearDown(self):
            if os.path.exists(self.video_path):
                os.remove(self.video_path)
            purge_cache_all()

        def test_save_and_load_success(self):
            save_cache(self.video_path, self.pts, self.max_fps, "00:00:00", "00:00:10")
            loaded_pts = try_load_cache(self.video_path, self.max_fps, "00:00:01", "00:00:05")
            self.assertEqual(loaded_pts, self.pts)

        def test_out_of_range(self):
            save_cache(self.video_path, self.pts, self.max_fps, "00:00:05", "00:00:10")

            self.assertIsNone(try_load_cache(self.video_path, self.max_fps, "00:00:00", "00:00:10"))
            self.assertIsNone(try_load_cache(self.video_path, self.max_fps, "00:00:05", "00:00:15"))

        def test_corrupted_json(self):
            cache_path = get_cache_path(self.video_path, self.max_fps)

            with open(cache_path, "w") as f:
                f.write("{ invalid json ...")
            result = try_load_cache(self.video_path, self.max_fps, "00:00:00", "00:00:10")
            self.assertIsNone(result)

        def test_file_modified(self):
            save_cache(self.video_path, self.pts, self.max_fps, "00:00:00", "00:00:10")

            mtime = os.path.getmtime(self.video_path)
            os.utime(self.video_path, (mtime + 1, mtime + 1))

            result = try_load_cache(self.video_path, self.max_fps, "00:00:00", "00:00:10")
            self.assertIsNone(result)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestSceneCache))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _test()
