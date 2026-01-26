import av
from av.sidedata.sidedata import Type as SideDataType
import ctypes


class AVRational(ctypes.Structure):
    _fields_ = [
        ("num", ctypes.c_int),
        ("den", ctypes.c_int),
    ]

    def float(self):
        return self.num / self.den if self.den else 0.0

    def __repr__(self):
        return f"AVRational({self.num}/{self.den})"


class AVMasteringDisplayMetadata(ctypes.Structure):
    _fields_ = [
        ("display_primaries", AVRational * 2 * 3),
        ("white_point", AVRational * 2),
        ("min_luminance", AVRational),
        ("max_luminance", AVRational),
        ("has_primaries", ctypes.c_int),
        ("has_luminance", ctypes.c_int),
    ]

    def __repr__(self):
        r = (self.display_primaries[0][0], self.display_primaries[0][1])
        g = (self.display_primaries[1][0], self.display_primaries[1][1])
        b = (self.display_primaries[2][0], self.display_primaries[2][1])
        white_point = (self.white_point[0], self.white_point[1])
        return (
            f"{self.__class__.__name__}("
            f"display_primaries={[r, g, b]}, "
            f"white_point={white_point}, "
            f"min_luminance={self.min_luminance}, "
            f"max_luminance={self.max_luminance}, "
            f"has_primaries={self.has_primaries}, "
            f"has_luminance={self.has_luminance})"
        )

    def to_x265_params(self):

        if not self.has_primaries:
            return None

        prim = self.display_primaries
        Gx, Gy = prim[1][0].float(), prim[1][1].float()
        Bx, By = prim[2][0].float(), prim[2][1].float()
        Rx, Ry = prim[0][0].float(), prim[0][1].float()
        Wx, Wy = self.white_point[0].float(), self.white_point[1].float()

        def xy(v):
            return int(round(v * 50000))

        def lum(v):
            return int(round(v * 10000))

        s = (
            f"master-display="
            f"G({xy(Gx)},{xy(Gy)})"
            f"B({xy(Bx)},{xy(By)})"
            f"R({xy(Rx)},{xy(Ry)})"
            f"WP({xy(Wx)},{xy(Wy)})"
        )

        if self.has_luminance:
            maxLum = self.max_luminance.float()
            minLum = self.min_luminance.float()
            s += f"L({lum(maxLum)},{lum(minLum)})"

        return s


class AVContentLightMetadata(ctypes.Structure):
    _fields_ = [
        ("MaxCLL", ctypes.c_uint),
        ("MaxFALL", ctypes.c_uint),
    ]

    def __repr__(self):
        return f"{self.__class__.__name__}(MaxCLL={self.MaxCLL}, MaxFALL={self.MaxFALL})"

    def to_x265_params(self):
        return f"max-cll={self.MaxCLL},{self.MaxFALL}"


class HDRMetadata:
    def __init__(self, master_display, max_cll, is_hdr):
        self.master_display = master_display
        self.max_cll = max_cll
        if self.max_cll is not None and self.max_cll.MaxCLL == 0 and self.max_cll.MaxFALL == 0:
            self.max_cll = None

        self.is_hdr = is_hdr

    def has_data(self):
        return self.master_display is not None or self.max_cll is not None

    def to_x265_params(self):
        params = []
        if self.is_hdr:
            params += ["hdr10=1", "no-hdr10-opt=1", "no-dhdr10-opt=1", "repeat-headers=1"]
        if self.has_data():
            params += [d.to_x265_params() for d in [self.master_display, self.max_cll] if d is not None]
        return params


def is_hdr(stream):
    # print(stream.codec_context.color_primaries, stream.codec_context.color_trc)
    return (stream.codec_context.color_primaries == 9 and
            stream.format.components[0].bits > 8)


def get_hdr_metadata(input_path):
    MAX_FRAMES = 10
    with av.open(input_path, mode="r", metadata_errors="ignore") as container:
        if len(container.streams.video) == 0 or not is_hdr(container.streams.video[0]):
            return HDRMetadata(None, None, False)

        master_display = None
        max_cll = None

        try:
            for i, frame in enumerate(container.decode(video=0)):
                for sd in frame.side_data:
                    if (
                        master_display is None and
                        sd.type == SideDataType.MASTERING_DISPLAY_METADATA and
                        sd.buffer_size == ctypes.sizeof(AVMasteringDisplayMetadata)
                    ):
                        master_display = ctypes.cast(
                            sd.buffer_ptr, ctypes.POINTER(AVMasteringDisplayMetadata)
                        ).contents

                    elif (
                        max_cll is None and
                        sd.type == SideDataType.CONTENT_LIGHT_LEVEL and
                        sd.buffer_size == ctypes.sizeof(AVContentLightMetadata)
                    ):
                        max_cll = ctypes.cast(
                            sd.buffer_ptr, ctypes.POINTER(AVContentLightMetadata)
                        ).contents

                if master_display is not None and max_cll is not None:
                    break

                if i >= MAX_FRAMES:
                    break
        except av.error.FFMpegError:
            return HDRMetadata(None, None, True)

    return HDRMetadata(master_display, max_cll, True)


def _test():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video file")

    args = parser.parse_args()
    meta = get_hdr_metadata(args.input)
    print(meta.has_data(), meta.to_x265_params())


if __name__ == "__main__":
    _test()
