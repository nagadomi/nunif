#!/bin/bash -e

BASE_DIR="tests/data/race_condition"
DURATION="1.0"
TOTAL_DURATION="3.0"

mkdir -p "$BASE_DIR"

# Generate 3-scene concatenated SDR video with audio (640x360, 30fps, 3s, bt709 yuv420p)
if [ ! -f "${BASE_DIR}/sd.mkv" ]; then
    echo "Generating ${BASE_DIR}/sd.mkv..."
    ffmpeg -y \
        -f lavfi -i "gradients=size=640x360:rate=30:n=8:seed=1" \
        -f lavfi -i "smptebars=size=640x360:rate=30" \
        -f lavfi -i "mandelbrot=size=640x360:rate=30" \
        -f lavfi -i "sine=frequency=1000:duration=${TOTAL_DURATION}:sample_rate=44100" \
        -filter_complex "[0:v]trim=duration=${DURATION},setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv[v0]; \
                         [1:v]trim=duration=${DURATION},setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv[v1]; \
                         [2:v]trim=duration=${DURATION},setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv[v2]; \
                         [v0][v1][v2]concat=n=3:v=1:a=0[v]" \
        -map "[v]" -map 3:a -pix_fmt yuv420p -vcodec libx264 -crf 16 -preset superfast \
        -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
        -acodec aac "${BASE_DIR}/sd.mkv"
fi

# Generate 3-scene concatenated HDR video with audio (1280x720, 30fps, 3s, bt2020/pq yuv420p10le)
if [ ! -f "${BASE_DIR}/hdr.mkv" ]; then
    echo "Generating ${BASE_DIR}/hdr.mkv..."
    ffmpeg -y \
        -f lavfi -i "gradients=size=1280x720:rate=30:n=8:seed=1" \
        -f lavfi -i "smptebars=size=1280x720:rate=30" \
        -f lavfi -i "mandelbrot=size=1280x720:rate=30" \
        -f lavfi -i "sine=frequency=1000:duration=${TOTAL_DURATION}:sample_rate=44100" \
        -filter_complex "[0:v]trim=duration=${DURATION},format=yuv420p10le,setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv[v0]; \
                         [1:v]trim=duration=${DURATION},format=yuv420p10le,setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv[v1]; \
                         [2:v]trim=duration=${DURATION},format=yuv420p10le,setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv[v2]; \
                         [v0][v1][v2]concat=n=3:v=1:a=0[v]" \
        -map "[v]" -map 3:a -pix_fmt yuv420p10le -vcodec libx265 -crf 16 -preset superfast -x265-params "log-level=error" \
        -colorspace bt2020nc -color_primaries bt2020 -color_trc smpte2084 -color_range tv \
        -acodec aac "${BASE_DIR}/hdr.mkv"
fi

# Generate SDR video without audio
if [ ! -f "${BASE_DIR}/sd_noaudio.mkv" ]; then
    echo "Generating ${BASE_DIR}/sd_noaudio.mkv..."
    ffmpeg -y -i "${BASE_DIR}/sd.mkv" -c:v copy -an "${BASE_DIR}/sd_noaudio.mkv"
fi

# Generate HDR video without audio
if [ ! -f "${BASE_DIR}/hdr_noaudio.mkv" ]; then
    echo "Generating ${BASE_DIR}/hdr_noaudio.mkv..."
    ffmpeg -y -i "${BASE_DIR}/hdr.mkv" -c:v copy -an "${BASE_DIR}/hdr_noaudio.mkv"
fi
