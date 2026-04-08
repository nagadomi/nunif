#!/bin/bash -e


BASE_DIR="tests/data/smoke"
DURATION="1.1"


generate_video() {
    local codec=$1
    local name=$2
    local pix_fmt=$3
    local colorspace=$4
    local primaries=$5
    local trc=$6
    local range=$7
    local input_filter=$8

    # Build setparams filter
    local sp=""
    [ "$colorspace" != "undefined" ] && sp="${sp}:colorspace=${colorspace}"
    [ "$primaries" != "undefined" ] && sp="${sp}:color_primaries=${primaries}"
    [ "$trc" != "undefined" ] && sp="${sp}:color_trc=${trc}"
    [ "$range" != "undefined" ] && sp="${sp}:range=${range}"
    sp=${sp#:} # Remove leading colon

    local filter_chain="${input_filter}"
    if [ -n "$sp" ]; then
        filter_chain="${filter_chain},setparams=${sp}"
    fi

    local out_args=()
    if [ "$codec" == "ffv1" ]; then
        out_args+=("-vcodec" "ffv1")
    elif [ "$codec" == "h264" ]; then
        out_args+=("-vcodec" "libx264" "-crf" "16" "-preset" "superfast" "-tune" "fastdecode")
    elif [ "$codec" == "hevc" ]; then
        out_args+=("-vcodec" "libx265" "-crf" "16" "-preset" "superfast" "-x265-params" "log-level=error")
    fi

    [ "$colorspace" != "undefined" ] && out_args+=("-colorspace" "$colorspace")
    [ "$primaries" != "undefined" ] && out_args+=("-color_primaries" "$primaries")
    [ "$trc" != "undefined" ] && out_args+=("-color_trc" "$trc")
    [ "$range" != "undefined" ] && out_args+=("-color_range" "$range")

    echo "Generating [$codec]: $name"
    ffmpeg -y -f lavfi -i "${filter_chain}" -t "${DURATION}" -pix_fmt "${pix_fmt}" "${out_args[@]}" "$BASE_DIR/${name}.mkv"
}


mkdir -p "$BASE_DIR"

# tests/data/sd.mkv
if [ ! -f ${BASE_DIR}/sd.mkv ]; then
   generate_video h264 sd yuv420p bt709 bt709 bt709 tv "gradients=size=640x360:rate=30:n=8:seed=1"
fi

for pix_fmt in "yuv444p" "yuv422p" "gbrp"; do
    if [ ! -f ${BASE_DIR}/h264_${pix_fmt}.mkv ]; then
        generate_video h264 h264_${pix_fmt} ${pix_fmt} bt709 bt709 bt709 tv "gradients=size=640x360:rate=30:n=8:seed=1"
    fi
    if [ ! -f ${BASE_DIR}/hevc_${pix_fmt}.mkv ]; then
        generate_video hevc hevc_${pix_fmt} ${pix_fmt} bt709 bt709 bt709 tv "gradients=size=640x360:rate=30:n=8:seed=1"
    fi
done

# tests/data/hdr.mkv
if [ ! -f ${BASE_DIR}/hdr.mkv ]; then
   generate_video hevc hdr yuv420p10le bt2020nc bt2020 smpte2084 tv "gradients=size=1280x720:rate=30:n=8:seed=1"
fi

# tests/data/sd.png
if [ ! -f ${BASE_DIR}/sd.png ]; then
    ffmpeg -i ${BASE_DIR}/sd.mkv  -frames:v 1 ${BASE_DIR}/sd.png
fi
