#!/bin/bash -e

echo "**** ${0}"

TEST_VIDEO=tests/data/smoke/sd.mkv
OUTPUT_DIR=tests/data/smoke/stlizer

mkdir -p ${OUTPUT_DIR}
set -x

python -m stlizer -i ${TEST_VIDEO} -o ${OUTPUT_DIR}/out.mkv
