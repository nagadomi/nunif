#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "$0"); pwd)

${SCRIPT_DIR}/smoke_data.sh
${SCRIPT_DIR}/smoke_waifu2x.sh
${SCRIPT_DIR}/smoke_stlizer.sh
${SCRIPT_DIR}/smoke_iw3.sh
