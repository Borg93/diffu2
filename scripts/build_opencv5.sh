#!/usr/bin/env bash
# Build OpenCV 5.x from source with Python bindings (headless) and install cv2 into the active venv.
#
# WHY THIS EXISTS: OpenCV 5 has no usable PyPI wheel (stable pip = 4.13; the *-rolling 5.0.0 is a dead
# 2022 nightly that won't import on py3.12+/numpy2 — verified). Compiling is the only way to get the
# real cv5 (faster warpAffine/warpPerspective/remap, TRUCO findContours, the new DNN engine).
#
# diffu_page uses only the cv2 API that is byte-for-byte identical on 4.x and 5.x, so this is a pure
# performance/feature upgrade — nothing in the module changes whether you run 4.13 or a source cv5.
#
# HOW: activate the project's uv venv first (so cv2 installs next to numpy), then run this on the box
# where you generate pages. Tunables via env: OPENCV_REF, BUILD_DIR, JOBS, WITH_ORT.
#
# CAVEAT: a full OpenCV build is ~20-40 min and needs a C++17 toolchain + cmake + python dev headers;
# this recipe is NOT CI-build-verified (no toolchain in the data-gen container). Adjust flags per box.
set -euo pipefail

OPENCV_REF="${OPENCV_REF:-5.x}"                 # branch/tag of opencv/opencv to build
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/opencv5-build}"
JOBS="${JOBS:-$(nproc)}"
WITH_ORT="${WITH_ORT:-OFF}"                      # ON downloads ONNX Runtime for validate.py's detector

PY="${PYTHON:-python}"
command -v cmake >/dev/null || { echo "need cmake (apt install cmake / brew install cmake)"; exit 1; }
"$PY" -c "import numpy" 2>/dev/null || { echo "activate a venv with numpy first (uv sync --extra page)"; exit 1; }

SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
NUMPY_INC="$("$PY" -c 'import numpy; print(numpy.get_include())')"
PY_EXEC="$("$PY" -c 'import sys; print(sys.executable)')"

mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
[ -d opencv ] || git clone --depth 1 --branch "$OPENCV_REF" https://github.com/opencv/opencv.git

# Build a lean, HEADLESS cv2 with just what diffu_page needs (core, imgproc, imgcodecs, dnn). Drop the
# BUILD_LIST line for a full build; add OPENCV_EXTRA_MODULES_PATH + opencv_contrib for contrib modules.
cmake -S opencv -B build \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_LIST=core,imgproc,imgcodecs,dnn,python3 \
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_EXECUTABLE="$PY_EXEC" \
  -D PYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INC" \
  -D OPENCV_PYTHON3_INSTALL_PATH="$SITE" \
  -D WITH_QT=OFF -D WITH_GTK=OFF -D WITH_WIN32UI=OFF -D HIGHGUI_ENABLE_PLUGINS=OFF \
  -D BUILD_opencv_apps=OFF -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_DOCS=OFF \
  -D WITH_ONNXRUNTIME="$WITH_ORT" $( [ "$WITH_ORT" = ON ] && echo "-D DOWNLOAD_ONNXRUNTIME=ON" ) \
  -D CMAKE_INSTALL_PREFIX=/usr/local

cmake --build build -j "$JOBS"
cmake --install build       # places the cv2 python extension into $SITE

"$PY" -c "import cv2; print('OpenCV', cv2.__version__, '->', cv2.__file__)"
echo "done. diffu_page now runs on cv5 with no code changes."
