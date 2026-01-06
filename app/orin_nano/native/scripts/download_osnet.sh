#!/bin/bash
# Download OSNet x0.25 ONNX model for ReID
# Source: https://github.com/PeppermintSummer/OSNet_tensorrt

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NATIVE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$NATIVE_DIR"

echo "=== Downloading OSNet x0.25 ONNX model ==="

# Download from GitHub LFS (raw file)
ONNX_URL="https://github.com/PeppermintSummer/OSNet_tensorrt/raw/master/osnet_x0_25_market_dynamic.onnx"
ONNX_FILE="osnet_x0_25.onnx"

if [ -f "$ONNX_FILE" ]; then
    echo "ONNX file already exists: $ONNX_FILE"
else
    echo "Downloading $ONNX_URL..."
    curl -L -o "$ONNX_FILE" "$ONNX_URL"
    echo "Downloaded: $ONNX_FILE"
fi

# Check file size (should be ~1MB for x0.25)
FILE_SIZE=$(stat -f%z "$ONNX_FILE" 2>/dev/null || stat -c%s "$ONNX_FILE" 2>/dev/null)
echo "File size: $FILE_SIZE bytes"

if [ "$FILE_SIZE" -lt 100000 ]; then
    echo "WARNING: File seems too small. It might be a GitHub LFS pointer."
    echo "Try downloading manually from the GitHub page."
    exit 1
fi

echo ""
echo "=== Converting to TensorRT engine ==="
echo "Run this command in the l4t-tensorrt container:"
echo ""
echo "  /usr/src/tensorrt/bin/trtexec \\"
echo "    --onnx=$ONNX_FILE \\"
echo "    --saveEngine=osnet_x0_25.engine \\"
echo "    --fp16"
echo ""
echo "Or run this script inside Docker:"
echo "  docker run --rm -v \$(pwd):/work -w /work nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime \\"
echo "    /usr/src/tensorrt/bin/trtexec --onnx=$ONNX_FILE --saveEngine=osnet_x0_25.engine --fp16"
