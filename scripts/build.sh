#!/usr/bin/env bash
# Build script for Linux & Raspberry Pi 5 (ARM64 / aarch64)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== 1. Checking System Dependencies ==="
if command -v apt-get &> /dev/null; then
    echo "Installing required apt-get dependencies..."
    sudo apt-get update -qq && sudo apt-get install -y -qq \
        build-essential cmake libpcl-dev libcgal-dev libboost-all-dev libeigen3-dev libtbb-dev
fi

echo "=== 2. Building Native C++ Engine ==="
cmake -B native/build -S native -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release -j$(nproc 2>/dev/null || echo 4)

echo "=== 3. Copying Built Shared Library ==="
mkdir -p src/svr_roughness/native
if [ -f native/build/Release/SurfInspectNative.so ]; then
    cp -f native/build/Release/SurfInspectNative.so src/svr_roughness/native/
elif [ -f native/build/libSurfInspectNative.so ]; then
    cp -f native/build/libSurfInspectNative.so src/svr_roughness/native/
elif [ -f native/build/SurfInspectNative.so ]; then
    cp -f native/build/SurfInspectNative.so src/svr_roughness/native/
fi

echo "=== 4. Running Unit Tests ==="
python3 -m pytest -v tests

echo "=== 5. Packaging Release Artifacts (.whl & .tar.gz) ==="
rm -rf dist/
if [ -d "vcpkg/scripts/buildsystems" ]; then
    export CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$(pwd)/vcpkg/scripts/buildsystems/vcpkg.cmake"
fi
python3 -m build --sdist --no-isolation
python3 -m build --wheel --no-isolation

echo "Successfully built release packages in dist/!"
ls -la dist/
