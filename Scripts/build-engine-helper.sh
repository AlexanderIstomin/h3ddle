#!/bin/sh
set -eu

if [ -n "${SRCROOT:-}" ]; then
    repository_root=$SRCROOT
else
    repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fi
engine_configuration=debug
if [ "${CONFIGURATION:-Debug}" = "Release" ]; then
    engine_configuration=release
fi

swift build \
    --package-path "$repository_root/Engine" \
    --configuration "$engine_configuration"

engine_binary="$repository_root/Engine/.build/$engine_configuration/H3ddleEngineService"
helper_directory="$TARGET_BUILD_DIR/$CONTENTS_FOLDER_PATH/Helpers"
shader_directory="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH/H3Engine"
mkdir -p "$helper_directory"
mkdir -p "$shader_directory"
/bin/rm -f "$helper_directory/h3_shaders.metal"
/usr/bin/ditto "$engine_binary" "$helper_directory/H3ddleEngineService"
/usr/bin/ditto \
    "$repository_root/Engine/Vendor/h3.c/h3_shaders.metal" \
    "$shader_directory/h3_shaders.metal"

air="$shader_directory/h3_shaders.air"
metallib="$shader_directory/h3_shaders.metallib"
metal_bin=$(/usr/bin/xcrun --sdk macosx --find metal 2>/dev/null || true)
if [ -n "$metal_bin" ] && [ -x "$metal_bin" ] \
    && "$metal_bin" -c \
        "$repository_root/Engine/Vendor/h3.c/h3_shaders.metal" \
        -o "$air" >/dev/null 2>&1 \
    && /usr/bin/xcrun --sdk macosx metallib "$air" -o "$metallib" >/dev/null 2>&1; then
    /bin/rm -f "$air"
else
    /bin/rm -f "$air" "$metallib"
fi

if [ "${CODE_SIGNING_ALLOWED:-YES}" != "NO" ] && [ -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" ]; then
    /usr/bin/codesign \
        --force \
        --sign "$EXPANDED_CODE_SIGN_IDENTITY" \
        --options runtime \
        "$helper_directory/H3ddleEngineService"
fi
