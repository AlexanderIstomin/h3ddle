#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

"$repository_root/Scripts/check-public-boundary.sh"
"$repository_root/Scripts/check-untracked-sources.sh"
xcodegen generate --spec "$repository_root/project.yml" --project "$repository_root"
swift test --package-path "$repository_root/Packages/H3ddleKit"
swift build --package-path "$repository_root/Engine"
engine_handshake=$(
    "$repository_root/Engine/.build/debug/H3ddleEngineService" \
        < "$repository_root/Engine/ProtocolFixtures/handshake.jsonl"
)
case "$engine_handshake" in
    *'"kind":"ready"'*) ;;
    *)
        echo "Engine capability handshake failed." >&2
        exit 1
        ;;
esac
case "$engine_handshake" in
    *'"modelInspection"'*'"videoGeneration"'*'"imageGeneration"'*'"standaloneAudioGeneration"'*'"referenceInputs"'*) ;;
    *)
        echo "Engine generation capabilities are incomplete." >&2
        exit 1
        ;;
esac
xcodebuild \
    -project "$repository_root/H3ddle.xcodeproj" \
    -scheme H3ddle \
    -destination 'platform=macOS' \
    -derivedDataPath "$repository_root/DerivedData" \
    CODE_SIGNING_ALLOWED=NO \
    -quiet \
    build-for-testing

# Building the UI tests was never the same as running them, and the gap
# let app-layer regressions through a green suite for a long time. These
# drive the real app, so they are the only check that covers the seam
# between the interface and the engine.
xcodebuild \
    -project "$repository_root/H3ddle.xcodeproj" \
    -scheme H3ddle \
    -destination 'platform=macOS' \
    -derivedDataPath "$repository_root/DerivedData" \
    -only-testing:H3ddleUITests \
    CODE_SIGNING_ALLOWED=NO \
    -quiet \
    test-without-building

test -x "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Helpers/H3ddleEngineService"
test -f "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Resources/H3Engine/h3_shaders.metal"
