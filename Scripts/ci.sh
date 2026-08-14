#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

"$repository_root/Scripts/check-public-boundary.sh"
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

test -x "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Helpers/H3ddleEngineService"
test -f "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Resources/H3Engine/h3_shaders.metal"
