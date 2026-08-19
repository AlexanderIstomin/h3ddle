#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

"$repository_root/Scripts/check-public-boundary.sh"
"$repository_root/Scripts/check-untracked-sources.sh"

# The LTX VAE decoder is native C and its tile planner deliberately has no
# model-weight dependency. Exercise the real planner here: a Swift duplicate
# could pass while the memory-bound production path drifted.
ltx_tiling_test="${TMPDIR:-/tmp}/h3ddle-ltx-tiling-test"
xcrun clang -std=c11 -Wall -Wextra -Werror \
    "$repository_root/Engine/Sources/LTX/ltx_tiling.c" \
    "$repository_root/Engine/Tests/LTXTilingTests/ltx_tiling_test.c" \
    -I"$repository_root/Engine/Sources/LTX" \
    -o "$ltx_tiling_test"
"$ltx_tiling_test"

xcodegen generate --spec "$repository_root/project.yml" --project "$repository_root"
swift test --package-path "$repository_root/Packages/H3ddleKit"
swift test --package-path "$repository_root/Engine"
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
    *'"modelInspection"'*'"videoGeneration"'*'"imageGeneration"'*'"ltxGeneration"'*'"standaloneAudioGeneration"'*'"referenceInputs"'*) ;;
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

# The UI tests drive the real application, which means launching a window that
# takes keyboard focus. On a developer's machine that interrupts whatever they
# were doing, and a dismissed window fails the run — so locally they are
# opt-in, and CI, where nothing is competing for the screen, sets the variable.
#
# Run them here with:  H3DDLE_UI_TESTS=1 Scripts/ci.sh
if [ "${H3DDLE_UI_TESTS:-0}" != "1" ]; then
    echo "UI tests: skipped (set H3DDLE_UI_TESTS=1 to run them locally)."
    test -x "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Helpers/H3ddleEngineService"
    test -f "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Resources/H3Engine/h3_shaders.metal"
    exit 0
fi

# Building the UI tests was never the same as running them, and the gap
# let app-layer regressions through a green suite for a long time. These
# drive the real app, so they are the only check that covers the seam
# between the interface and the engine.
#
# The count is asserted rather than trusted. Under -quiet a run that executed
# nothing at all still exits zero and prints a summary indistinguishable from
# a suite that passed, so a test bundle that stopped being picked up would
# look exactly like success. Comparing against the number of test functions
# on disk means tests can only leave the suite deliberately.
ui_test_log="$repository_root/DerivedData/ui-tests.log"
set +e
xcodebuild \
    -project "$repository_root/H3ddle.xcodeproj" \
    -scheme H3ddle \
    -destination 'platform=macOS' \
    -derivedDataPath "$repository_root/DerivedData" \
    -only-testing:H3ddleUITests \
    CODE_SIGNING_ALLOWED=NO \
    test-without-building > "$ui_test_log" 2>&1
ui_test_status=$?
set -e
if [ "$ui_test_status" -ne 0 ]; then
    grep -E 'error:|failed|XCTAssert' "$ui_test_log" | head -40 >&2
    echo "UI tests failed; full log at $ui_test_log" >&2
    exit 1
fi

expected_ui_tests=$(grep -rho 'func test[A-Za-z0-9_]*' \
    "$repository_root/Tests/H3ddleUITests" | wc -l | tr -d ' ')
executed_ui_tests=$(sed -n 's/.*Executed \([0-9][0-9]*\) test.*/\1/p' \
    "$ui_test_log" | sort -rn | head -1)
skipped_ui_tests=$(sed -n 's/.*with \([0-9][0-9]*\) test[s]* skipped.*/\1/p' \
    "$ui_test_log" | sort -rn | head -1)
if [ "${executed_ui_tests:-0}" -ne "$expected_ui_tests" ]; then
    echo "UI tests: ran ${executed_ui_tests:-0} of $expected_ui_tests on disk." >&2
    echo "A suite that runs nothing exits zero, so this is treated as a" >&2
    echo "failure. Full log at $ui_test_log" >&2
    exit 1
fi
echo "UI tests: $expected_ui_tests ran, ${skipped_ui_tests:-0} skipped."

test -x "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Helpers/H3ddleEngineService"
test -f "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Resources/H3Engine/h3_shaders.metal"
