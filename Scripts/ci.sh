#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

"$repository_root/Scripts/check-public-boundary.sh"
"$repository_root/Scripts/check-untracked-sources.sh"
python3 -B "$repository_root/Scripts/test-repack-ltx-input-major.py"
python3 -B "$repository_root/Scripts/test-convert-turbo-package.py"
python3 -B "$repository_root/Scripts/test-convert-fasth3-package.py"
python3 -B "$repository_root/Scripts/test-run-with-timeout.py"

# The tokenizer JSON embedded in LTX-2.5 deliberately emits bare Gemma IDs;
# the LTX pipeline must add Gemma's BOS itself. Keep this outside the shared
# tokenizer so H3 tokenization cannot change with it.
ltx_prompt_test="${TMPDIR:-/tmp}/h3ddle-ltx-prompt-test"
xcrun clang -std=c11 -Wall -Wextra -Werror \
    "$repository_root/Engine/Sources/LTX/ltx_prompt.c" \
    "$repository_root/Engine/Tests/LTXPromptTests/ltx_prompt_test.c" \
    -I"$repository_root/Engine/Sources/LTX" \
    -o "$ltx_prompt_test"
"$ltx_prompt_test"

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

# The tiny LTX preview decoder ends in a learned 48-channel image followed by
# a 4x channel-to-space rearrangement. Keep that layout executable without
# requiring the optional checkpoint or a Metal device.
ltx_tae_geometry_test="${TMPDIR:-/tmp}/h3ddle-ltx-tae-geometry-test"
xcrun clang -std=c11 -Wall -Wextra -Werror \
    "$repository_root/Engine/Sources/LTX/ltx_tae_geometry.c" \
    "$repository_root/Engine/Tests/LTXTAETests/ltx_tae_geometry_test.c" \
    -I"$repository_root/Engine/Sources/LTX" \
    -o "$ltx_tae_geometry_test"
"$ltx_tae_geometry_test"

# LTX's vocoder emits interleaved stereo, while the shared H3 muxer accepts
# channel-major PCM. Exercise that engine boundary without model weights so a
# future fix to either audio path cannot silently corrupt the other one.
ltx_audio_layout_test="${TMPDIR:-/tmp}/h3ddle-ltx-audio-layout-test"
xcrun clang -std=c11 -Wall -Wextra -Werror \
    "$repository_root/Engine/Sources/LTX/ltx_resample.c" \
    "$repository_root/Engine/Tests/LTXAudioLayoutTests/ltx_audio_layout_test.c" \
    -I"$repository_root/Engine/Sources/LTX" \
    -o "$ltx_audio_layout_test"
"$ltx_audio_layout_test"

# The inpainting mask crosses the VAE's non-uniform temporal grouping and the
# transformer's 2x2 latent patches. Keep that geometry executable without
# requiring model weights or a GPU.
h3_inpaint_test="${TMPDIR:-/tmp}/h3ddle-h3-inpaint-test"
xcrun clang -std=c11 -Wall -Wextra -Werror \
    "$repository_root/Engine/Vendor/h3.c/h3_inpaint.c" \
    "$repository_root/Engine/Tests/H3InpaintTests/h3_inpaint_test.c" \
    -I"$repository_root/Engine/Vendor/h3.c" \
    -o "$h3_inpaint_test"
"$h3_inpaint_test"

# Recovery files are an on-disk compatibility boundary. Exercise atomic
# round-trip and checksum rejection without model weights or a GPU.
h3_checkpoint_test="${TMPDIR:-/tmp}/h3ddle-h3-checkpoint-test"
xcrun clang -std=c11 -Wall -Wextra -Werror -D_DARWIN_C_SOURCE \
    "$repository_root/Engine/Vendor/h3.c/h3_checkpoint.c" \
    "$repository_root/Engine/Tests/H3CheckpointTests/h3_checkpoint_test.c" \
    -I"$repository_root/Engine/Vendor/h3.c" \
    -o "$h3_checkpoint_test"
"$h3_checkpoint_test"

xcodegen generate --spec "$repository_root/project.yml" --project "$repository_root"
# Swift Testing normally puts every target in one process and runs tests
# concurrently. Several targets exercise process-global macOS facilities (Core
# Image, AVFoundation, URLProtocol, environment variables, and helper
# processes); Xcode 26 can terminate that shared runner during teardown even
# after the assertions pass. Discover targets from SwiftPM so new suites cannot
# be omitted, then give each target a fresh, serial runner process.
run_swift_test_targets() {
    package_path=$1
    swift_test_timeout_seconds=${H3DDLE_SWIFT_TEST_TIMEOUT_SECONDS:-120}
    test_targets=$(
        swift test list --package-path "$package_path" |
            sed -n 's/^\([[:alnum:]_]*Tests\)\..*/\1/p' |
            sort -u
    )
    if [ -z "$test_targets" ]; then
        echo "No Swift test targets discovered in $package_path." >&2
        exit 1
    fi
    for test_target in $test_targets; do
        attempt=1
        while [ "$attempt" -le 2 ]; do
            set +e
            python3 -B "$repository_root/Scripts/run-with-timeout.py" \
                "$swift_test_timeout_seconds" \
                swift test --no-parallel --package-path "$package_path" \
                --filter "$test_target"
            test_status=$?
            set -e
            if [ "$test_status" -eq 0 ]; then
                break
            fi
            if [ "$attempt" -eq 2 ]; then
                echo "Swift test target $test_target failed with status $test_status." >&2
                return "$test_status"
            fi
            if [ "$test_status" -eq 124 ]; then
                echo "Swift test target $test_target timed out; retrying once in a fresh process." >&2
            elif [ "$test_target" = "H3ddleEngineClientTests" ] && [ "$test_status" -eq 1 ]; then
                # Xcode 26 can surface teardown of this target's helper-process
                # fixtures as signal 5, which `swift test` normalizes to status
                # 1. Retry only this process-owning target; ordinary assertion
                # failures remain deterministic and fail on the second run.
                echo "Swift test target $test_target exited during process teardown; retrying once in a fresh process." >&2
            else
                echo "Swift test target $test_target failed with status $test_status." >&2
                return "$test_status"
            fi
            attempt=$((attempt + 1))
        done
    done
}

run_swift_test_targets "$repository_root/Packages/H3ddleKit"
run_swift_test_targets "$repository_root/Engine"
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
    *'"modelInspection"'*'"videoGeneration"'*'"imageGeneration"'*'"ltxGeneration"'*'"standaloneAudioGeneration"'*'"referenceInputs"'*'"videoInpainting"'*'"generationCheckpointing"'*) ;;
    *)
        echo "Engine generation capabilities are incomplete." >&2
        exit 1
        ;;
esac
# macOS can attach an access-control record to a UI-test runner after launch;
# cleaning derived products keeps a later signed run from trying to overwrite
# that protected executable in place.
xcodebuild \
    -project "$repository_root/H3ddle.xcodeproj" \
    -scheme H3ddle \
    -destination 'platform=macOS' \
    -derivedDataPath "$repository_root/DerivedData" \
    CODE_SIGN_IDENTITY=- \
    CODE_SIGNING_REQUIRED=YES \
    CODE_SIGNING_ALLOWED=YES \
    -quiet \
    clean build-for-testing

# A linker-signed executable inside an otherwise unsigned .app looks built but
# fails Gatekeeper's bundle validation as "damaged". Ad-hoc signing needs no
# certificate and makes local/UI-test products structurally valid.
codesign --verify --deep --strict \
    "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app"

# The UI tests drive the real application, which means launching a window that
# takes keyboard focus. Hosted runners can also lose the application process or
# accessibility tree after a successful launch. Keep execution opt-in while
# still compiling the UI test bundle above.
#
# Run them here with:  H3DDLE_UI_TESTS=1 Scripts/ci.sh
if [ "${H3DDLE_UI_TESTS:-0}" != "1" ]; then
    echo "UI tests: skipped (set H3DDLE_UI_TESTS=1 to run them locally)."
    test -x "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Helpers/H3ddleEngineService"
    test -f "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Resources/H3Engine/h3_shaders.metal"
    test -f "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Resources/H3Engine/h3_sol_attention.metal"
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
    CODE_SIGN_IDENTITY=- \
    CODE_SIGNING_REQUIRED=YES \
    CODE_SIGNING_ALLOWED=YES \
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
test -f "$repository_root/DerivedData/Build/Products/Debug/H3ddle.app/Contents/Resources/H3Engine/h3_sol_attention.metal"
