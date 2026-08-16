// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "H3ddleEngine",
  platforms: [.macOS(.v15)],
  products: [
    .executable(name: "H3ddleEngineService", targets: ["H3ddleEngineService"])
  ],
  dependencies: [
    .package(path: "../Packages/H3ddleKit")
  ],
  targets: [
    .target(
      name: "H3Native",
      path: ".",
      sources: [
        "Sources/H3Native/H3NativeBridge.c",
        "Vendor/h3.c/h3.c",
        "Vendor/h3.c/h3_host.c",
        "Vendor/h3.c/h3_safetensors.c",
        "Vendor/h3.c/h3_weights.c",
        "Vendor/h3.c/h3_text_encoder.c",
        "Vendor/h3.c/h3_dit_schedule.c",
        "Vendor/h3.c/h3_dit.c",
        "Vendor/h3.c/h3_video_vae.c",
        "Vendor/h3.c/h3_tae.c",
        "Vendor/h3.c/h3_video_encoder.c",
        "Vendor/h3.c/h3_audio_vae.c",
        "Vendor/h3.c/h3_ffmpeg.c",
        "Vendor/h3.c/h3_terminal.c",
        "Vendor/h3.c/h3_vision_encoder.c",
        "Vendor/h3.c/h3_multimodal.c",
        "Vendor/h3.c/h3_metal.m",
        "Vendor/h3.c/h3_gpu.m",
        "Vendor/h3.c/h3_avwriter.m",
        "Vendor/h3.c/h3_avreader.m",
        "Vendor/h3.c/h3_tokenizer.m",
        // Stable Audio 3, kept outside the vendored engine so upstream merges
        // stay clean while still reaching its headers and GPU layer.
        "Sources/SA3/sa3_generate.c",
        "Sources/SA3/sa3_text.c",
        "Sources/SA3/sa3_dit.c",
        "Sources/SA3/sa3_dit_gpu.c",
        "Sources/SA3/sa3_decoder.c",
        "Sources/SA3/sa3_tokenizer.m",
      ],
      publicHeadersPath: "Sources/H3Native/include",
      cSettings: [
        .define("_DARWIN_C_SOURCE"),
        .headerSearchPath("Vendor/h3.c"),
        .headerSearchPath("Sources/SA3"),
        // h3_ffmpeg.c uses SSIZE_MAX, whose SDK expansion requires LONG_MAX.
        .unsafeFlags(["-include", "limits.h"]),
      ],
      linkerSettings: [
        .linkedFramework("Foundation"),
        .linkedFramework("Metal"),
        .linkedFramework("MetalPerformanceShaders"),
        .linkedFramework("MetalPerformanceShadersGraph"),
        .linkedFramework("Accelerate"),
        .linkedFramework("AVFoundation"),
        .linkedFramework("CoreMedia"),
        .linkedFramework("CoreVideo"),
        .linkedFramework("VideoToolbox"),
        .linkedFramework("AudioToolbox"),
        .linkedFramework("ImageIO"),
        .linkedFramework("CoreGraphics"),
        .linkedLibrary("icucore"),
        .linkedLibrary("m"),
      ]
    ),
    .executableTarget(
      name: "H3ddleEngineService",
      dependencies: [
        "H3Native",
        .product(name: "H3ddleEngineProtocol", package: "H3ddleKit"),
      ]
    ),
  ],
  swiftLanguageModes: [.v6],
  cLanguageStandard: .c11
)
