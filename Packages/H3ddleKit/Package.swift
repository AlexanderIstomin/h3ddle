// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "H3ddleKit",
  platforms: [.macOS(.v15)],
  products: [
    .library(name: "H3ddleCore", targets: ["H3ddleCore"]),
    .library(name: "H3ddleDesignSystem", targets: ["H3ddleDesignSystem"]),
    .library(name: "H3ddleEngineClient", targets: ["H3ddleEngineClient"]),
    .library(name: "H3ddleEngineProtocol", targets: ["H3ddleEngineProtocol"]),
    .library(name: "H3ddleGeneration", targets: ["H3ddleGeneration"]),
    .library(name: "H3ddleMedia", targets: ["H3ddleMedia"]),
    .library(name: "H3ddleModels", targets: ["H3ddleModels"]),
  ],
  targets: [
    .target(name: "H3ddleCore"),
    .target(name: "H3ddleDesignSystem"),
    .target(
      name: "H3ddleEngineClient",
      dependencies: ["H3ddleCore", "H3ddleEngineProtocol", "H3ddleGeneration"]
    ),
    .target(name: "H3ddleEngineProtocol"),
    .target(
      name: "H3ddleGeneration",
      dependencies: ["H3ddleCore", "H3ddleEngineProtocol"]
    ),
    .target(
      name: "H3ddleMedia",
      dependencies: ["H3ddleCore"]
    ),
    .target(name: "H3ddleModels"),
    .testTarget(
      name: "H3ddleCoreTests",
      dependencies: ["H3ddleCore"]
    ),
    .testTarget(
      name: "H3ddleEngineClientTests",
      dependencies: ["H3ddleEngineClient", "H3ddleEngineProtocol", "H3ddleGeneration"],
      exclude: ["Fixtures/fake-engine.py"]
    ),
    .testTarget(
      name: "H3ddleEngineProtocolTests",
      dependencies: ["H3ddleEngineProtocol", "H3ddleGeneration"]
    ),
    .testTarget(
      name: "H3ddleGenerationTests",
      dependencies: ["H3ddleCore", "H3ddleGeneration"]
    ),
    .testTarget(
      name: "H3ddleMediaTests",
      dependencies: ["H3ddleCore", "H3ddleMedia"]
    ),
    .testTarget(
      name: "H3ddleModelsTests",
      dependencies: ["H3ddleModels"]
    ),
  ],
  swiftLanguageModes: [.v6]
)
