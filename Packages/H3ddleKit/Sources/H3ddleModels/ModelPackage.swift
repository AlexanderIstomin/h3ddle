import Foundation

public enum ModelPackageRole: String, Codable, Equatable, Sendable {
  case transformer
  case textEncoder
  case videoVAE
  case audioVAE
  case runtimeMetadata
}

public enum ModelEngineCompatibility: String, Codable, Equatable, Sendable {
  case ready
  case adapterRequired
}

public struct ModelPackageFile: Codable, Equatable, Sendable, Identifiable {
  public let role: ModelPackageRole
  public let path: String
  public let byteCount: Int64
  public let sha256: String
  public let sourceRepository: String?
  public let sourceRevision: String?
  /// Path inside the source repository when it differs from the path this
  /// file takes inside the installed package.
  public let sourcePath: String?
  /// Absolute path of a machine-local file to install from (verified against
  /// sha256), tried before any download.
  public let localCandidatePath: String?
  /// The file has no download source: it must come from a local candidate or
  /// an identical file in another installed package.
  public let requiresLocalSource: Bool

  public var id: String { path }

  public init(
    role: ModelPackageRole,
    path: String,
    byteCount: Int64,
    sha256: String,
    sourceRepository: String? = nil,
    sourceRevision: String? = nil,
    sourcePath: String? = nil,
    localCandidatePath: String? = nil,
    requiresLocalSource: Bool = false
  ) {
    self.role = role
    self.path = path
    self.byteCount = byteCount
    self.sha256 = sha256
    self.sourceRepository = sourceRepository
    self.sourceRevision = sourceRevision
    self.sourcePath = sourcePath
    self.localCandidatePath = localCandidatePath
    self.requiresLocalSource = requiresLocalSource
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    role = try container.decode(ModelPackageRole.self, forKey: .role)
    path = try container.decode(String.self, forKey: .path)
    byteCount = try container.decode(Int64.self, forKey: .byteCount)
    sha256 = try container.decode(String.self, forKey: .sha256)
    sourceRepository = try container.decodeIfPresent(String.self, forKey: .sourceRepository)
    sourceRevision = try container.decodeIfPresent(String.self, forKey: .sourceRevision)
    sourcePath = try container.decodeIfPresent(String.self, forKey: .sourcePath)
    localCandidatePath = try container.decodeIfPresent(String.self, forKey: .localCandidatePath)
    requiresLocalSource =
      try container.decodeIfPresent(Bool.self, forKey: .requiresLocalSource) ?? false
  }

  public var displayName: String {
    URL(fileURLWithPath: path).lastPathComponent
  }
}

/// How generation defaults should configure themselves for a package.
public enum ModelGenerationProfile: String, Codable, Equatable, Sendable {
  case standard
  /// Step-distilled weights: few denoising passes, beta sigma spacing,
  /// high fidelity with loose prompt control.
  case turbo

  /// Overrides the quality preset's step default when set.
  public var defaultDenoisingSteps: Int? {
    self == .turbo ? 8 : nil
  }

  public var usesBetaSchedule: Bool {
    self == .turbo
  }
}

public struct ModelPackageManifest: Codable, Equatable, Sendable, Identifiable {
  public let schemaVersion: Int
  public let id: String
  public let displayName: String
  public let detail: String
  public let repository: String
  public let revision: String
  public let licenseName: String
  public let licenseURL: URL
  public let minimumUnifiedMemoryBytes: Int64
  public let compatibility: ModelEngineCompatibility
  public let generationProfile: ModelGenerationProfile
  public let files: [ModelPackageFile]

  public init(
    schemaVersion: Int = 1,
    id: String,
    displayName: String,
    detail: String,
    repository: String,
    revision: String,
    licenseName: String,
    licenseURL: URL,
    minimumUnifiedMemoryBytes: Int64,
    compatibility: ModelEngineCompatibility,
    generationProfile: ModelGenerationProfile = .standard,
    files: [ModelPackageFile]
  ) {
    self.schemaVersion = schemaVersion
    self.id = id
    self.displayName = displayName
    self.detail = detail
    self.repository = repository
    self.revision = revision
    self.licenseName = licenseName
    self.licenseURL = licenseURL
    self.minimumUnifiedMemoryBytes = minimumUnifiedMemoryBytes
    self.compatibility = compatibility
    self.generationProfile = generationProfile
    self.files = files
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
    id = try container.decode(String.self, forKey: .id)
    displayName = try container.decode(String.self, forKey: .displayName)
    detail = try container.decode(String.self, forKey: .detail)
    repository = try container.decode(String.self, forKey: .repository)
    revision = try container.decode(String.self, forKey: .revision)
    licenseName = try container.decode(String.self, forKey: .licenseName)
    licenseURL = try container.decode(URL.self, forKey: .licenseURL)
    minimumUnifiedMemoryBytes = try container.decode(
      Int64.self, forKey: .minimumUnifiedMemoryBytes
    )
    compatibility = try container.decode(ModelEngineCompatibility.self, forKey: .compatibility)
    generationProfile =
      try container.decodeIfPresent(ModelGenerationProfile.self, forKey: .generationProfile)
      ?? .standard
    files = try container.decode([ModelPackageFile].self, forKey: .files)
  }

  public var totalByteCount: Int64 {
    files.reduce(0) { $0 + $1.byteCount }
  }

  public func downloadURL(for file: ModelPackageFile) -> URL {
    var url = URL(string: "https://huggingface.co")!
    for component in (file.sourceRepository ?? repository).split(separator: "/") {
      url.appendPathComponent(String(component))
    }
    url.appendPathComponent("resolve")
    url.appendPathComponent(file.sourceRevision ?? revision)
    for component in (file.sourcePath ?? file.path).split(separator: "/") {
      url.appendPathComponent(String(component))
    }
    return url
  }
}

public enum ModelCatalog {
  /// Everything except the transformer is byte-identical across the standard
  /// and turbo packages; the installer reuses these files across installs.
  private static let sharedMinimaxH3Files: [ModelPackageFile] = [
    ModelPackageFile(
      role: .textEncoder,
      path: "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
      byteCount: 27_141_342_152,
      sha256: "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"
    ),
    // The int8 ConvRot decoder was measured about 2x slower to decode than
    // the fp16 one at 512 with a 22-frame clip on Apple silicon, so the
    // released decoder stays the shipped default. The engine still reads an
    // int8 decoder when a package provides one.
    ModelPackageFile(
      role: .videoVAE,
      path: "vae/minimax_h3_video_vae_fp16.safetensors",
      byteCount: 5_207_808_496,
      sha256: "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
    ),
    ModelPackageFile(
      role: .audioVAE,
      path: "vae/minimax_h3_audio_vae_fp32.safetensors",
      byteCount: 605_254_808,
      sha256: "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
    ),
    officialMetadata(
      path: "FL2VA/tokenizer/tokenizer.json",
      byteCount: 7_032_403,
      sha256: "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
    ),
    officialMetadata(
      path: "FL2VA/transformer/config.json",
      byteCount: 604,
      sha256: "f619093a231fcfbcc3d035bec26c50ad864e7331a500d5c519f5045dc1e50458"
    ),
    officialMetadata(
      path: "FL2VA/text_encoder/config.json",
      byteCount: 1_474,
      sha256: "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427"
    ),
    officialMetadata(
      path: "FL2VA/video_vae/config.json",
      byteCount: 1_807,
      sha256: "3edd2cdd1ebc823c868be55ef917e1b3b8a398fde4d3150dae44a3bf05d9f627"
    ),
    officialMetadata(
      path: "FL2VA/audio_vae/config.json",
      byteCount: 1_973,
      sha256: "d8f3bcc62e23c7e9806970fa63cca6139c06faa3797cf9c94034f60db8512771"
    ),
  ]

  public static let minimaxH3Int8 = ModelPackageManifest(
    id: "comfy-minimax-h3-int8-v1",
    displayName: "MiniMax H3 · INT8",
    detail: "Pruned FL2VA, INT8 ConvRot transformer and text encoder",
    repository: "Comfy-Org/MiniMax-H3",
    revision: "014cd40f7e177756c6b2473c0d93b1c89a790dd2",
    licenseName: "MiniMax H3 Community License Agreement",
    licenseURL: URL(
      string:
        "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/939557dc319dd91227e30195a763f272ba7f8765/LICENSE"
    )!,
    minimumUnifiedMemoryBytes: 32 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        byteCount: 20_970_379_616,
        sha256: "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
      )
    ] + sharedMinimaxH3Files
  )

  /// The lightx2v turbo distillation merged into the pruned INT8 transformer
  /// by `Scripts/convert-turbo-package.py`, hosted on Hugging Face. A local
  /// conversion output installs instantly when present; every shared file
  /// reuses the standard package's bytes.
  public static let minimaxH3TurboInt8 = ModelPackageManifest(
    id: "h3ddle-minimax-h3-turbo-int8-v1",
    displayName: "MiniMax H3 · Turbo (Experimental)",
    detail:
      "Step-distilled transformer: highest fidelity at 4–8 passes, loose "
      + "prompt control. Describe what to see, not what happens.",
    repository: "Comfy-Org/MiniMax-H3",
    revision: "014cd40f7e177756c6b2473c0d93b1c89a790dd2",
    licenseName: "MiniMax H3 Community License Agreement",
    licenseURL: URL(
      string:
        "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/939557dc319dd91227e30195a763f272ba7f8765/LICENSE"
    )!,
    minimumUnifiedMemoryBytes: 32 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    generationProfile: .turbo,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        byteCount: 20_970_379_854,
        sha256: "9ad5c98b533894c122050d32804a14f49fca8edc16c52564a281cdc5825ac934",
        sourceRepository: "PulpCut/MiniMax-H3-Turbo-INT8-ConvRot",
        sourceRevision: "4aea334367e4007d7b3630810ec28eb97639ae65",
        sourcePath: "minimax_h3_fl2va_pruned_turbo_int8_convrot.safetensors",
        localCandidatePath: URL.applicationSupportDirectory
          .appendingPathComponent("H3ddle", isDirectory: true)
          .appendingPathComponent("Conversion", isDirectory: true)
          .appendingPathComponent(
            "minimax_h3_fl2va_pruned_turbo_int8_convrot.safetensors",
            isDirectory: false
          )
          .path
      )
    ] + sharedMinimaxH3Files
  )

  private static func officialMetadata(
    path: String,
    byteCount: Int64,
    sha256: String
  ) -> ModelPackageFile {
    ModelPackageFile(
      role: .runtimeMetadata,
      path: path,
      byteCount: byteCount,
      sha256: sha256,
      sourceRepository: "MiniMaxAI/MiniMax-H3",
      sourceRevision: "939557dc319dd91227e30195a763f272ba7f8765"
    )
  }
}
