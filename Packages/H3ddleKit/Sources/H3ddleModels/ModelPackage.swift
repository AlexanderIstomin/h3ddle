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

  public var id: String { path }

  public init(
    role: ModelPackageRole,
    path: String,
    byteCount: Int64,
    sha256: String,
    sourceRepository: String? = nil,
    sourceRevision: String? = nil
  ) {
    self.role = role
    self.path = path
    self.byteCount = byteCount
    self.sha256 = sha256
    self.sourceRepository = sourceRepository
    self.sourceRevision = sourceRevision
  }

  public var displayName: String {
    URL(fileURLWithPath: path).lastPathComponent
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
    self.files = files
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
    for component in file.path.split(separator: "/") {
      url.appendPathComponent(String(component))
    }
    return url
  }
}

public enum ModelCatalog {
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
      ),
      ModelPackageFile(
        role: .textEncoder,
        path: "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        byteCount: 27_141_342_152,
        sha256: "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"
      ),
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
