import Foundation

public enum ModelPackageRole: String, Codable, Equatable, Sendable {
  case transformer
  case textEncoder
  case videoVAE
  case imageVAE
  case audioVAE
  case referenceTransformer
  case previewDecoder
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

/// Which list a package belongs to.
///
/// Video and audio models are not alternatives — a project can want one of
/// each installed, and choosing one does not unchoose the other — so the
/// picker keeps a separate list per category rather than presenting one
/// list of interchangeable choices.
///
/// A package appears in exactly one list, under what it is chosen *for*.
/// H3 also writes the soundtrack that accompanies its video, but nobody
/// installs it to make a sound effect, so it belongs under video.
public enum ModelCapability: String, Codable, Sendable, CaseIterable {
  case video
  case audio
  /// A still from a model built for stills, as against the frame H3 keeps out
  /// of a very short clip. H3 remains under `.video` and still makes images.
  case image

  public var sectionTitle: String {
    switch self {
    case .video: "Video"
    case .audio: "Audio"
    case .image: "Image"
    }
  }
}

/// What an audio package was trained to make. Sound effects and music are one
/// Stable Audio transformer trained on different material and run down the
/// same engine path; speech is a different model entirely.
public enum ModelAudioRole: String, Codable, Sendable, CaseIterable {
  case soundEffects
  case music
  case speech
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
  /// Unified memory the engine needs resident, which is far less than the
  /// package weighs: weights are mapped from the file and paged in on
  /// demand, so a 72 GiB transformer streams through about 1.5 GiB. What
  /// has to fit is the working set — activations, the decoder, and the
  /// hot slice of weights — not the download.
  public let minimumUnifiedMemoryBytes: Int64
  public let compatibility: ModelEngineCompatibility
  public let generationProfile: ModelGenerationProfile
  public let capability: ModelCapability
  /// Set on audio packages only; nil for video ones.
  public let audioRole: ModelAudioRole?
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
    capability: ModelCapability = .video,
    audioRole: ModelAudioRole? = nil,
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
    self.capability = capability
    self.audioRole = audioRole
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
    capability =
      try container.decodeIfPresent(ModelCapability.self, forKey: .capability)
      ?? .video
    audioRole = try container.decodeIfPresent(ModelAudioRole.self, forKey: .audioRole)
    files = try container.decode([ModelPackageFile].self, forKey: .files)
  }

  /// Whether an install made from `other` is still the package this
  /// manifest describes.
  ///
  /// Comparing whole manifests meant that editing a description, or adding
  /// a field, threw away tens of gigabytes already on disk and demanded
  /// the whole download again. What actually decides usability is the
  /// revision and the files: same names, same sizes, same digests.
  public func describesSameFiles(as other: ModelPackageManifest) -> Bool {
    guard id == other.id, revision == other.revision else { return false }
    let mine = Set(files.map { [$0.path, String($0.byteCount), $0.sha256] })
    let theirs = Set(other.files.map { [$0.path, String($0.byteCount), $0.sha256] })
    return mine == theirs
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
    // Kijai's tiny 2D preview decoder (Apache-2.0). The engine prefers it for
    // live denoising previews: milliseconds per frame and megabytes resident
    // where the full decoder costs a long load and about ten gigabytes.
    ModelPackageFile(
      role: .previewDecoder,
      path: "vae_approx/taeh3.safetensors",
      byteCount: 9_791_388,
      sha256: "f0f60fa072089997f817402098c2fd90777cb2660dd79cf5df42fc1e3e08e527",
      sourceRepository: "Kijai/MiniMax-H3-TAE",
      sourceRevision: "a213ac8bf2f148b4f32372279a7f207846978900"
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
    minimumUnifiedMemoryBytes: 16 * 1_024 * 1_024 * 1_024,
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

  /// Adds the Ref2VA transformer alongside the standard FL2VA one, which is
  /// what ordered reference images need. Both ship together because the engine
  /// picks between them per generation, so one package covers prompt-only,
  /// keyframe, and reference work. Every file except the extra transformer is
  /// shared with the standard package and installs as a hardlink, so the real
  /// cost of adding this is the Ref2VA weights alone.
  public static let minimaxH3Ref2VAInt8 = ModelPackageManifest(
    id: "comfy-minimax-h3-int8-ref2va-v1",
    displayName: "MiniMax H3 · INT8 + References",
    detail: "Adds the Ref2VA transformer for ordered reference images",
    repository: "Comfy-Org/MiniMax-H3",
    revision: "014cd40f7e177756c6b2473c0d93b1c89a790dd2",
    licenseName: "MiniMax H3 Community License Agreement",
    licenseURL: URL(
      string:
        "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/939557dc319dd91227e30195a763f272ba7f8765/LICENSE"
    )!,
    minimumUnifiedMemoryBytes: 16 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        byteCount: 20_970_379_616,
        sha256: "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
      ),
      ModelPackageFile(
        role: .referenceTransformer,
        path: "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        byteCount: 20_970_379_616,
        sha256: "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
      )
    ] + sharedMinimaxH3Files
  )

  /// Both transformers step-distilled: the FL2VA turbo we host plus a Ref2VA
  /// turbo merged the same way, so reference generations get a fast path they
  /// have never had — every published turbo adapter targets FL2VA only.
  /// Measured 2.4x faster than base Ref2VA with reference identity intact.
  public static let minimaxH3Ref2VATurboInt8 = ModelPackageManifest(
    id: "h3ddle-minimax-h3-ref2va-turbo-int8-v1",
    displayName: "MiniMax H3 · Turbo + References (Experimental)",
    detail:
      "Step-distilled prompt and reference transformers: 8 passes, ordered "
      + "reference images, loose prompt control.",
    repository: "Comfy-Org/MiniMax-H3",
    revision: "014cd40f7e177756c6b2473c0d93b1c89a790dd2",
    licenseName: "MiniMax H3 Community License Agreement",
    licenseURL: URL(
      string:
        "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/939557dc319dd91227e30195a763f272ba7f8765/LICENSE"
    )!,
    minimumUnifiedMemoryBytes: 16 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    generationProfile: .turbo,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        byteCount: 20_970_380_012,
        sha256: "1dfe28c517a937fb9876f0975f224fd6e7ecb8744219f89bb8ba954403e10dc3",
        sourceRepository: "PulpCut/MiniMax-H3-Turbo-INT8-ConvRot",
        sourceRevision: "7a8e67cc51737428938fd9e39903a66e8ba58a18",
        sourcePath: "minimax_h3_fl2va_pruned_turbo_int8_convrot_input_major.safetensors",
        localCandidatePath: URL.applicationSupportDirectory
          .appendingPathComponent("H3ddle", isDirectory: true)
          .appendingPathComponent("Conversion", isDirectory: true)
          .appendingPathComponent(
            "minimax_h3_fl2va_pruned_turbo_int8_convrot_input_major.safetensors",
            isDirectory: false
          )
          .path
      ),
      ModelPackageFile(
        role: .referenceTransformer,
        path: "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        byteCount: 20_970_380_012,
        sha256: "5ca6696fe1cd9a8f254594ac67ee541f151b2377735dea3557364bd868270463",
        sourceRepository: "PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot",
        sourceRevision: "1d1391e63fb2c314f7a5a616f0aff08ad4e41b04",
        sourcePath: "minimax_h3_ref2va_pruned_turbo_int8_convrot_input_major.safetensors",
        localCandidatePath: URL.applicationSupportDirectory
          .appendingPathComponent("H3ddle", isDirectory: true)
          .appendingPathComponent("Conversion", isDirectory: true)
          .appendingPathComponent(
            "minimax_h3_ref2va_pruned_turbo_int8_convrot_input_major.safetensors",
            isDirectory: false
          )
          .path
      ),
    ] + sharedMinimaxH3Files
  )

  /// LTX-2.5, mirrored rather than repackaged: Lightricks publishes these four
  /// files in the identical Comfy INT8 ConvRot format h3.c already reads, so
  /// there was no conversion to do. Every SHA-256 below matches upstream.
  ///
  /// The mirror exists because the source repository is gated, and a gated
  /// repository cannot be fetched on a user's behalf — the download builds a
  /// plain URL with no token. The complete Community License Agreement travels
  /// with the weights in the mirror, which is what Section 3.2 requires.
  ///
  /// Deliberately a subset: 38.69 GB against upstream's ~180. The BF16 and
  /// NVFP4 transformers, the non-distilled `dev` checkpoint, the LoRA, the
  /// latent upscalers and the duration head are all absent, because the app
  /// runs the distilled checkpoint at eight steps and has no stage-2 ladder.
  public static let ltx25 = ModelPackageManifest(
    id: "h3ddle-ltx-2-5-int8-v1",
    displayName: "LTX-2.5 · Distilled",
    detail:
      "Creates 2–20 second videos with synchronized sound from a prompt, "
      + "start/end frames, or up to four reference images. Supports portrait, "
      + "square, and landscape output from 320p to 1080p; eight steps is the "
      + "recommended balance.",
    repository: "PulpCut/LTX-2.5-INT8-ConvRot-safetensors",
    revision: "d28e7aae3bfdb47184682838cd11989f1c8aa5dc",
    licenseName: "LTX-2.x Community License Agreement",
    licenseURL: URL(
      string:
        "https://huggingface.co/PulpCut/LTX-2.5-INT8-ConvRot-safetensors/blob/main/LICENSE"
    )!,
    // Measured, not estimated: a 2.7-second clip at 512² peaks at 5.23 GB of
    // physical footprint (5.05 GB RSS — the two agreeing is what says Metal's
    // unified buffers are counted). The package weighs 38.69 GB and never has
    // more than one 388 MB transformer block resident, because the tower and
    // the DiT cannot both fit and are loaded, run and freed in turn. 16 GB
    // leaves the app and the system room around that.
    minimumUnifiedMemoryBytes: 16 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    generationProfile: .turbo,
    capability: .video,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "diffusion_models/"
          + "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        byteCount: 21_504_034_224,
        sha256: "c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334"
      ),
      ModelPackageFile(
        role: .textEncoder,
        path: "text_encoders/"
          + "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        byteCount: 15_372_969_374,
        sha256: "6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f"
      ),
      ModelPackageFile(
        role: .videoVAE,
        path: "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        byteCount: 1_452_269_922,
        sha256: "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d"
      ),
      ModelPackageFile(
        role: .audioVAE,
        path: "vae/ltx-2.5-audio-vae-bf16.safetensors",
        byteCount: 364_866_540,
        sha256: "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5"
      ),
    ]
  )

  /// The lightx2v turbo distillation merged into the pruned INT8 transformer
  /// by `Scripts/convert-turbo-package.py`, hosted on Hugging Face. A local
  /// conversion output installs instantly when present; every shared file
  /// reuses the standard package's bytes.
  /// Z-Image-Turbo, repackaged: one file per subsystem and the int8 matrices
  /// stored input-major so the GPU reads them coalesced. No tensor was
  /// retrained, merged, pruned, or quantized in the repackaging.
  ///
  /// The autoencoder arrives in two pieces from two places. The decoder is
  /// the repackaged half, and every render runs it. The encoder only runs
  /// when a picture is being worked from, and it comes as the *whole*
  /// released autoencoder straight from Tongyi-MAI: that file carries both
  /// halves under `encoder.` and `decoder.` names, the engine reads the half
  /// it wants by name, and the decoder half in it is byte-for-byte the one
  /// packaged here — all 138 tensors. Taking it whole costs 99 MB of
  /// duplicated decoder against splitting it, and buys provenance directly
  /// from the release rather than another repackaged artifact to host and
  /// keep honest.
  public static let zImageTurbo = ModelPackageManifest(
    id: "h3ddle-z-image-turbo-int8-v1",
    displayName: "Z-Image · Turbo",
    detail:
      "Creates high-quality still images from a text prompt, or repaints a "
      + "source picture with adjustable strength. Supports portrait, square, "
      + "and landscape output from 512p to 1536p; eight passes is the "
      + "recommended balance.",
    repository: "PulpCut/Z-Image-Turbo-INT8-ConvRot-safetensors",
    revision: "07a468a19068386ab85b0f9d9e391ef57d78eb38",
    licenseName: "Apache License 2.0",
    licenseURL: URL(
      string:
        "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/LICENSE"
    )!,
    minimumUnifiedMemoryBytes: 16 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    capability: .image,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "transformer.safetensors",
        byteCount: 6_175_253_480,
        sha256: "c7b59609b92545842826ef0302316350772152e8e8491a93c226b9e3511cfa13",
        sourceRepository: "PulpCut/Z-Image-Turbo-INT8-ConvRot-safetensors",
        sourceRevision: "07a468a19068386ab85b0f9d9e391ef57d78eb38",
        sourcePath: "transformer.safetensors"
      ),
      ModelPackageFile(
        role: .textEncoder,
        path: "text_encoder.safetensors",
        byteCount: 8_044_982_208,
        sha256: "3182dbe94a375be115bd4f9a1c6d82c7bad8e441e8c27e890cbb053d10ad4f88",
        sourceRepository: "PulpCut/Z-Image-Turbo-INT8-ConvRot-safetensors",
        sourceRevision: "07a468a19068386ab85b0f9d9e391ef57d78eb38",
        sourcePath: "text_encoder.safetensors"
      ),
      ModelPackageFile(
        role: .imageVAE,
        path: "vae_decoder.safetensors",
        byteCount: 99_106_470,
        sha256: "b17536564790de05150409e5fb10c90e47459b2502153d38c6465784b016e4c4",
        sourceRepository: "PulpCut/Z-Image-Turbo-INT8-ConvRot-safetensors",
        sourceRevision: "07a468a19068386ab85b0f9d9e391ef57d78eb38",
        sourcePath: "vae_decoder.safetensors"
      ),
      ModelPackageFile(
        role: .imageVAE,
        path: "vae_encoder.safetensors",
        byteCount: 167_666_902,
        sha256: "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
        sourceRepository: "Tongyi-MAI/Z-Image-Turbo",
        sourceRevision: "f332072aa78be7aecdf3ee76d5c247082da564a6",
        sourcePath: "vae/diffusion_pytorch_model.safetensors"
      ),
      // TAEF1 is trained against the same FLUX.1 latent API Z-Image uses. It
      // makes a presentable per-pass image without repeatedly opening the
      // 99 MB full VAE or allocating its multi-gigabyte working set.
      ModelPackageFile(
        role: .previewDecoder,
        path: "vae_approx/taef1.safetensors",
        byteCount: 9_848_636,
        sha256: "47a6c2bff850da04b267cab70fe3553fef57255eb9a8e76852baa0a87850e54d",
        sourceRepository: "madebyollin/taef1",
        sourceRevision: "b1b2d00e9e440cfbf3dedb34266864da86016ceb",
        sourcePath: "diffusion_pytorch_model.safetensors"
      ),
      ModelPackageFile(
        role: .runtimeMetadata,
        path: "tokenizer.json",
        byteCount: 11_422_654,
        sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        sourceRepository: "PulpCut/Z-Image-Turbo-INT8-ConvRot-safetensors",
        sourceRevision: "07a468a19068386ab85b0f9d9e391ef57d78eb38",
        sourcePath: "tokenizer.json"
      ),
    ]
  )

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
    minimumUnifiedMemoryBytes: 16 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    generationProfile: .turbo,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        byteCount: 20_970_380_012,
        sha256: "1dfe28c517a937fb9876f0975f224fd6e7ecb8744219f89bb8ba954403e10dc3",
        sourceRepository: "PulpCut/MiniMax-H3-Turbo-INT8-ConvRot",
        sourceRevision: "7a8e67cc51737428938fd9e39903a66e8ba58a18",
        sourcePath: "minimax_h3_fl2va_pruned_turbo_int8_convrot_input_major.safetensors",
        localCandidatePath: URL.applicationSupportDirectory
          .appendingPathComponent("H3ddle", isDirectory: true)
          .appendingPathComponent("Conversion", isDirectory: true)
          .appendingPathComponent(
            "minimax_h3_fl2va_pruned_turbo_int8_convrot_input_major.safetensors",
            isDirectory: false
          )
          .path
      ),
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

  /// Sound effects and ambience, which H3 will not produce: its audio branch
  /// is trained on dialogue and answers a request for rain with speech.
  ///
  /// Unlike the H3 packages this one is small enough to install on any
  /// machine that runs the app at all, and it needs no Hugging Face account.
  public static let stableAudio3SmallSFX = ModelPackageManifest(
    id: "h3ddle-stable-audio-3-small-sfx-v1",
    displayName: "Stable Audio 3 Small · Sound Effects",
    detail:
      "Text to sound effects and ambience at 44.1 kHz stereo, eight passes, "
      + "faster than real time.",
    repository: "PulpCut/Stable-Audio-3-Small-SFX-safetensors",
    revision: "17914096d9e51e3486dbf97ee080d9c8f5512fd3",
    licenseName: "Stability AI Community License",
    licenseURL: URL(
      string:
        "https://huggingface.co/PulpCut/Stable-Audio-3-Small-SFX-safetensors/blob/17914096d9e51e3486dbf97ee080d9c8f5512fd3/LICENSE.md"
    )!,
    minimumUnifiedMemoryBytes: 8 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    capability: .audio,
    audioRole: .soundEffects,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "dit.safetensors",
        byteCount: 919_105_120,
        sha256: "3a7e7094db258990a8eccc3b1e6689b1368b557c4fb7bf14e61060d13efb8dbc",
        sourceRepository: "PulpCut/Stable-Audio-3-Small-SFX-safetensors",
        sourceRevision: "17914096d9e51e3486dbf97ee080d9c8f5512fd3",
        sourcePath: "dit.safetensors"
      ),
      ModelPackageFile(
        role: .textEncoder,
        path: "text_encoder.safetensors",
        byteCount: 563_175_776,
        sha256: "3f97a58e7674a4ff0063629c463fa7af343b1e792f28b0f302fc54ce1fc8cce4",
        sourceRepository: "PulpCut/Stable-Audio-3-Small-SFX-safetensors",
        sourceRevision: "17914096d9e51e3486dbf97ee080d9c8f5512fd3",
        sourcePath: "text_encoder.safetensors"
      ),
      ModelPackageFile(
        role: .audioVAE,
        path: "decoder.safetensors",
        byteCount: 218_069_724,
        sha256: "bbe71a56368240fa89e82c13c402c5097c9f2165390fb64fec756ada37e57249",
        sourceRepository: "PulpCut/Stable-Audio-3-Small-SFX-safetensors",
        sourceRevision: "17914096d9e51e3486dbf97ee080d9c8f5512fd3",
        sourcePath: "decoder.safetensors"
      ),
      ModelPackageFile(
        role: .runtimeMetadata,
        path: "tokenizer.json",
        byteCount: 34_362_429,
        sha256: "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd",
        sourceRepository: "PulpCut/Stable-Audio-3-Small-SFX-safetensors",
        sourceRevision: "17914096d9e51e3486dbf97ee080d9c8f5512fd3",
        sourcePath: "tokenizer.json"
      ),
    ]
  )

  /// Text spoken in a voice cloned from a reference clip, which neither H3
  /// nor Stable Audio will do: H3's audio branch improvises dialogue rather
  /// than reciting given words, and Stable Audio makes no speech at all.
  ///
  /// Converted from the Qwen3-TTS release by
  /// Scripts/convert-qwen3-tts-package.py and hosted as one file per
  /// subsystem, pinned to its upload commit and verified against the same
  /// SHA-256 the conversion produced. A local conversion output installs
  /// instantly when present.
  public static let qwen3TTSSpeech = ModelPackageManifest(
    id: "h3ddle-qwen3-tts-12hz-0.6b-base-v1",
    displayName: "Qwen3-TTS 12Hz 0.6B Base · Speech",
    detail:
      "Text to speech at 24 kHz, in a voice cloned from a few seconds of a "
      + "reference clip. Ten languages, faster than real time.",
    repository: "PulpCut/Qwen3-TTS-12Hz-0.6B-Base-safetensors",
    revision: "14a92e3816b3edc4cd0fb7f4d922156c58044ce0",
    licenseName: "Apache 2.0",
    licenseURL: URL(
      string:
        "https://huggingface.co/PulpCut/Qwen3-TTS-12Hz-0.6B-Base-safetensors/blob/14a92e3816b3edc4cd0fb7f4d922156c58044ce0/LICENSE"
    )!,
    minimumUnifiedMemoryBytes: 8 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    capability: .audio,
    audioRole: .speech,
    files: [
      speechFile(
        role: .transformer,
        path: "talker.safetensors",
        byteCount: 1_528_475_896,
        sha256: "dd45f15564979beac76583e4d06d842b7304daee0f636af746abd67dba73ff3e"
      ),
      speechFile(
        role: .transformer,
        path: "code_predictor.safetensors",
        byteCount: 283_152_152,
        sha256: "c818a0c46c458dfd106280258cb5e62b81fe7a7926732d98c564150f10da4448"
      ),
      speechFile(
        role: .textEncoder,
        path: "speaker_encoder.safetensors",
        byteCount: 35_425_464,
        sha256: "0ba0dcbb490e86606b3c2ed32e5d9d1b1a586c95ad8c5eefff23823f4e97bb11"
      ),
      speechFile(
        role: .audioVAE,
        path: "codec_decoder.safetensors",
        byteCount: 457_190_116,
        sha256: "234d1f2ec4583f48ae95e6b7d1d0d5003148034776310b99021c509ac0279983"
      ),
      speechFile(
        role: .runtimeMetadata,
        path: "tokenizer.json",
        byteCount: 4_757_709,
        sha256: "9aa75ab12fe8b491d89413103024f02ea0e91b77d337985a4718381f29feb0f9"
      ),
    ]
  )

  /// Every speech file sits at the repository root under the name it takes in
  /// the package, so only the conversion directory needs writing out — and
  /// that once rather than five times.
  private static func speechFile(
    role: ModelPackageRole,
    path: String,
    byteCount: Int64,
    sha256: String
  ) -> ModelPackageFile {
    ModelPackageFile(
      role: role,
      path: path,
      byteCount: byteCount,
      sha256: sha256,
      localCandidatePath: URL.applicationSupportDirectory
        .appendingPathComponent("H3ddle", isDirectory: true)
        .appendingPathComponent("Conversion", isDirectory: true)
        .appendingPathComponent("Qwen3-TTS-12Hz-0.6B-Base", isDirectory: true)
        .appendingPathComponent(path, isDirectory: false)
        .path
    )
  }

  public static let stableAudio3SmallMusic = ModelPackageManifest(
    id: "h3ddle-stable-audio-3-small-music-v1",
    displayName: "Stable Audio 3 Small · Music",
    detail:
      "Text to instrumental music and ambient beds at 44.1 kHz stereo, "
      + "eight passes, faster than real time.",
    repository: "PulpCut/Stable-Audio-3-Small-Music-safetensors",
    revision: "59e92686c56f6411f9aa9f09ece25041b4962d46",
    licenseName: "Stability AI Community License",
    licenseURL: URL(
      string:
        "https://huggingface.co/PulpCut/Stable-Audio-3-Small-Music-safetensors/blob/59e92686c56f6411f9aa9f09ece25041b4962d46/LICENSE.md"
    )!,
    minimumUnifiedMemoryBytes: 8 * 1_024 * 1_024 * 1_024,
    compatibility: .ready,
    capability: .audio,
    audioRole: .music,
    files: [
      ModelPackageFile(
        role: .transformer,
        path: "dit.safetensors",
        byteCount: 919_105_120,
        sha256: "f007992271b571a2ff9cbd8b054b8fab381d66e7a4ba6004adff170692102679",
        sourceRepository: "PulpCut/Stable-Audio-3-Small-Music-safetensors",
        sourceRevision: "59e92686c56f6411f9aa9f09ece25041b4962d46",
        sourcePath: "dit.safetensors"
      ),
      ModelPackageFile(
        role: .textEncoder,
        path: "text_encoder.safetensors",
        byteCount: 563_175_776,
        sha256: "3f97a58e7674a4ff0063629c463fa7af343b1e792f28b0f302fc54ce1fc8cce4",
        sourceRepository: "PulpCut/Stable-Audio-3-Small-Music-safetensors",
        sourceRevision: "59e92686c56f6411f9aa9f09ece25041b4962d46",
        sourcePath: "text_encoder.safetensors"
      ),
      ModelPackageFile(
        role: .audioVAE,
        path: "decoder.safetensors",
        byteCount: 218_069_724,
        sha256: "bbe71a56368240fa89e82c13c402c5097c9f2165390fb64fec756ada37e57249",
        sourceRepository: "PulpCut/Stable-Audio-3-Small-Music-safetensors",
        sourceRevision: "59e92686c56f6411f9aa9f09ece25041b4962d46",
        sourcePath: "decoder.safetensors"
      ),
      ModelPackageFile(
        role: .runtimeMetadata,
        path: "tokenizer.json",
        byteCount: 34_362_429,
        sha256: "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd",
        sourceRepository: "PulpCut/Stable-Audio-3-Small-Music-safetensors",
        sourceRevision: "59e92686c56f6411f9aa9f09ece25041b4962d46",
        sourcePath: "tokenizer.json"
      ),
    ]
  )
}
