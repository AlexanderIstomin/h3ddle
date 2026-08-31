import Foundation
import H3ddleCore

public enum GenerationRecipeInputRole: String, Codable, Sendable {
  case firstFrame
  case lastFrame
  case referenceImage
  case inpaintingSource
  case inpaintingMask
  case speechReference
  case voiceEmbedding
}

public struct GenerationRecipeInput: Hashable, Codable, Sendable {
  public var id: UUID
  public var role: GenerationRecipeInputRole
  public var index: Int
  public var assetID: AssetID?
  public var fileName: String

  public init(
    id: UUID = UUID(),
    role: GenerationRecipeInputRole,
    index: Int = 0,
    assetID: AssetID? = nil,
    fileName: String
  ) {
    self.id = id
    self.role = role
    self.index = index
    self.assetID = assetID
    self.fileName = fileName
  }
}

public enum GenerationRecipeError: LocalizedError, Equatable, Sendable {
  case invalidMetadata
  case missingInput(String)

  public var errorDescription: String? {
    switch self {
    case .invalidMetadata:
      "This asset's generation metadata is invalid."
    case .missingInput(let name):
      "A generation input is unavailable: \(name)"
    }
  }
}

/// Immutable provenance for one generated asset. Input URLs are replaced by
/// opaque recipe URLs before persistence, so neither project JSON nor a file
/// shared outside H3ddle discloses absolute paths.
public struct GenerationRecipe: Hashable, Codable, Sendable {
  public static let currentVersion = 1

  public var version: Int
  public var id: UUID
  public var createdAt: Date
  public var parentAssetID: AssetID?
  public var modelID: String?
  public var displayPrompt: String
  public var settingsDescription: String
  public var studioSettings: GenerationStudioSettings
  public var aspectRatio: String
  public var soundscape: String
  public var music: String
  public var request: GenerationRequest
  public var inputs: [GenerationRecipeInput]

  public init(
    version: Int = currentVersion,
    id: UUID = UUID(),
    createdAt: Date = Date(),
    parentAssetID: AssetID? = nil,
    modelID: String? = nil,
    displayPrompt: String,
    settingsDescription: String,
    studioSettings: GenerationStudioSettings,
    aspectRatio: String,
    soundscape: String = "",
    music: String = "",
    request: GenerationRequest,
    inputs: [GenerationRecipeInput] = []
  ) {
    self.version = version
    self.id = id
    self.createdAt = createdAt
    self.parentAssetID = parentAssetID
    self.modelID = modelID
    self.displayPrompt = displayPrompt
    self.settingsDescription = settingsDescription
    self.studioSettings = studioSettings
    self.aspectRatio = aspectRatio
    self.soundscape = soundscape
    self.music = music
    self.request = request
    self.inputs = inputs
  }

  public init(
    job: GenerationQueueJob,
    projectAssets: [AssetReference],
    parentAssetID: AssetID? = nil
  ) {
    var portableRequest = job.request
    portableRequest.recovery = nil
    var captured: [GenerationRecipeInput] = []

    func capture(
      _ url: URL?,
      role: GenerationRecipeInputRole,
      index: Int = 0
    ) -> URL? {
      guard let url else { return nil }
      let assetID = projectAssets.first {
        $0.url.standardizedFileURL == url.standardizedFileURL
      }?.id
      let input = GenerationRecipeInput(
        role: role,
        index: index,
        assetID: assetID,
        fileName: url.lastPathComponent
      )
      captured.append(input)
      return Self.portableURL(for: input.id)
    }

    portableRequest.firstFrameURL = capture(job.request.firstFrameURL, role: .firstFrame)
    portableRequest.lastFrameURL = capture(job.request.lastFrameURL, role: .lastFrame)
    portableRequest.referenceImageURLs = job.request.referenceImageURLs.enumerated().compactMap {
      capture($0.element, role: .referenceImage, index: $0.offset)
    }
    if var inpainting = job.request.videoInpainting {
      inpainting.sourceVideoURL = capture(
        inpainting.sourceVideoURL,
        role: .inpaintingSource
      )!
      inpainting.maskURL = capture(inpainting.maskURL, role: .inpaintingMask)!
      portableRequest.videoInpainting = inpainting
    }
    if var speech = job.request.speech {
      speech.referenceAudioURL = capture(speech.referenceAudioURL, role: .speechReference)
      speech.voiceEmbeddingURL = capture(speech.voiceEmbeddingURL, role: .voiceEmbedding)
      portableRequest.speech = speech
    }

    self.init(
      id: job.id,
      createdAt: job.finishedAt ?? job.createdAt,
      parentAssetID: parentAssetID,
      modelID: job.modelID,
      displayPrompt: job.displayPrompt,
      settingsDescription: job.settingsDescription,
      studioSettings: job.studioSettings,
      aspectRatio: job.aspectRatio,
      soundscape: job.soundscape,
      music: job.music,
      request: portableRequest,
      inputs: captured
    )
  }

  public func resolvedRequest(
    projectAssets: [AssetReference],
    fallbackRequest: GenerationRequest? = nil
  ) throws -> GenerationRequest {
    var resolved = request
    resolved.recovery = nil

    func resolve(_ url: URL?) throws -> URL? {
      guard let url else { return nil }
      guard url.scheme == Self.inputScheme,
        let host = url.host,
        let id = UUID(uuidString: host),
        let input = inputs.first(where: { $0.id == id })
      else {
        // Accept older recipes that still point at a concrete file only when
        // it remains available; a missing path is never silently ignored.
        guard url.isFileURL, FileManager.default.fileExists(atPath: url.path) else {
          throw GenerationRecipeError.missingInput(url.lastPathComponent)
        }
        return url
      }
      if let assetID = input.assetID,
        let asset = projectAssets.first(where: { $0.id == assetID }),
        FileManager.default.fileExists(atPath: asset.url.path)
      {
        return asset.url
      }
      if let fallbackRequest,
        let fallback = Self.inputURL(for: input, in: fallbackRequest),
        fallback.lastPathComponent == input.fileName,
        FileManager.default.fileExists(atPath: fallback.path)
      {
        return fallback
      }
      throw GenerationRecipeError.missingInput(input.fileName)
    }

    resolved.firstFrameURL = try resolve(request.firstFrameURL)
    resolved.lastFrameURL = try resolve(request.lastFrameURL)
    resolved.referenceImageURLs = try request.referenceImageURLs.compactMap(resolve)
    if var inpainting = request.videoInpainting {
      guard let source = try resolve(inpainting.sourceVideoURL),
        let mask = try resolve(inpainting.maskURL)
      else {
        throw GenerationRecipeError.invalidMetadata
      }
      inpainting.sourceVideoURL = source
      inpainting.maskURL = mask
      resolved.videoInpainting = inpainting
    }
    if var speech = request.speech {
      speech.referenceAudioURL = try resolve(speech.referenceAudioURL)
      speech.voiceEmbeddingURL = try resolve(speech.voiceEmbeddingURL)
      resolved.speech = speech
    }
    return resolved
  }

  /// Reconnects a path-free recipe to project-owned input assets after an
  /// older queue job supplied the concrete URLs needed for migration.
  public func rebindingInputs(
    to resolvedRequest: GenerationRequest,
    projectAssets: [AssetReference]
  ) -> GenerationRecipe {
    var rebound = self
    for index in rebound.inputs.indices {
      guard let url = Self.inputURL(for: rebound.inputs[index], in: resolvedRequest),
        let asset = projectAssets.first(where: {
          $0.url.standardizedFileURL == url.standardizedFileURL
        })
      else { continue }
      rebound.inputs[index].assetID = asset.id
    }
    return rebound
  }

  public func attaching(to asset: AssetReference) throws -> AssetReference {
    var asset = asset
    let data = try encodedJSON()
    asset.metadata[AssetMetadataKey.generationRecipe] = try JSONDecoder().decode(
      JSONValue.self,
      from: data
    )
    return asset
  }

  public func encodedJSON() throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return try encoder.encode(self)
  }

  public static func recipe(from asset: AssetReference) throws -> GenerationRecipe? {
    guard let value = asset.metadata[AssetMetadataKey.generationRecipe] else { return nil }
    let data = try JSONEncoder().encode(value)
    let recipe = try JSONDecoder().decode(GenerationRecipe.self, from: data)
    guard recipe.version <= currentVersion else {
      throw GenerationRecipeError.invalidMetadata
    }
    return recipe
  }

  private static let inputScheme = "h3ddle-input"

  private static func portableURL(for id: UUID) -> URL {
    URL(string: "\(inputScheme)://\(id.uuidString.lowercased())")!
  }

  private static func inputURL(
    for input: GenerationRecipeInput,
    in request: GenerationRequest
  ) -> URL? {
    switch input.role {
    case .firstFrame:
      request.firstFrameURL
    case .lastFrame:
      request.lastFrameURL
    case .referenceImage:
      request.referenceImageURLs.indices.contains(input.index)
        ? request.referenceImageURLs[input.index] : nil
    case .inpaintingSource:
      request.videoInpainting?.sourceVideoURL
    case .inpaintingMask:
      request.videoInpainting?.maskURL
    case .speechReference:
      request.speech?.referenceAudioURL
    case .voiceEmbedding:
      request.speech?.voiceEmbeddingURL
    }
  }
}

public enum GenerationReplacementLane: String, Codable, Sendable {
  case visual
}

public struct GenerationReplacementTarget: Hashable, Codable, Sendable {
  public var projectID: UUID
  public var lane: GenerationReplacementLane
  public var clipID: UUID
  public var expectedAssetID: AssetID

  public init(
    projectID: UUID,
    lane: GenerationReplacementLane = .visual,
    clipID: UUID,
    expectedAssetID: AssetID
  ) {
    self.projectID = projectID
    self.lane = lane
    self.clipID = clipID
    self.expectedAssetID = expectedAssetID
  }
}
