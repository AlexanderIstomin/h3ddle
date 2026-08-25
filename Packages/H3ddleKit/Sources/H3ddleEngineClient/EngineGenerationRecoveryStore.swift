import Foundation
import H3ddleGeneration

public struct EngineGenerationRecoveryRecord: Hashable, Codable, Sendable {
  public var request: GenerationRequest
  public var modelDirectory: URL
  public var displayPrompt: String
  public var settingsDescription: String
  public var createdAt: Date

  public init(
    request: GenerationRequest,
    modelDirectory: URL,
    displayPrompt: String,
    settingsDescription: String,
    createdAt: Date = Date()
  ) {
    self.request = request
    self.modelDirectory = modelDirectory
    self.displayPrompt = displayPrompt
    self.settingsDescription = settingsDescription
    self.createdAt = createdAt
  }
}

/// Small app-owned manifests around h3.c's binary sampler checkpoints. The
/// payload can be tens of megabytes for a long clip; this store never
/// decodes or rewrites it, and only keeps enough information to reissue the
/// exact request after the app or helper disappears.
public struct EngineGenerationRecoveryStore: Sendable {
  public static let manifestName = "job.json"
  public static let checkpointName = "denoiser.h3ckpt"

  public var rootURL: URL
  public var outputDirectory: URL

  public init(
    rootURL: URL = URL.applicationSupportDirectory
      .appendingPathComponent("H3ddle", isDirectory: true)
      .appendingPathComponent("GenerationRecovery", isDirectory: true),
    outputDirectory: URL = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleGenerated", isDirectory: true)
  ) {
    self.rootURL = rootURL
    self.outputDirectory = outputDirectory
  }

  public func makeContext(
    for kind: GenerationKind,
    jobID: UUID = UUID()
  ) -> GenerationRecoveryContext {
    let directory = rootURL.appendingPathComponent(
      jobID.uuidString, isDirectory: true)
    let output = outputDirectory
      .appendingPathComponent(jobID.uuidString)
      .appendingPathExtension(EngineGenerationProvider.outputExtension(for: kind))
    return GenerationRecoveryContext(
      jobID: jobID,
      directoryURL: directory,
      outputURL: output
    )
  }

  public func save(_ record: EngineGenerationRecoveryRecord) throws {
    guard let recovery = record.request.recovery else { return }
    try FileManager.default.createDirectory(
      at: recovery.directoryURL,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    try FileManager.default.createDirectory(
      at: outputDirectory,
      withIntermediateDirectories: true
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(record)
    let manifest = recovery.directoryURL.appendingPathComponent(Self.manifestName)
    try data.write(to: manifest, options: .atomic)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o600], ofItemAtPath: manifest.path)
  }

  public func latest() -> EngineGenerationRecoveryRecord? {
    guard let directories = try? FileManager.default.contentsOfDirectory(
      at: rootURL,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    ) else { return nil }
    let decoder = JSONDecoder()
    return directories.compactMap { directory in
      let manifest = directory.appendingPathComponent(Self.manifestName)
      guard let data = try? Data(contentsOf: manifest),
        let record = try? decoder.decode(
          EngineGenerationRecoveryRecord.self, from: data),
        record.request.recovery?.directoryURL.standardizedFileURL
          == directory.standardizedFileURL
      else { return nil }
      return record
    }
    .max { $0.createdAt < $1.createdAt }
  }

  public func discard(_ recovery: GenerationRecoveryContext) {
    try? FileManager.default.removeItem(at: recovery.directoryURL)
  }

  public func discardAll(except kept: GenerationRecoveryContext? = nil) {
    guard let directories = try? FileManager.default.contentsOfDirectory(
      at: rootURL,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    ) else { return }
    for directory in directories
    where directory.standardizedFileURL != kept?.directoryURL.standardizedFileURL {
      try? FileManager.default.removeItem(at: directory)
    }
  }
}
