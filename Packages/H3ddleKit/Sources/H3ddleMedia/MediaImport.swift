import AVFoundation
import Foundation
import H3ddleCore
import ImageIO
import UniformTypeIdentifiers

public enum MediaImportLane: String, Sendable {
  case visual
  case audio

  public func accepts(_ kind: MediaKind) -> Bool {
    switch self {
    case .visual: kind.isVisual
    case .audio: kind == .audio
    }
  }
}

public enum MediaImportError: Error, Equatable, LocalizedError {
  case unsupportedType
  case unreadable
  case wrongLane

  public var errorDescription: String? {
    switch self {
    case .unsupportedType:
      "This file type cannot be added to the timeline."
    case .unreadable:
      "This file could not be read as media."
    case .wrongLane:
      "That file does not belong on this track."
    }
  }
}

public struct MediaImportProbe: Equatable, Sendable {
  public var kind: MediaKind
  public var duration: TimeInterval
  public var hasNativeAudio: Bool
  public var displayName: String

  public init(
    kind: MediaKind,
    duration: TimeInterval,
    hasNativeAudio: Bool,
    displayName: String
  ) {
    self.kind = kind
    self.duration = max(0, duration)
    self.hasNativeAudio = hasNativeAudio
    self.displayName = displayName
  }
}

public struct ImportedMedia: Equatable, Sendable {
  public var asset: AssetReference
  public var includesNativeAudio: Bool

  public init(asset: AssetReference, includesNativeAudio: Bool) {
    self.asset = asset
    self.includesNativeAudio = includesNativeAudio
  }
}

public enum MediaImport {
  public static let stillDuration: TimeInterval = 3

  public static var visualContentTypes: [UTType] {
    [.mpeg4Movie, .quickTimeMovie, .png, .jpeg, .heic, .webP, .tiff, .bmp]
      + Self.type("public.mpeg-4")
      + Self.type("com.apple.m4v-video")
  }

  public static var audioContentTypes: [UTType] {
    [.wav, .mp3, .mpeg4Audio, .aiff, .audio]
  }

  public static func partition(
    _ urls: [URL],
    onto lane: MediaImportLane
  ) -> (accepted: [URL], rejected: [URL]) {
    var accepted: [URL] = []
    var rejected: [URL] = []
    for url in urls {
      let file = url.standardizedFileURL
      guard let kind = kind(for: file), lane.accepts(kind) else {
        rejected.append(file)
        continue
      }
      accepted.append(file)
    }
    return (accepted, rejected)
  }

  public static func containsCompatible(_ urls: [URL], onto lane: MediaImportLane) -> Bool {
    !partition(urls, onto: lane).accepted.isEmpty
  }

  public static func kind(for url: URL) -> MediaKind? {
    let ext = url.pathExtension.lowercased()
    if imageExtensions.contains(ext) { return .image }
    if videoExtensions.contains(ext) { return .video }
    if audioExtensions.contains(ext) { return .audio }
    guard let type = UTType(filenameExtension: ext) else { return nil }
    if type.conforms(to: .image) { return .image }
    if type.conforms(to: .audiovisualContent), !type.conforms(to: .audio) { return .video }
    if type.conforms(to: .audio) { return .audio }
    return nil
  }

  public static func probe(_ url: URL) async throws -> MediaImportProbe {
    guard let kind = kind(for: url) else { throw MediaImportError.unsupportedType }
    let name = url.deletingPathExtension().lastPathComponent
    switch kind {
    case .image:
      try verifyImage(url)
      return MediaImportProbe(
        kind: .image,
        duration: stillDuration,
        hasNativeAudio: false,
        displayName: name
      )
    case .video, .audio:
      return try await probeAVAsset(url, kind: kind, displayName: name)
    }
  }

  public static func makeAsset(
    from source: URL,
    onto lane: MediaImportLane,
    copyingInto directory: URL
  ) async throws -> ImportedMedia {
    let probed = try await probe(source)
    guard lane.accepts(probed.kind) else { throw MediaImportError.wrongLane }
    let stored = try copy(source, into: directory)
    var asset = AssetReference(
      kind: probed.kind,
      displayName: probed.displayName,
      url: stored,
      duration: probed.duration
    )
    if let data = try? EmbeddedGenerationMetadata.read(from: stored),
      let value = try? JSONDecoder().decode(JSONValue.self, from: data),
      value.object != nil
    {
      asset.metadata[AssetMetadataKey.generationRecipe] = value
    }
    return ImportedMedia(
      asset: asset,
      includesNativeAudio: probed.kind == .video && probed.hasNativeAudio
    )
  }

  public static func copy(_ source: URL, into directory: URL) throws -> URL {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let ext = source.pathExtension
    let name = ext.isEmpty ? UUID().uuidString : "\(UUID().uuidString).\(ext)"
    let destination = directory.appendingPathComponent(name)
    if FileManager.default.fileExists(atPath: destination.path) {
      try FileManager.default.removeItem(at: destination)
    }
    try FileManager.default.copyItem(at: source, to: destination)
    return destination
  }

  private static let imageExtensions: Set<String> = [
    "png", "jpg", "jpeg", "heic", "heif", "webp", "tif", "tiff", "bmp",
  ]
  private static let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]
  private static let audioExtensions: Set<String> = [
    "wav", "m4a", "aac", "mp3", "aiff", "aif", "caf", "flac",
  ]

  private static func type(_ identifier: String) -> [UTType] {
    UTType(identifier).map { [$0] } ?? []
  }

  private static func verifyImage(_ url: URL) throws {
    guard FileManager.default.fileExists(atPath: url.path),
      let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      CGImageSourceGetCount(source) > 0
    else {
      throw MediaImportError.unreadable
    }
  }

  private static func probeAVAsset(
    _ url: URL,
    kind: MediaKind,
    displayName: String
  ) async throws -> MediaImportProbe {
    let asset = AVURLAsset(url: url)
    let duration = try await asset.load(.duration)
    let seconds = duration.seconds
    guard seconds.isFinite, seconds > 0 else { throw MediaImportError.unreadable }
    if kind == .video {
      let videoTracks = try await asset.loadTracks(withMediaType: .video)
      guard !videoTracks.isEmpty else { throw MediaImportError.unreadable }
    } else {
      let audioTracks = try await asset.loadTracks(withMediaType: .audio)
      guard !audioTracks.isEmpty else { throw MediaImportError.unreadable }
    }
    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
    return MediaImportProbe(
      kind: kind,
      duration: seconds,
      hasNativeAudio: !audioTracks.isEmpty,
      displayName: displayName
    )
  }
}
