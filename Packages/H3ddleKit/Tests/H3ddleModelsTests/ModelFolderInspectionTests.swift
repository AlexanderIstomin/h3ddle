import Foundation
import Testing

@testable import H3ddleModels

@Suite("Model folder inspection")
struct ModelFolderInspectionTests {
  /// Writes a file shaped like safetensors: an 8-byte little-endian header
  /// length, then that many bytes of JSON.
  private func writeSafetensors(
    metadata: [String: String]?, to url: URL
  ) throws {
    var object: [String: Any] = ["fake.weight": ["dtype": "I8", "shape": [1]]]
    if let metadata { object["__metadata__"] = metadata }
    let header = try JSONSerialization.data(withJSONObject: object)
    var length = UInt64(header.count).littleEndian
    var data = Data(bytes: &length, count: 8)
    data.append(header)
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url)
  }

  private func folder() throws -> URL {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-inspect-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  @Test("A merged turbo checkpoint is recognised from its own metadata")
  func detectsTurbo() throws {
    let directory = try folder()
    defer { try? FileManager.default.removeItem(at: directory) }
    try writeSafetensors(
      metadata: [
        "conversion": "h3ddle convert-turbo-package.py",
        "lora": "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        "lora_strength": "1.0",
      ],
      to: directory.appendingPathComponent(
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"))
    #expect(ModelFolderInspection.generationProfile(at: directory) == .turbo)
  }

  @Test("A stock checkpoint carries no merge and stays standard")
  func detectsStandard() throws {
    let directory = try folder()
    defer { try? FileManager.default.removeItem(at: directory) }
    try writeSafetensors(
      metadata: nil,
      to: directory.appendingPathComponent(
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"))
    #expect(ModelFolderInspection.generationProfile(at: directory) == .standard)
  }

  @Test("A reference-only folder is inspected too")
  func detectsTurboOnReferenceTransformer() throws {
    let directory = try folder()
    defer { try? FileManager.default.removeItem(at: directory) }
    try writeSafetensors(
      metadata: ["lora": "turbo.safetensors", "lora_strength": "1.0"],
      to: directory.appendingPathComponent(
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"))
    #expect(ModelFolderInspection.generationProfile(at: directory) == .turbo)
  }

  @Test("A strength-zero self-check output is not distilled")
  func selfCheckOutputIsNotTurbo() {
    #expect(!ModelFolderInspection.isDistilled(
      ["lora": "turbo.safetensors", "lora_strength": "0"]))
    #expect(!ModelFolderInspection.isDistilled(["lora": "none"]))
    #expect(ModelFolderInspection.isDistilled(
      ["lora": "turbo.safetensors", "lora_strength": "1.0"]))
  }

  @Test("A folder with no weights, or unreadable ones, is standard")
  func missingOrJunk() throws {
    let directory = try folder()
    defer { try? FileManager.default.removeItem(at: directory) }
    #expect(ModelFolderInspection.generationProfile(at: directory) == .standard)
    let junk = directory.appendingPathComponent(
      "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    try FileManager.default.createDirectory(
      at: junk.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("not a model".utf8).write(to: junk)
    #expect(ModelFolderInspection.generationProfile(at: directory) == .standard)
  }
}

@Suite("Model folder category matching")
struct ModelFolderCategoryTests {
  private func folder() throws -> URL {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-category-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  private func touch(_ relative: String, in directory: URL) throws {
    let url = directory.appendingPathComponent(relative)
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data([0]).write(to: url)
  }

  @Test("A sound-effect package needs every file the loader opens")
  func soundEffectRequiresAllFiles() throws {
    let directory = try folder()
    defer { try? FileManager.default.removeItem(at: directory) }
    for name in ModelFolderInspection.soundEffectNames.dropLast() {
      try touch(name, in: directory)
    }
    // One missing file is still the wrong answer: the engine would fail on it.
    #expect(!ModelFolderInspection.matches(.audio, at: directory))
    try touch(ModelFolderInspection.soundEffectNames.last!, in: directory)
    #expect(ModelFolderInspection.matches(.audio, at: directory))
  }

  @Test("Either H3 layout counts as a video model")
  func videoAcceptsBothLayouts() throws {
    let released = try folder()
    let optimized = try folder()
    defer {
      try? FileManager.default.removeItem(at: released)
      try? FileManager.default.removeItem(at: optimized)
    }
    try touch("FL2VA/transformer/config.json", in: released)
    try touch(
      "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
      in: optimized)
    #expect(ModelFolderInspection.matches(.video, at: released))
    #expect(ModelFolderInspection.matches(.video, at: optimized))
  }

  @Test("A model filed under the wrong heading is rejected")
  func categoriesDoNotCrossOver() throws {
    let video = try folder()
    let audio = try folder()
    defer {
      try? FileManager.default.removeItem(at: video)
      try? FileManager.default.removeItem(at: audio)
    }
    try touch("FL2VA/transformer/config.json", in: video)
    for name in ModelFolderInspection.soundEffectNames {
      try touch(name, in: audio)
    }
    #expect(!ModelFolderInspection.matches(.audio, at: video))
    #expect(!ModelFolderInspection.matches(.video, at: audio))
  }

  @Test("An unrelated folder belongs to neither")
  func emptyFolderMatchesNothing() throws {
    let directory = try folder()
    defer { try? FileManager.default.removeItem(at: directory) }
    try touch("notes.txt", in: directory)
    #expect(!ModelFolderInspection.matches(.video, at: directory))
    #expect(!ModelFolderInspection.matches(.audio, at: directory))
  }
}
