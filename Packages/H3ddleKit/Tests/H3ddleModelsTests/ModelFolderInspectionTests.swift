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
