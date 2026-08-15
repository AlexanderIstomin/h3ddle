import Foundation

/// Works out how a model folder should be driven when no catalog manifest
/// says so.
///
/// A managed package carries its generation profile in the manifest, but a
/// folder the user added by hand has none, and defaulting those to the
/// standard profile silently mis-drives distilled weights: they lose the
/// Beta(0.6, 0.6) sigma spacing they were trained against and the lower pass
/// default, which costs quality on every run without any visible sign.
///
/// Converted turbo checkpoints record how they were made in their
/// safetensors metadata, so the answer is in the file itself rather than in
/// its name. Only the header is read — a few kilobytes off the front of a
/// twenty-gigabyte file.
public enum ModelFolderInspection {
  /// Names the engine looks for, most specific first.
  static let transformerNames = [
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
  ]

  public static func generationProfile(at directory: URL) -> ModelGenerationProfile {
    for name in transformerNames {
      let url = directory.appendingPathComponent(name, isDirectory: false)
      guard let metadata = safetensorsMetadata(at: url) else { continue }
      if isDistilled(metadata) { return .turbo }
    }
    return .standard
  }

  /// A merge records the adapter it folded in; strength 0 means the file was
  /// produced by the pipeline's self-check and carries no distillation.
  static func isDistilled(_ metadata: [String: String]) -> Bool {
    guard let lora = metadata["lora"], lora != "none", !lora.isEmpty else {
      return false
    }
    if let strength = metadata["lora_strength"], Double(strength) == 0 {
      return false
    }
    return true
  }

  /// Reads a safetensors header without mapping the payload.
  static func safetensorsMetadata(at url: URL) -> [String: String]? {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
    defer { try? handle.close() }
    guard
      let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8
    else { return nil }
    let length = lengthData.withUnsafeBytes {
      $0.loadUnaligned(as: UInt64.self).littleEndian
    }
    // Headers are small; a huge value means this is not a safetensors file.
    guard length > 0, length <= 64 * 1024 * 1024 else { return nil }
    guard
      let headerData = try? handle.read(upToCount: Int(length)),
      headerData.count == Int(length),
      let json = try? JSONSerialization.jsonObject(with: headerData),
      let object = json as? [String: Any],
      let metadata = object["__metadata__"] as? [String: String]
    else { return nil }
    return metadata
  }
}
