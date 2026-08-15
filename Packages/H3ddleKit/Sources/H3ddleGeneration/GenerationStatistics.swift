import Foundation

/// What one finished generation cost and used.
///
/// Kept per result so a clip can still describe itself after later runs have
/// moved on, and rendered as prose rather than a settings dump: the point is
/// a sentence someone can paste into a post and have others understand what
/// the machine actually did.
public struct GenerationStatistics: Equatable, Sendable {
  public var kind: GenerationKind
  public var seconds: TimeInterval
  public var canvasWidth: Int?
  public var canvasHeight: Int?
  public var denoisingSteps: Int
  public var clipSeconds: TimeInterval
  public var modelName: String
  public var blockCache: Bool
  public var deviceName: String?
  public var deviceMemoryBytes: UInt64?

  public init(
    kind: GenerationKind,
    seconds: TimeInterval,
    canvasWidth: Int? = nil,
    canvasHeight: Int? = nil,
    denoisingSteps: Int,
    clipSeconds: TimeInterval,
    modelName: String,
    blockCache: Bool = false,
    deviceName: String? = nil,
    deviceMemoryBytes: UInt64? = nil
  ) {
    self.kind = kind
    self.seconds = seconds
    self.canvasWidth = canvasWidth
    self.canvasHeight = canvasHeight
    self.denoisingSteps = denoisingSteps
    self.clipSeconds = clipSeconds
    self.modelName = modelName
    self.blockCache = blockCache
    self.deviceName = deviceName
    self.deviceMemoryBytes = deviceMemoryBytes
  }

  /// "6 min 1 s", "48 s" — coarse on purpose, because a post does not want
  /// tenths of a second.
  public var elapsedPhrase: String {
    let total = Int(seconds.rounded())
    if total < 60 { return "\(total) s" }
    let minutes = total / 60
    let remainder = total % 60
    return remainder == 0 ? "\(minutes) min" : "\(minutes) min \(remainder) s"
  }

  var subject: String {
    switch kind {
    case .image: "an image"
    case .video: String(format: "a %.1fs video with sound", clipSeconds)
    case .audio: String(format: "%.1fs of audio", clipSeconds)
    }
  }

  var canvasPhrase: String? {
    guard let canvasWidth, let canvasHeight else { return nil }
    return "\(canvasWidth)x\(canvasHeight)"
  }

  var devicePhrase: String {
    guard let deviceName else { return "Apple silicon" }
    guard let deviceMemoryBytes else { return deviceName }
    let gigabytes = Int(
      (Double(deviceMemoryBytes) / 1_073_741_824).rounded())
    return "\(deviceName) with \(gigabytes) GB"
  }

  public var socialSummary: String {
    var sentence = "Generated \(subject)"
    if let canvasPhrase { sentence += " at \(canvasPhrase)" }
    sentence += " in \(elapsedPhrase) on \(devicePhrase), fully offline."

    var settings = ["\(denoisingSteps) denoising passes", "model \(modelName)"]
    if blockCache { settings.append("block cache on") }
    sentence += " Settings: " + settings.joined(separator: ", ") + "."
    sentence +=
      " Made with H3ddle, an open-source local MiniMax H3 app for macOS:"
      + " https://github.com/AlexanderIstomin/h3ddle"
    return sentence
  }
}
