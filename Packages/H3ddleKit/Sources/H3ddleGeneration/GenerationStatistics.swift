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
  public var denoisingSteps: Int?
  public var stepLabel: String
  public var clipSeconds: TimeInterval
  public var modelName: String
  public var aspectRatio: String?
  public var transformerBlocks: Int?
  public var coreReuse: Int?
  public var blockCache: Bool?
  public var stillFrameCount: Int?
  public var previewDenoise: Bool?
  public var seed: UInt64?
  public var conditioning: String?
  public var speechLanguage: String?
  public var speechVariation: Double?
  public var voiceName: String?
  public var deviceName: String?
  public var deviceMemoryBytes: UInt64?

  public init(
    kind: GenerationKind,
    seconds: TimeInterval,
    canvasWidth: Int? = nil,
    canvasHeight: Int? = nil,
    denoisingSteps: Int? = nil,
    stepLabel: String = "passes",
    clipSeconds: TimeInterval,
    modelName: String,
    aspectRatio: String? = nil,
    transformerBlocks: Int? = nil,
    coreReuse: Int? = nil,
    blockCache: Bool? = nil,
    stillFrameCount: Int? = nil,
    previewDenoise: Bool? = nil,
    seed: UInt64? = nil,
    conditioning: String? = nil,
    speechLanguage: String? = nil,
    speechVariation: Double? = nil,
    voiceName: String? = nil,
    deviceName: String? = nil,
    deviceMemoryBytes: UInt64? = nil
  ) {
    self.kind = kind
    self.seconds = seconds
    self.canvasWidth = canvasWidth
    self.canvasHeight = canvasHeight
    self.denoisingSteps = denoisingSteps
    self.stepLabel = stepLabel
    self.clipSeconds = clipSeconds
    self.modelName = modelName
    self.aspectRatio = aspectRatio
    self.transformerBlocks = transformerBlocks
    self.coreReuse = coreReuse
    self.blockCache = blockCache
    self.stillFrameCount = stillFrameCount
    self.previewDenoise = previewDenoise
    self.seed = seed
    self.conditioning = conditioning
    self.speechLanguage = speechLanguage
    self.speechVariation = speechVariation
    self.voiceName = voiceName
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

    var settings = ["model \(modelName)"]
    if let aspectRatio { settings.append("aspect \(aspectRatio)") }
    if let denoisingSteps { settings.append("\(denoisingSteps) \(stepLabel)") }
    if let transformerBlocks { settings.append("\(transformerBlocks) transformer blocks") }
    if let coreReuse {
      settings.append(coreReuse > 1 ? "core reuse every \(coreReuse)th pass" : "core reuse off")
    }
    if let blockCache { settings.append("block cache \(blockCache ? "on" : "off")") }
    if let stillFrameCount {
      settings.append("still detail \(stillFrameCount == 5 ? "Fast" : "Full") \(stillFrameCount)f")
    }
    if let previewDenoise {
      settings.append("denoising preview \(previewDenoise ? "on" : "off")")
    }
    if let seed { settings.append("seed \(seed)") }
    if let conditioning { settings.append("conditioning \(conditioning)") }
    if let speechLanguage { settings.append("language \(speechLanguage)") }
    if let speechVariation {
      settings.append(String(format: "variation %.2f", speechVariation))
    }
    if let voiceName { settings.append("voice \(voiceName)") }
    sentence += " Settings: " + settings.joined(separator: ", ") + "."
    sentence +=
      " Made with H3ddle, an open-source local MiniMax H3 app for macOS:"
      + " https://github.com/AlexanderIstomin/h3ddle"
    return sentence
  }
}
