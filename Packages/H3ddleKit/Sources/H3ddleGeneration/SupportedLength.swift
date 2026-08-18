import Foundation

/// The clip lengths a model will actually generate.
///
/// Not a user preference and not a safety margin. A generative model asked for
/// a length outside what it was trained on does not refuse and does not
/// degrade visibly — it returns confident, well-made output with no relation
/// to the prompt. H3 read as broken for a week on exactly that: every clip the
/// app could produce was shorter than the shortest it was trained for, and
/// about half of seeds came back as an unrelated scene.
///
/// So each engine states its own range here, and the studio can only offer
/// what the engine will honour.
public struct SupportedLength: Equatable, Sendable {
  public let minimumSeconds: Double
  public let maximumSeconds: Double
  /// The spacing between legal lengths, when a model only accepts a grid —
  /// H3's latent stack advances 17 frames at a time and nothing lands
  /// between. Nil where any length in the range is legal.
  public let stepSeconds: Double?

  public init(minimumSeconds: Double, maximumSeconds: Double,
              stepSeconds: Double? = nil) {
    self.minimumSeconds = minimumSeconds
    self.maximumSeconds = maximumSeconds
    self.stepSeconds = stepSeconds
  }

  /// The nearest length this model will actually generate, at or above the
  /// floor. Rounds down within the range so a request is never silently
  /// extended past what was asked for, except at the minimum.
  public func resolved(_ seconds: Double) -> Double {
    let clamped = min(max(seconds, minimumSeconds), maximumSeconds)
    guard let stepSeconds, stepSeconds > 0 else { return clamped }
    // The tolerance is load-bearing: a length that is exactly one step above
    // the floor divides to 0.9999999 and floors to zero, which silently
    // returns the shortest clip for a request that named the second one.
    let steps = ((clamped - minimumSeconds) / stepSeconds + 1e-9).rounded(.down)
    return min(minimumSeconds + steps * stepSeconds, maximumSeconds)
  }

  /// Every legal length, for a control that offers choices rather than a
  /// continuous slide. Empty when the range is continuous.
  public var options: [Double] {
    guard let stepSeconds, stepSeconds > 0 else { return [] }
    var values: [Double] = []
    var value = minimumSeconds
    while value <= maximumSeconds + 1e-9 {
      values.append(value)
      value += stepSeconds
    }
    return values
  }

  public var stepCount: Int { max(0, options.count - 1) }

  /// H3's video lane: 124 to 362 frames at 24 fps on the 17k+5 grid, which is
  /// the range the released pipeline documents as trained.
  public static let h3Video = SupportedLength(
    minimumSeconds: Double(H3Duration.minimumFrames) / H3Duration.fps,
    maximumSeconds: Double(H3Duration.maximumFrames) / H3Duration.fps,
    stepSeconds: Double(H3Duration.chunk) / H3Duration.fps
  )

  /// LTX-2.5's clip lengths. Nothing like H3's, in either direction.
  ///
  /// The video VAE compresses 8x in time and cannot produce the seven leading
  /// frames, so a clip is 8k+1 frames and nothing between — a third of a second
  /// apart at 24 fps. The floor is 17 frames because that is the shortest thing
  /// the decoder makes; H3's floor of 124 would put the cheapest LTX clip at
  /// eight minutes and make the two-second clips this port was validated on
  /// unrequestable.
  ///
  /// The ceiling is this app's, not the model's. Cost is linear in tokens and
  /// tokens are linear in length, so 193 frames at 512 square is already about
  /// thirteen minutes of denoising on an M1 Pro. Longer works and is simply not
  /// worth offering by default.
  public static let ltxVideo = SupportedLength(
    minimumSeconds: 17.0 / 24.0,
    maximumSeconds: 193.0 / 24.0,
    stepSeconds: 8.0 / 24.0
  )

  /// Both small Stable Audio 3 checkpoints, music and sound effects, take one
  /// second to a hundred and twenty in whole seconds. That is the range the
  /// model ships with rather than one this app chose: the released pipeline
  /// offers `minimum=1, step=1` against a per-checkpoint table reading
  /// `{"sm-music": 120, "sm-sfx": 120, "medium": 380}`, and the checkpoint's
  /// own `sample_size` of 5,292,032 samples at 44.1 kHz is 120 seconds to
  /// four decimal places. The medium checkpoint reaches 380 and would need
  /// its own entry here if it is ever packaged.
  public static let stableAudio = SupportedLength(
    minimumSeconds: 1, maximumSeconds: 120, stepSeconds: 1)

  /// Speech has no length to choose: the line decides, and the number is a
  /// ceiling on a runaway rather than a target.
  public static let speechCeiling = SupportedLength(
    minimumSeconds: 1, maximumSeconds: 60)

  /// A still has no clip length; the number is how long it is shown for.
  public static let still = SupportedLength(minimumSeconds: 1, maximumSeconds: 30)
}
