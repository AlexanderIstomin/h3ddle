import Foundation

/// The synthesis budgets compared on the same YuE2 score and semantic tokens.
/// These are midpoint solver steps, not autoregressive token counts.
public enum YuE2SynthesisSteps: Int, CaseIterable, Codable, Sendable {
  case fast = 8
  case balanced = 16
  case reference = 32
}

public struct EngineMusicOptions: Hashable, Codable, Sendable {
  public var lyrics: String

  public init(lyrics: String) {
    self.lyrics = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
  }
}
