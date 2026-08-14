import Foundation

/// Composes prompts in the three-field schema MiniMax H3 was trained on,
/// documented in the model repository's prompt-writing guides
/// (MiniMaxAI/MiniMax-H3, docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
///
/// H3 reads a prompt as labelled fields — the audiovisual timeline, the
/// ambient soundscape, and non-diegetic music — not as free prose. Raw prose
/// measurably misleads it: the model has been observed reciting the prompt
/// text aloud as speech and rendering it as handwritten characters, because
/// nothing marked the words as a scene to depict rather than content to
/// reproduce. Wrapping the body in the trained schema stopped the recitation
/// in direct A/B runs at a fixed seed.
public enum H3StructuredPrompt {
  static let descriptionLabel = "integrated_multimodal_description:"
  static let soundscapeLabel = "overall_soundscape:"
  static let musicLabel = "non_diegetic_music:"

  /// Wraps `body` in the trained schema.
  ///
  /// A body that already carries the description label passes through
  /// unchanged, so hand-written schema prompts are never re-wrapped. The
  /// soundscape line appears only when the caller provides one — the composer
  /// never invents ambience. An empty music field becomes the guide's `N/A`
  /// marker, which is its documented "no background music" value. Image
  /// generations drop both audio fields: a still has no soundtrack to
  /// describe.
  ///
  /// `endFrameAlignmentSeconds` is the effective clip length and belongs only
  /// to generations conditioned on a last frame without a first frame; the
  /// guide requires those to open with a sentence placing the picture at the
  /// clip's final second. Keyframed generations with a first frame need no
  /// leading sentence.
  public static func compose(
    body: String,
    soundscape: String? = nil,
    music: String? = nil,
    kind: GenerationKind = .video,
    endFrameAlignmentSeconds: Double? = nil
  ) -> String {
    let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedBody.isEmpty, !trimmedBody.contains(descriptionLabel) else {
      return trimmedBody
    }

    var sections: [String] = []
    if let seconds = endFrameAlignmentSeconds {
      let mark = String(format: "%.2f", seconds)
      sections.append(
        "How the reference pictures align with the target video — <Picture 1> "
          + "(from [Shot 1]) aligns with the \(mark)-second mark of the target "
          + "video."
      )
    }

    let shotBody =
      trimmedBody.hasPrefix("[Shot") ? trimmedBody : "[Shot 1] " + trimmedBody
    sections.append("\(descriptionLabel) \(shotBody)")

    if kind != .image {
      let trimmedSoundscape = soundscape?.trimmingCharacters(
        in: .whitespacesAndNewlines)
      if let trimmedSoundscape, !trimmedSoundscape.isEmpty {
        sections.append("\(soundscapeLabel) \(trimmedSoundscape)")
      }
      let trimmedMusic = music?.trimmingCharacters(in: .whitespacesAndNewlines)
      if let trimmedMusic, !trimmedMusic.isEmpty {
        sections.append("\(musicLabel) \(trimmedMusic)")
      } else {
        sections.append("\(musicLabel) N/A")
      }
    }

    return sections.joined(separator: "\n\n")
  }
}
