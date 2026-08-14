import Testing

@testable import H3ddleGeneration

@Suite("Structured prompts")
struct H3StructuredPromptTests {
  @Test("A bare prompt gains the schema and the no-music marker")
  func wrapsBareProse() {
    let composed = H3StructuredPrompt.compose(body: "A cat sits on a windowsill.")
    #expect(
      composed == """
        integrated_multimodal_description: [Shot 1] A cat sits on a windowsill.

        non_diegetic_music: N/A
        """
    )
  }

  @Test("A provided soundscape becomes its own field")
  func includesSoundscape() {
    let composed = H3StructuredPrompt.compose(
      body: "Rain falls over a forest at night.",
      soundscape: "Steady rain patters on leaves while distant thunder rolls."
    )
    #expect(
      composed.contains(
        "\n\noverall_soundscape: Steady rain patters on leaves while distant thunder rolls."
      )
    )
    #expect(composed.hasSuffix("non_diegetic_music: N/A"))
  }

  @Test("Music text replaces the N/A marker")
  func includesMusic() {
    let composed = H3StructuredPrompt.compose(
      body: "A dancer crosses the stage.",
      music: "Sparse piano notes at a slow tempo."
    )
    #expect(composed.hasSuffix("non_diegetic_music: Sparse piano notes at a slow tempo."))
  }

  @Test("Hand-written schema passes through untouched")
  func passesThroughStructuredBodies() {
    let handWritten = """
      integrated_multimodal_description: [Shot 1] A baker opens the shutters.

      overall_soundscape: Shutters scrape over a quiet street.

      non_diegetic_music: N/A
      """
    #expect(
      H3StructuredPrompt.compose(body: handWritten, soundscape: "ignored") == handWritten
    )
  }

  @Test("Bodies that open with a shot label are not double-labelled")
  func keepsExistingShotLabels() {
    let composed = H3StructuredPrompt.compose(body: "[Shot 1] A train leaves the station.")
    #expect(
      composed.hasPrefix(
        "integrated_multimodal_description: [Shot 1] A train leaves the station."
      )
    )
    #expect(!composed.contains("[Shot 1] [Shot 1]"))
  }

  @Test("An end frame without a start frame gains the alignment sentence")
  func alignsLoneEndFrames() {
    let composed = H3StructuredPrompt.compose(
      body: "The skyline settles into dusk.",
      endFrameAlignmentSeconds: 3.04
    )
    #expect(
      composed.hasPrefix(
        "How the reference pictures align with the target video — <Picture 1> "
          + "(from [Shot 1]) aligns with the 3.04-second mark of the target video."
      )
    )
  }

  @Test("Image generations carry no audio fields")
  func imagesDropAudioFields() {
    let composed = H3StructuredPrompt.compose(
      body: "A lighthouse at dawn.",
      soundscape: "Waves",
      music: "Strings",
      kind: .image
    )
    #expect(composed == "integrated_multimodal_description: [Shot 1] A lighthouse at dawn.")
  }

  @Test("Whitespace-only bodies stay empty")
  func leavesEmptyBodiesAlone() {
    #expect(H3StructuredPrompt.compose(body: "  \n ") == "")
  }
}
