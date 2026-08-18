import Testing

@testable import H3ddleGeneration

@Suite("Supported length")
struct SupportedLengthTests {
  /// The numbers the released pipeline documents as trained, expressed as
  /// seconds so a control can offer them directly.
  @Test("H3's video lane offers exactly the lengths on its frame grid")
  func h3VideoGrid() {
    let length = SupportedLength.h3Video
    #expect(length.options.count == 15)
    #expect(abs(length.minimumSeconds - 124.0 / 24) < 1e-9)
    #expect(abs(length.maximumSeconds - 362.0 / 24) < 1e-9)
    // Every option lands on a whole number of frames, on the 17k+5 grid.
    for seconds in length.options {
      let frames = Int((seconds * 24).rounded())
      #expect((frames - 5) % 17 == 0, "\(frames) is off the grid")
      #expect(frames >= 124 && frames <= 362)
    }
  }

  @Test("Requests are resolved onto the grid, never below the floor")
  func resolution() {
    let length = SupportedLength.h3Video
    // Below the floor comes up to it rather than being refused.
    #expect(length.resolved(0.9) == length.minimumSeconds)
    #expect(length.resolved(3.0) == length.minimumSeconds)
    // Above the ceiling comes down to it.
    #expect(length.resolved(60) == length.maximumSeconds)
    // Between grid points, down to the one asked past — never silently longer.
    let second = length.options[1]
    #expect(length.resolved(second - 0.01) == length.options[0])
    #expect(length.resolved(second) == second)
    // Everything it returns is a length the model will actually generate.
    for request in stride(from: 0.5, through: 20.0, by: 0.37) {
      #expect(length.options.contains { abs($0 - length.resolved(request)) < 1e-9 })
    }
  }

  /// Speech is the continuous case: its length is whatever the line takes to
  /// say, so the range is a ceiling and any value inside it is legal. Stable
  /// Audio moved onto a one-second grid once the shipped reference was read,
  /// so it is no longer an example of this.
  @Test("A continuous range has no grid and passes values through")
  func continuousRanges() {
    let speech = SupportedLength.speechCeiling
    #expect(speech.options.isEmpty)
    #expect(speech.resolved(7.3) == 7.3)
    #expect(speech.resolved(0.2) == speech.minimumSeconds)
    #expect(speech.resolved(500) == speech.maximumSeconds)
  }

  /// The numbers come from the released pipeline, not from this app: its
  /// slider is `minimum=1, step=1` and its per-checkpoint ceiling table reads
  /// `{"sm-music": 120, "sm-sfx": 120, "medium": 380}`. The checkpoint's own
  /// sample_size agrees — 5,292,032 samples at 44,100 is 120 seconds.
  @Test("Stable Audio offers whole seconds across the window it ships with")
  func stableAudioMatchesTheShippedRange() {
    let sa = SupportedLength.stableAudio
    #expect(sa.minimumSeconds == 1)
    #expect(sa.maximumSeconds == 120)
    #expect(5_292_032.0 / 44_100.0 >= 120)
    #expect(sa.resolved(0.5) == 1)
    #expect(sa.resolved(7.4) == 7)
    #expect(sa.resolved(1000) == 120)
    #expect(sa.options.count == 120)
  }
}
