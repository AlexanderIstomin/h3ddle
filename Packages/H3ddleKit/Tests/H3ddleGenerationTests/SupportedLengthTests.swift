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

  @Test("A continuous range has no grid and passes values through")
  func continuousRanges() {
    let audio = SupportedLength.stableAudio
    #expect(audio.options.isEmpty)
    #expect(audio.resolved(7.3) == 7.3)
    #expect(audio.resolved(0.2) == audio.minimumSeconds)
    #expect(audio.resolved(500) == audio.maximumSeconds)
  }
}
