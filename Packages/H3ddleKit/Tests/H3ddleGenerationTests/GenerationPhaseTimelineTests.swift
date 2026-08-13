import Foundation
import Testing

@testable import H3ddleGeneration

@Suite("Generation phase timeline")
struct GenerationPhaseTimelineTests {
  @Test("Phase changes close the previous phase")
  func phaseDurations() {
    var timeline = GenerationPhaseTimeline()
    timeline.record(phase: "text encoder", elapsed: 0)
    timeline.record(phase: "text encoder", elapsed: 4)
    timeline.record(phase: "denoise", elapsed: 10)
    timeline.record(phase: "video VAE", elapsed: 70.5)
    timeline.finish(elapsed: 80)

    #expect(
      timeline.entries == [
        GenerationPhaseTimeline.Entry(phase: "text encoder", duration: 10),
        GenerationPhaseTimeline.Entry(phase: "denoise", duration: 60.5),
        GenerationPhaseTimeline.Entry(phase: "video VAE", duration: 9.5),
      ]
    )
    #expect(timeline.summary == "text encoder 10.0s · denoise 60.5s · video VAE 9.5s")
  }

  @Test("Recurring phases are recorded as separate entries")
  func recurringPhases() {
    var timeline = GenerationPhaseTimeline()
    timeline.record(phase: "denoise", elapsed: 0)
    timeline.record(phase: "preview VAE load", elapsed: 5)
    timeline.record(phase: "denoise", elapsed: 8)
    timeline.finish(elapsed: 20)

    #expect(timeline.entries.map(\.phase) == ["denoise", "preview VAE load", "denoise"])
    #expect(timeline.entries.map(\.duration) == [5, 3, 12])
  }

  @Test("An empty timeline has no summary and finishing twice is harmless")
  func emptyTimeline() {
    var timeline = GenerationPhaseTimeline()
    #expect(timeline.summary == nil)
    timeline.finish(elapsed: 10)
    timeline.finish(elapsed: 20)
    #expect(timeline.entries.isEmpty)
  }
}
