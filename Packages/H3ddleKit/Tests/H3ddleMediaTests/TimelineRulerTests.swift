import Foundation
import Testing

@testable import H3ddleMedia

@Suite("Timeline ruler")
struct TimelineRulerTests {
  @Test("Major ticks keep at least 60 pixels between labels")
  func majorTicksFollowZoom() {
    #expect(TimelineRuler.majorInterval(pointsPerSecond: 13) == 5)
    let ticks = TimelineRuler.ticks(duration: 20, pointsPerSecond: 13)
    let majors = ticks.filter(\.isMajor)
    #expect(majors.map(\.time) == [0, 5, 10, 15, 20])
    #expect(majors[1].label == "0:05")
    #expect(TimelineRuler.drawsMinorTicks(pointsPerSecond: 13))
    #expect(TimelineRuler.minorInterval(pointsPerSecond: 13) == 1.25)
  }

  @Test("Content duration never collapses below the fallback window")
  func contentDurationHasAFloor() {
    #expect(TimelineRuler.contentDuration(visualDuration: 4, audioTrackEnd: 6) == 180)
    #expect(TimelineRuler.contentDuration(visualDuration: 200, audioTrackEnd: 40) == 200)
    #expect(
      TimelineRuler.contentDuration(visualDuration: 4, audioTrackEnd: 6, textTrackEnd: 220) == 220
    )
  }

  @Test("Zoom is clamped to the pinch range")
  func clampsZoom() {
    #expect(TimelineRuler.pointsPerSecond(zoom: 0.01) == 13 * 0.05)
    #expect(TimelineRuler.pointsPerSecond(zoom: 20) == 13 * 8)
  }

  @Test("Anchored zoom keeps the time under the pointer")
  func anchoredZoomKeepsPointerTime() {
    let currentZoom = 1.0
    let nextZoom = 2.0
    let scroll = 130.0
    let anchor = 65.0
    let newScroll = TimelineRuler.anchoredScrollOffset(
      currentZoom: currentZoom,
      nextZoom: nextZoom,
      scrollOffset: scroll,
      anchor: anchor
    )
    let timeBefore = (scroll + anchor) / TimelineRuler.pointsPerSecond(zoom: currentZoom)
    let timeAfter = (newScroll + anchor) / TimelineRuler.pointsPerSecond(zoom: nextZoom)
    #expect(abs(timeBefore - timeAfter) < 0.000_1)
  }
}
