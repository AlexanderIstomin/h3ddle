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

  @Test("Visible tick generation stays bounded on long timelines")
  func visibleTicksStayBounded() {
    let ticks = TimelineRuler.ticks(
      duration: 10_000,
      pointsPerSecond: 13,
      visibleRange: 500...510
    )
    #expect(
      ticks.map(\.time) == [500, 501.25, 502.5, 503.75, 505, 506.25, 507.5, 508.75, 510]
    )
    #expect(ticks.filter(\.isMajor).map(\.time) == [500, 505, 510])
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

  @Test("Visible range includes overscan and rejects distant spans")
  func visibleRangeSupportsTimelineVirtualization() {
    let range = TimelineRuler.visibleTimeRange(
      scrollOffset: 400,
      viewportWidth: 200,
      pointsPerSecond: 20,
      overscanPoints: 100
    )
    #expect(range == 15...35)
    #expect(TimelineRuler.spanIntersects(start: 14, duration: 2, visibleRange: range))
    #expect(TimelineRuler.spanIntersects(start: 34, duration: 2, visibleRange: range))
    #expect(!TimelineRuler.spanIntersects(start: 10, duration: 2, visibleRange: range))
    #expect(!TimelineRuler.spanIntersects(start: 40, duration: 2, visibleRange: range))
  }

  @Test("Filmstrip samples cover the trimmed clip without requesting its exact out point")
  func filmstripSamplesTrimmedRange() {
    let times = TimelineFilmstripSampling.times(
      sourceOffset: 2,
      duration: 6,
      frameCount: 3
    )
    #expect(times.count == 3)
    #expect(abs(times[0] - 2) < 1e-6)
    #expect(abs(times[1] - 4.999_166_666_7) < 1e-6)
    #expect(times[2] < 8)
    #expect(times[2] > 7.99)
  }
}
