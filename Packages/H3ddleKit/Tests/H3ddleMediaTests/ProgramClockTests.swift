import Foundation
import Testing

@testable import H3ddleMedia

@Suite("Program clock")
struct ProgramClockTests {
  @Test("Step moves by whole frames and clamps to the visual duration")
  func stepsAndClamps() {
    var clock = ProgramClock()
    clock.step(frames: 12, duration: 2)
    #expect(clock.currentTime == 0.5)

    clock.step(frames: 100, duration: 2)
    #expect(clock.currentTime == 2)

    clock.setTime(-4, duration: 2)
    #expect(clock.currentTime == 0)
  }

  @Test("Advance stops at the end unless looping")
  func advanceStopsUnlessLooping() {
    var clock = ProgramClock(currentTime: 1.8, isPlaying: true)
    #expect(clock.advance(by: 0.5, duration: 2) == false)
    #expect(clock.currentTime == 2)
    #expect(clock.isPlaying == false)

    clock = ProgramClock(currentTime: 1.8, isPlaying: true, isLooping: true)
    #expect(clock.advance(by: 0.5, duration: 2) == true)
    #expect(abs(clock.currentTime - 0.3) < 0.000_1)
    #expect(clock.isPlaying == true)
  }

  @Test("An empty program stays at zero")
  func emptyProgramStaysAtZero() {
    var clock = ProgramClock(currentTime: 4, isPlaying: true)
    #expect(clock.advance(by: 0.1, duration: 0) == false)
    #expect(clock.currentTime == 0)
    #expect(clock.isPlaying == false)
  }

  @Test("Timecode uses HH:MM:SS:FF at 24 fps")
  func formatsTimecode() {
    #expect(ProgramClock.formatTimecode(0) == "00:00:00:00")
    #expect(ProgramClock.formatTimecode(16.5) == "00:00:16:12")
    #expect(ProgramClock.formatShort(90) == "1:30")
  }

  @Test("Timecode frame digits follow the project rate")
  func formatsTimecodeAtProjectRate() {
    #expect(ProgramClock.formatTimecode(1, framesPerSecond: 30) == "00:00:01:00")
    #expect(ProgramClock.formatTimecode(1.5, framesPerSecond: 30) == "00:00:01:15")

    var clock = ProgramClock(framesPerSecond: 30)
    clock.step(frames: 15, duration: 2)
    #expect(abs(clock.currentTime - 0.5) < 0.000_1)
    #expect(clock.formattedTimecode() == "00:00:00:15")
  }
}
