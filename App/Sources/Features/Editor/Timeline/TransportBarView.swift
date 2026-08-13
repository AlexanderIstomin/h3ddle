import AppKit
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct TransportBarView: View {
  @Bindable var model: AppModel

  private var sliderZoom: Binding<Double> {
    Binding(
      get: {
        min(max(model.timelineZoom, TimelineRuler.sliderMinimumZoom), TimelineRuler.sliderMaximumZoom)
      },
      set: { model.setTimelineZoom($0) }
    )
  }

  var body: some View {
    HStack(spacing: 14) {
      VStack(alignment: .leading, spacing: 1) {
        Text(model.playback.clock.formattedTimecode())
          .font(.system(size: 21, weight: .semibold, design: .monospaced))
          .monospacedDigit()
          .lineLimit(1)
        Text(
          ProgramClock.formatTimecode(
            model.programDuration,
            framesPerSecond: model.project.settings.framesPerSecond
          )
        )
          .font(.system(size: 11, design: .monospaced))
          .tracking(0.8)
          .foregroundStyle(H3Color.textSecondary.opacity(0.55))
          .monospacedDigit()
          .lineLimit(1)
      }
      .frame(width: 180, alignment: .leading)

      Spacer(minLength: 0)

      HStack(spacing: 6) {
        Button(action: model.skipToStart) {
          Image(systemName: "backward.end.fill")
        }
        .buttonStyle(H3IconButtonStyle())
        .help("Skip to start")

        Button(action: model.togglePlayback) {
          Image(systemName: model.playback.isPlaying ? "pause.fill" : "play.fill")
            .font(.system(size: 19, weight: .semibold))
            .offset(x: model.playback.isPlaying ? 0 : 1)
            .foregroundStyle(Color.white)
            .frame(width: 40, height: 40)
            .background(H3Color.accent)
            .clipShape(Circle())
            .shadow(color: H3Color.accent.opacity(0.45), radius: 8, y: 3)
        }
        .buttonStyle(.plain)
        .help(model.playback.isPlaying ? "Pause" : "Play")
        .accessibilityIdentifier("transport-play")

        Button(action: { model.playback.clock.isLooping.toggle() }) {
          Image(systemName: "repeat")
        }
        .buttonStyle(H3IconButtonStyle(isActive: model.playback.clock.isLooping))
        .help("Toggle loop")
      }

      Spacer(minLength: 0)

      HStack(spacing: 10) {
        Button(action: { model.showsEffectLanes.toggle() }) {
          Image(systemName: "sparkle")
        }
        .buttonStyle(H3IconButtonStyle(isActive: model.showsEffectLanes, size: 30))
        .help("Effect lanes")

        HStack(spacing: 6) {
          Button(action: { model.adjustTimelineZoom(-0.3) }) {
            Image(systemName: "minus.magnifyingglass")
          }
          .buttonStyle(H3IconButtonStyle(size: 30))
          .help("Zoom out")
          TicklessSlider(value: sliderZoom, range: 0.4...3)
            .frame(width: 80, height: 16)
          Button(action: { model.adjustTimelineZoom(0.3) }) {
            Image(systemName: "plus.magnifyingglass")
          }
          .buttonStyle(H3IconButtonStyle(size: 30))
          .help("Zoom in")
        }
      }
    }
    .padding(.horizontal, 16)
    .frame(minHeight: 50)
    .background(H3Color.transport)
    .overlay(alignment: .bottom) {
      Rectangle().fill(H3Color.line).frame(height: 1)
    }
  }
}

private struct TicklessSlider: NSViewRepresentable {
  @Binding var value: Double
  var range: ClosedRange<Double>

  func makeNSView(context: Context) -> NSSlider {
    let slider = NSSlider(
      value: value,
      minValue: range.lowerBound,
      maxValue: range.upperBound,
      target: context.coordinator,
      action: #selector(Coordinator.changed(_:))
    )
    slider.isContinuous = true
    slider.numberOfTickMarks = 0
    slider.allowsTickMarkValuesOnly = false
    slider.controlSize = .small
    slider.trackFillColor = NSColor(
      red: 224 / 255,
      green: 82 / 255,
      blue: 28 / 255,
      alpha: 1
    )
    return slider
  }

  func updateNSView(_ slider: NSSlider, context: Context) {
    if abs(slider.doubleValue - value) > 0.000_1 {
      slider.doubleValue = value
    }
  }

  func makeCoordinator() -> Coordinator {
    Coordinator(value: $value)
  }

  @MainActor
  final class Coordinator: NSObject {
    var value: Binding<Double>

    init(value: Binding<Double>) {
      self.value = value
    }

    @objc
    func changed(_ sender: NSSlider) {
      value.wrappedValue = sender.doubleValue
    }
  }
}
