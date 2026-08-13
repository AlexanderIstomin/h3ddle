import H3ddleDesignSystem
import SwiftUI

enum TimelineChrome {
  static let headerWidth: CGFloat = 128
  static let rulerHeight: CGFloat = 30
  static let visualLaneHeight: CGFloat = 112
  static let audioLaneHeight: CGFloat = 56
  static let effectLaneHeight: CGFloat = 24
  static let appendButtonSize: CGFloat = 36

  static func bodyHeight(showsEffectLanes: Bool) -> CGFloat {
    rulerHeight
      + visualLaneHeight
      + audioLaneHeight
      + (showsEffectLanes ? effectLaneHeight * 2 : 0)
  }
}

struct TrackHeaderColumn: View {
  @Bindable var model: AppModel

  var body: some View {
    VStack(spacing: 0) {
      HStack {
        Text("TRACKS")
          .font(.system(size: 8, weight: .medium, design: .monospaced))
          .tracking(1.6)
          .foregroundStyle(H3Color.textSecondary.opacity(0.55))
          .accessibilityIdentifier("program-timeline")
        Spacer()
      }
      .padding(.horizontal, 8)
      .frame(width: TimelineChrome.headerWidth, height: TimelineChrome.rulerHeight, alignment: .leading)
      .overlay(alignment: .bottom) {
        Rectangle().fill(H3Color.line).frame(height: 1)
      }

      if model.showsEffectLanes {
        effectHeader
      }
      TrackHeaderView(
        code: "V1",
        title: "Visual",
        tag: H3Color.clipVideo,
        height: TimelineChrome.visualLaneHeight,
        isDisabled: model.visualTrackMuted,
        onToggleEnabled: { model.visualTrackMuted.toggle() }
      )
      if model.showsEffectLanes {
        effectHeader
      }
      TrackHeaderView(
        code: "A1",
        title: "Audio",
        tag: H3Color.clipAudio,
        height: TimelineChrome.audioLaneHeight,
        isDisabled: model.audioTrackMuted,
        onToggleEnabled: { model.audioTrackMuted.toggle() }
      )
    }
    .frame(width: TimelineChrome.headerWidth, alignment: .top)
    .background(H3Color.chrome)
    .overlay(alignment: .trailing) {
      Rectangle().fill(H3Color.line).frame(width: 1)
    }
  }

  private var effectHeader: some View {
    HStack {
      Text("FX")
        .font(.system(size: 8, weight: .bold, design: .monospaced))
        .tracking(1.4)
        .foregroundStyle(H3Color.textSecondary.opacity(0.7))
      Spacer()
    }
    .padding(.horizontal, 8)
    .frame(width: TimelineChrome.headerWidth, height: TimelineChrome.effectLaneHeight)
    .background(H3Color.hairSoft)
  }
}

struct TrackHeaderView: View {
  var code: String
  var title: String
  var tag: Color
  var height: CGFloat
  var isDisabled: Bool
  var onToggleEnabled: () -> Void

  var body: some View {
    HStack(spacing: 7) {
      RoundedRectangle(cornerRadius: 2, style: .continuous)
        .fill(tag)
        .frame(width: 3, height: 24)
      VStack(alignment: .leading, spacing: 1) {
        Text(code)
          .font(.system(size: 13, weight: .bold, design: .monospaced))
          .tracking(0.6)
          .lineLimit(1)
        Text(title)
          .font(.system(size: 7.5, weight: .medium))
          .foregroundStyle(H3Color.textSecondary.opacity(0.75))
          .lineLimit(1)
      }
      Spacer(minLength: 4)
      Button(action: onToggleEnabled) {
        Image(systemName: "power")
          .font(.system(size: 12, weight: .semibold))
          .frame(width: 22, height: 22)
      }
      .buttonStyle(.plain)
      .foregroundStyle(isDisabled ? Color.white : H3Color.textSecondary.opacity(0.75))
      .background(isDisabled ? H3Color.accent : H3Color.controlFill)
      .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
      .help(isDisabled ? "Enable track" : "Disable track")
      .accessibilityLabel(isDisabled ? "Enable track" : "Disable track")
    }
    .padding(.horizontal, 8)
    .frame(width: TimelineChrome.headerWidth, height: height, alignment: .leading)
    .overlay(alignment: .bottom) {
      Rectangle().fill(H3Color.hairSoft).frame(height: 1)
    }
    .opacity(isDisabled ? 0.55 : 1)
  }
}
