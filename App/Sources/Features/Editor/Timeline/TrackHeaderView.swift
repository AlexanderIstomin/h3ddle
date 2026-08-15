import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

enum TimelineChrome {
  static let headerWidth: CGFloat = 128
  static let rulerHeight: CGFloat = 30
  static let visualLaneHeight: CGFloat = 112
  static let audioLaneHeight: CGFloat = 56
  static let effectLaneHeight: CGFloat = 24
  static let appendButtonSize: CGFloat = 36

  static func bodyHeight(showsEffectLanes: Bool, expandedEffectCount: Int = 0) -> CGFloat {
    rulerHeight
      + visualLaneHeight
      + audioLaneHeight
      + effectLanesHeight(
        showsEffectLanes: showsEffectLanes,
        expandedEffectCount: expandedEffectCount
      )
  }

  static func effectLanesHeight(showsEffectLanes: Bool, expandedEffectCount: Int) -> CGFloat {
    guard showsEffectLanes else { return 0 }
    let extra = expandedEffectCount > 0 ? expandedEffectCount : 0
    return effectLaneHeight * CGFloat(1 + extra)
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
        if model.fxLanesExpanded {
          ForEach(model.effectLaneItems) { item in
            effectInstanceHeader(item)
          }
        }
      }
      TrackHeaderView(
        code: "V1",
        title: "Visual",
        tag: H3Color.clipVideo,
        height: TimelineChrome.visualLaneHeight,
        isDisabled: model.visualTrackMuted,
        onToggleEnabled: { model.visualTrackMuted.toggle() }
      )
      .timelineMediaDrop(lane: .visual, model: model, accessibilityID: "visual-header-drop")
      TrackHeaderView(
        code: "A1",
        title: "Audio",
        tag: H3Color.clipAudio,
        height: TimelineChrome.audioLaneHeight,
        isDisabled: model.audioTrackMuted,
        onToggleEnabled: { model.audioTrackMuted.toggle() }
      )
      .timelineMediaDrop(lane: .audio, model: model, accessibilityID: "audio-header-drop")
    }
    .frame(width: TimelineChrome.headerWidth, alignment: .top)
    .background(H3Color.chrome)
    .overlay(alignment: .trailing) {
      Rectangle().fill(H3Color.line).frame(width: 1)
    }
  }

  private var effectHeader: some View {
    HStack(spacing: 5) {
      Image(systemName: "sparkle")
        .font(.system(size: 9, weight: .semibold))
        .foregroundStyle(Color(red: 47 / 255, green: 179 / 255, blue: 191 / 255))
      Text("FX")
        .font(.system(size: 8, weight: .bold, design: .monospaced))
        .tracking(1.4)
        .foregroundStyle(H3Color.textSecondary.opacity(0.7))
      Spacer(minLength: 2)
      if !model.effectLaneItems.isEmpty {
        Button {
          model.fxLanesExpanded.toggle()
        } label: {
          Image(systemName: model.fxLanesExpanded ? "chevron.up" : "chevron.down")
            .font(.system(size: 8, weight: .bold))
            .frame(width: 16, height: 16)
            .background(H3Color.controlFill)
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
        }
        .buttonStyle(.plain)
        .foregroundStyle(H3Color.textSecondary)
        .help(model.fxLanesExpanded ? "Collapse FX lanes" : "Expand FX lanes")
        .accessibilityIdentifier("fx-expand")
        .accessibilityLabel(model.fxLanesExpanded ? "Collapse FX lanes" : "Expand FX lanes")
      }
      Button {
        model.openEffectsCatalog()
      } label: {
        Image(systemName: "plus")
          .font(.system(size: 10, weight: .semibold))
          .frame(width: 16, height: 16)
          .background(H3Color.controlFill)
          .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
      }
      .buttonStyle(.plain)
      .foregroundStyle(H3Color.textSecondary)
      .disabled(model.visualTrackMuted || model.project.timeline.visualItems.isEmpty)
      .opacity(model.visualTrackMuted || model.project.timeline.visualItems.isEmpty ? 0.34 : 1)
      .help(
        model.project.timeline.visualItems.isEmpty
          ? "Add a visual clip to apply effects"
          : "Add effect"
      )
      .accessibilityIdentifier("append-effect")
    }
    .padding(.horizontal, 8)
    .frame(width: TimelineChrome.headerWidth, height: TimelineChrome.effectLaneHeight)
    .background(Color.white.opacity(0.045))
    .overlay(alignment: .bottom) {
      Rectangle().fill(H3Color.hairSoft).frame(height: 1)
    }
    .opacity(model.visualTrackMuted ? 0.55 : 1)
  }

  private func effectInstanceHeader(_ item: EffectLaneItem) -> some View {
    Button {
      model.openEffectSettings(clipID: item.clipID, effectID: item.effect.id)
    } label: {
      HStack(spacing: 7) {
        RoundedRectangle(cornerRadius: 3, style: .continuous)
          .fill(item.effect.kind.swatch)
          .frame(width: 9, height: 9)
        Text(item.effect.kind.label)
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .lineLimit(1)
        Spacer(minLength: 0)
      }
      .padding(.horizontal, 11)
      .frame(width: TimelineChrome.headerWidth, height: TimelineChrome.effectLaneHeight, alignment: .leading)
      .background(
        model.selectedEffectID == item.effect.id
          ? H3Color.controlHover
          : Color.white.opacity(0.045)
      )
      .overlay(alignment: .bottom) {
        Rectangle().fill(H3Color.hairSoft).frame(height: 1)
      }
    }
    .buttonStyle(.plain)
    .opacity(model.visualTrackMuted ? 0.55 : 1)
    .accessibilityIdentifier("fx-effect-row")
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
