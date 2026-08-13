import H3ddleCore
import H3ddleDesignSystem
import H3ddleGeneration
import SwiftUI

struct ProgramTimelineView: View {
  @Bindable var model: AppModel

  private let pointsPerSecond: CGFloat = 18

  var body: some View {
    VStack(spacing: 0) {
      timelineHeader
      timeRuler
      Divider().overlay(H3Color.line)
      visualTrack
      Divider().overlay(H3Color.line.opacity(0.7))
      audioTrack
    }
    .background(H3Color.surface)
  }

  private var timelineHeader: some View {
    HStack {
      Text("PROGRAM")
        .font(.system(size: 10, weight: .bold))
        .tracking(1.1)
        .foregroundStyle(H3Color.textSecondary)
        .accessibilityIdentifier("program-timeline")
      Spacer()
      Text(durationLabel(model.project.timeline.visualDuration))
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
    }
    .padding(.horizontal, H3Spacing.medium)
    .frame(height: 34)
  }

  private var timeRuler: some View {
    HStack(spacing: 0) {
      Color.clear.frame(width: 116)
      GeometryReader { proxy in
        Path { path in
          let count = max(1, Int(proxy.size.width / 90))
          for index in 0...count {
            let x = CGFloat(index) * 90
            path.move(to: CGPoint(x: x, y: 12))
            path.addLine(to: CGPoint(x: x, y: 21))
          }
        }
        .stroke(H3Color.line, lineWidth: 1)
      }
    }
    .frame(height: 24)
  }

  private var visualTrack: some View {
    TrackRow(label: "Visual", symbol: "rectangle.on.rectangle") {
      ScrollView(.horizontal) {
        HStack(spacing: 2) {
          ForEach(model.project.timeline.visualItems) { item in
            TimelineItemView(
              title: model.project.asset(id: item.assetID)?.displayName ?? "Visual",
              duration: item.duration,
              isEnabled: item.isEnabled,
              tint: H3Color.accent,
              pointsPerSecond: pointsPerSecond
            )
            .contextMenu {
              Button(item.isEnabled ? "Disable" : "Enable") {
                model.toggleVisual(item.id)
              }
              Button(item.includesNativeAudio ? "Mute native audio" : "Include native audio") {
                model.toggleVisualNativeAudio(item.id)
              }
              Divider()
              Button("Remove", role: .destructive) {
                model.removeVisual(item.id)
              }
            }
          }

          Menu {
            Button("Generate video") { model.presentGeneration(.video) }
            Button("Generate image") { model.presentGeneration(.image) }
          } label: {
            Label("Append visual", systemImage: "plus")
          }
          .menuStyle(.borderlessButton)
          .buttonStyle(H3QuietButtonStyle())
          .fixedSize()
          .accessibilityIdentifier("append-visual")
        }
        .padding(.horizontal, H3Spacing.small)
      }
      .scrollIndicators(.hidden)
    }
  }

  private var audioTrack: some View {
    TrackRow(label: "Audio", symbol: "waveform") {
      ScrollView(.horizontal) {
        HStack(spacing: 2) {
          ForEach(audioPlacements) { placement in
            if placement.gap > 0 {
              Color.clear
                .frame(width: placement.gap * pointsPerSecond)
            }
            TimelineItemView(
              title: model.project.asset(id: placement.item.assetID)?.displayName ?? "Audio",
              duration: placement.item.duration,
              isEnabled: placement.item.isEnabled,
              tint: Color(red: 0.40, green: 0.68, blue: 0.63),
              pointsPerSecond: pointsPerSecond
            )
            .contextMenu {
              Button(placement.item.isEnabled ? "Disable" : "Enable") {
                model.toggleAudio(placement.item.id)
              }
              Divider()
              Button("Remove", role: .destructive) {
                model.removeAudio(placement.item.id)
              }
            }
          }

          Button {
            model.presentGeneration(.audio)
          } label: {
            Label("Append audio", systemImage: "plus")
          }
          .buttonStyle(H3QuietButtonStyle())
          .fixedSize()
          .accessibilityIdentifier("append-audio")
        }
        .padding(.horizontal, H3Spacing.small)
      }
      .scrollIndicators(.hidden)
    }
  }

  private var audioPlacements: [AudioPlacement] {
    var cursor: TimeInterval = 0
    return model.project.timeline.audioItems
      .sorted { $0.startTime < $1.startTime }
      .map { item in
        let placement = AudioPlacement(item: item, gap: max(0, item.startTime - cursor))
        cursor = max(cursor, item.endTime)
        return placement
      }
  }

  private func durationLabel(_ duration: TimeInterval) -> String {
    let totalSeconds = max(0, Int(duration.rounded()))
    return String(format: "%02d:%02d", totalSeconds / 60, totalSeconds % 60)
  }
}

private struct AudioPlacement: Identifiable {
  var item: AudioItem
  var gap: TimeInterval
  var id: UUID { item.id }
}

private struct TrackRow<Content: View>: View {
  var label: String
  var symbol: String
  @ViewBuilder var content: Content

  var body: some View {
    HStack(spacing: 0) {
      Label(label, systemImage: symbol)
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(H3Color.textSecondary)
        .padding(.leading, H3Spacing.medium)
        .frame(width: 116, alignment: .leading)

      Divider().overlay(H3Color.line)
      content
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .frame(height: 72)
  }
}

private struct TimelineItemView: View {
  var title: String
  var duration: TimeInterval
  var isEnabled: Bool
  var tint: Color
  var pointsPerSecond: CGFloat

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      Text(title)
        .font(.system(size: 11, weight: .semibold))
        .lineLimit(1)
      Text(String(format: "%.1fs", duration))
        .font(.system(size: 10, weight: .medium, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
    }
    .padding(.horizontal, 10)
    .frame(width: max(86, duration * pointsPerSecond), height: 50, alignment: .leading)
    .background(tint.opacity(isEnabled ? 0.18 : 0.07))
    .overlay(alignment: .leading) {
      Rectangle()
        .fill(tint.opacity(isEnabled ? 1 : 0.35))
        .frame(width: 2)
    }
    .overlay {
      RoundedRectangle(cornerRadius: H3Radius.small, style: .continuous)
        .stroke(tint.opacity(isEnabled ? 0.45 : 0.18), lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: H3Radius.small, style: .continuous))
    .opacity(isEnabled ? 1 : 0.62)
    .animation(.easeOut(duration: 0.16), value: isEnabled)
  }
}
