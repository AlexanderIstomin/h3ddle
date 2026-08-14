import H3ddleDesignSystem
import H3ddleGeneration
import SwiftUI

struct AppendMenuPlacement: Equatable {
  var isVisual: Bool
  var origin: CGPoint
}

enum TimelineAppendAction: Equatable {
  case generate(GenerationKind)
  case importFiles
}

struct TimelineAppendMenuItem: Identifiable {
  var id: String { label }
  var label: String
  var symbol: String
  var tint: Color
  var action: TimelineAppendAction
}

struct TimelineAppendMenu: View {
  var trackName: String
  var appendTime: TimeInterval
  var items: [TimelineAppendMenuItem]
  var onSelect: (TimelineAppendMenuItem) -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      HStack(spacing: 8) {
        Image(systemName: "plus.circle")
          .font(.system(size: 13, weight: .medium))
          .foregroundStyle(H3Color.accent)
        Text("Append to \(trackName)")
          .font(.system(size: 8, weight: .semibold, design: .monospaced))
          .tracking(1.2)
          .textCase(.uppercase)
          .foregroundStyle(H3Color.textSecondary)
          .lineLimit(1)
        Spacer(minLength: 8)
        Text(Self.formatTime(appendTime))
          .font(.system(size: 8, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary.opacity(0.8))
      }
      .padding(.horizontal, 8)
      .padding(.top, 5)
      .padding(.bottom, 8)

      Rectangle()
        .fill(H3Color.line)
        .frame(height: 1)
        .padding(.bottom, 3)

      ForEach(items) { item in
        AppendMenuRow(item: item) {
          onSelect(item)
        }
      }
    }
    .padding(6)
    .frame(width: 208, alignment: .leading)
    .background(H3Color.surface)
    .overlay {
      RoundedRectangle(cornerRadius: 11, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 11, style: .continuous))
    .shadow(color: .black.opacity(0.45), radius: 24, y: 14)
  }

  static func formatTime(_ time: TimeInterval) -> String {
    let clamped = max(0, time)
    let minutes = Int(clamped / 60)
    let seconds = clamped - Double(minutes * 60)
    return String(format: "%d:%04.1f", minutes, seconds)
  }

  static func visualItems() -> [TimelineAppendMenuItem] {
    [
      TimelineAppendMenuItem(
        label: "Video",
        symbol: "film",
        tint: H3Color.clipVideo,
        action: .generate(.video)
      ),
      TimelineAppendMenuItem(
        label: "Image",
        symbol: "photo",
        tint: Color(red: 210 / 255, green: 162 / 255, blue: 78 / 255),
        action: .generate(.image)
      ),
      TimelineAppendMenuItem(
        label: "Import…",
        symbol: "square.and.arrow.down",
        tint: H3Color.textSecondary,
        action: .importFiles
      ),
    ]
  }

  static func audioItems() -> [TimelineAppendMenuItem] {
    [
      TimelineAppendMenuItem(
        label: "Generate",
        symbol: "wand.and.stars",
        tint: H3Color.accent,
        action: .generate(.audio)
      ),
      TimelineAppendMenuItem(
        label: "Import…",
        symbol: "square.and.arrow.down",
        tint: H3Color.textSecondary,
        action: .importFiles
      ),
    ]
  }
}

private struct AppendMenuRow: View {
  var item: TimelineAppendMenuItem
  var action: () -> Void
  @State private var isHovered = false

  var body: some View {
    Button(action: action) {
      HStack(spacing: 9) {
        Image(systemName: item.symbol)
          .font(.system(size: 12, weight: .semibold))
          .foregroundStyle(item.tint)
          .frame(width: 22, height: 22)
          .background(item.tint.opacity(0.13))
          .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        Text(item.label)
          .font(.system(size: 12, weight: .medium))
          .foregroundStyle(H3Color.textPrimary)
          .accessibilityIdentifier(item.accessibilityIdentifier)
        Spacer(minLength: 0)
      }
      .padding(.horizontal, 8)
      .padding(.vertical, 7)
      .contentShape(Rectangle())
      .background(
        RoundedRectangle(cornerRadius: 7, style: .continuous)
          .fill(isHovered ? H3Color.controlHover : Color.clear)
      )
    }
    .buttonStyle(.plain)
    .onHover { isHovered = $0 }
    .accessibilityIdentifier(item.accessibilityIdentifier)
  }
}

extension TimelineAppendMenuItem {
  var accessibilityIdentifier: String {
    switch action {
    case .generate(let kind): "append-generate-\(kind.rawValue)"
    case .importFiles: "append-import"
    }
  }
}
