import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct ClipMenuPlacement: Equatable {
  enum Target: Equatable {
    case visual(UUID)
    case audio(UUID)
  }

  var target: Target
  var origin: CGPoint
}

struct TimelineClipMenu: View {
  var title: String
  var items: [TimelineClipMenuItem]
  var onSelect: (TimelineClipMenuItem) -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      Text(title)
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .tracking(1.1)
        .foregroundStyle(H3Color.textPrimary.opacity(0.38))
        .lineLimit(1)
        .padding(.horizontal, 10)
        .padding(.top, 6)
        .padding(.bottom, 8)
        .frame(maxWidth: .infinity, alignment: .leading)

      Rectangle()
        .fill(H3Color.line)
        .frame(height: 1)
        .padding(.bottom, 4)

      ForEach(items) { item in
        if item.isSeparator {
          Rectangle()
            .fill(H3Color.line)
            .frame(height: 1)
            .padding(.horizontal, 6)
            .padding(.vertical, 5)
        } else {
          ClipMenuRow(item: item) {
            onSelect(item)
          }
        }
      }
    }
    .padding(5)
    .frame(width: 208, alignment: .leading)
    .background(H3Color.surface)
    .overlay {
      RoundedRectangle(cornerRadius: 10, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    .shadow(color: .black.opacity(0.7), radius: 18, y: 14)
    .accessibilityElement(children: .contain)
    .accessibilityIdentifier("timeline-clip-menu")
  }

  static func visualItems(
    item: VisualItem,
    kind: MediaKind,
    canSplit: Bool
  ) -> [TimelineClipMenuItem] {
    var rows: [TimelineClipMenuItem] = [
      TimelineClipMenuItem(
        id: "duplicate",
        label: "Duplicate",
        symbol: "plus.square.on.square",
        action: .duplicate
      ),
      TimelineClipMenuItem(
        id: "enable",
        label: item.isEnabled ? "Disable" : "Enable",
        symbol: "power",
        action: .toggleEnabled
      ),
    ]
    if kind == .video {
      rows.append(
        TimelineClipMenuItem(
          id: "native-audio",
          label: item.includesNativeAudio ? "Mute native audio" : "Include native audio",
          symbol: item.includesNativeAudio ? "speaker.slash" : "speaker.wave.2",
          action: .toggleNativeAudio
        )
      )
    }
    rows.append(
      TimelineClipMenuItem(
        id: "split",
        label: "Split at playhead",
        symbol: "scissors",
        action: .split,
        isEnabled: canSplit
      )
    )
    rows.append(.separator("canvas"))
    rows.append(
      TimelineClipMenuItem(
        id: "cover",
        label: "Cover canvas",
        symbol: "arrow.up.left.and.arrow.down.right",
        action: .coverCanvas,
        isSelected: item.canvasFit == .cover
      )
    )
    rows.append(
      TimelineClipMenuItem(
        id: "fit",
        label: "Fit to canvas",
        symbol: "arrow.down.right.and.arrow.up.left",
        action: .fitToCanvas,
        isSelected: item.canvasFit == .fit
      )
    )
    rows.append(
      TimelineClipMenuItem(
        id: "rotate",
        label: "Rotate",
        symbol: "rotate.right",
        action: .rotate
      )
    )
    rows.append(.separator("remove"))
    rows.append(
      TimelineClipMenuItem(
        id: "remove",
        label: "Remove",
        symbol: "trash",
        action: .remove,
        isDestructive: true
      )
    )
    return rows
  }

  static func audioItems(item: AudioItem, canSplit: Bool) -> [TimelineClipMenuItem] {
    [
      TimelineClipMenuItem(
        id: "duplicate",
        label: "Duplicate",
        symbol: "plus.square.on.square",
        action: .duplicate
      ),
      TimelineClipMenuItem(
        id: "enable",
        label: item.isEnabled ? "Disable" : "Enable",
        symbol: "power",
        action: .toggleEnabled
      ),
      TimelineClipMenuItem(
        id: "split",
        label: "Split at playhead",
        symbol: "scissors",
        action: .split,
        isEnabled: canSplit
      ),
      .separator("remove"),
      TimelineClipMenuItem(
        id: "remove",
        label: "Remove",
        symbol: "trash",
        action: .remove,
        isDestructive: true
      ),
    ]
  }
}

enum TimelineClipMenuAction: Equatable {
  case duplicate
  case toggleEnabled
  case toggleNativeAudio
  case split
  case coverCanvas
  case fitToCanvas
  case rotate
  case remove
}

struct TimelineClipMenuItem: Identifiable, Equatable {
  var id: String
  var label: String
  var symbol: String
  var action: TimelineClipMenuAction?
  var isEnabled: Bool
  var isDestructive: Bool
  var isSelected: Bool
  var isSeparator: Bool

  init(
    id: String,
    label: String,
    symbol: String,
    action: TimelineClipMenuAction,
    isEnabled: Bool = true,
    isDestructive: Bool = false,
    isSelected: Bool = false
  ) {
    self.id = id
    self.label = label
    self.symbol = symbol
    self.action = action
    self.isEnabled = isEnabled
    self.isDestructive = isDestructive
    self.isSelected = isSelected
    self.isSeparator = false
  }

  static func separator(_ id: String) -> TimelineClipMenuItem {
    TimelineClipMenuItem(separatorID: id)
  }

  private init(separatorID: String) {
    id = "separator-\(separatorID)"
    label = ""
    symbol = ""
    action = nil
    isEnabled = false
    isDestructive = false
    isSelected = false
    isSeparator = true
  }
}

private struct ClipMenuRow: View {
  var item: TimelineClipMenuItem
  var action: () -> Void
  @State private var isHovered = false

  var body: some View {
    Button(action: action) {
      HStack(spacing: 10) {
        Image(systemName: item.symbol)
          .font(.system(size: 14, weight: .medium))
          .foregroundStyle(foreground)
          .opacity(0.82)
          .frame(width: 14, height: 14)
        Text(item.label)
          .font(.system(size: 12.5, weight: .regular))
          .foregroundStyle(foreground)
          .lineLimit(1)
        Spacer(minLength: 0)
        if item.isSelected {
          Image(systemName: "checkmark")
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(foreground.opacity(0.7))
        }
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 8)
      .frame(maxWidth: .infinity, alignment: .leading)
      .contentShape(Rectangle())
      .background(
        RoundedRectangle(cornerRadius: 6, style: .continuous)
          .fill(rowFill)
      )
    }
    .buttonStyle(.plain)
    .disabled(!item.isEnabled)
    .opacity(item.isEnabled ? 1 : 0.42)
    .onHover { isHovered = $0 }
    .accessibilityIdentifier("clip-menu-\(item.id)")
    .accessibilityAddTraits(item.isSelected ? .isSelected : [])
  }

  private var foreground: Color {
    item.isDestructive ? H3Color.danger : H3Color.textPrimary
  }

  private var rowFill: Color {
    guard isHovered, item.isEnabled else { return .clear }
    return item.isDestructive ? H3Color.danger.opacity(0.14) : Color.white.opacity(0.11)
  }
}
