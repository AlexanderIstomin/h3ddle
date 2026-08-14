import AppKit
import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct TimelineClipView: View {
  var title: String
  var kind: MediaKind
  var duration: TimeInterval
  var isEnabled: Bool
  var isTrackMuted: Bool = false
  var isSelected: Bool
  var metrics: TimelineMetrics
  var height: CGFloat
  var showsTrimHandles: Bool = false
  var onTrimChanged: ((TimelineTrimEdge, CGFloat) -> Void)?
  var onTrimEnded: (() -> Void)?

  private var clipWidth: CGFloat {
    max(8, metrics.x(for: duration) - 2)
  }

  var body: some View {
    ZStack(alignment: .bottomLeading) {
      clipBody
      if showsTrimHandles {
        trimHandle(.leading)
        trimHandle(.trailing)
      }
    }
    .frame(width: clipWidth, height: height)
    .zIndex(isSelected ? 4 : 0)
  }

  private var clipBody: some View {
    ZStack(alignment: .bottomLeading) {
      clipFill
      decoration
      Text(title)
        .font(.system(size: 9, weight: .semibold, design: .monospaced))
        .foregroundStyle(Color.white)
        .lineLimit(1)
        .shadow(color: .black.opacity(0.85), radius: 2, y: 1)
        .padding(.horizontal, 8)
        .padding(.bottom, 4)
    }
    .frame(width: clipWidth, height: height)
    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 4, style: .continuous)
        .stroke(isSelected ? H3Color.accent : Color.white.opacity(0.2), lineWidth: isSelected ? 1.5 : 1)
    }
    .overlay(alignment: .top) {
      Rectangle()
        .fill(isSelected ? H3Color.accent : accent)
        .frame(height: 2)
        .opacity(isEnabled ? 1 : 0)
        .overlay {
          if !isEnabled {
            Rectangle()
              .stroke(style: StrokeStyle(lineWidth: 2, dash: [4, 3]))
              .foregroundStyle(isSelected ? H3Color.accent : accent.opacity(0.7))
          }
        }
    }
    .overlay {
      if !isEnabled {
        TimelineDisabledHatch()
          .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
      } else if isTrackMuted {
        Color.black.opacity(0.36)
          .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
      }
    }
    .shadow(color: isSelected ? H3Color.accent.opacity(0.35) : .black.opacity(0.45), radius: isSelected ? 8 : 4, y: 3)
    .saturation(clipSaturation)
    .brightness(clipBrightness)
    .opacity(clipOpacity)
    .grayscale(isEnabled ? 0 : 0.68)
  }

  private var clipSaturation: Double {
    if !isEnabled { return 0.2 }
    if isTrackMuted { return 0.42 }
    return 1
  }

  private var clipBrightness: Double {
    if !isEnabled { return -0.18 }
    if isTrackMuted { return -0.12 }
    return 0
  }

  private var clipOpacity: Double {
    if !isEnabled { return 0.68 }
    if isTrackMuted { return 0.78 }
    return 1
  }

  private func trimHandle(_ edge: TimelineTrimEdge) -> some View {
    TimelineTrimHandle()
      .offset(x: edge == .leading ? -10 : clipWidth - 2)
      .highPriorityGesture(
        DragGesture(minimumDistance: 1, coordinateSpace: .global)
          .onChanged { value in
            onTrimChanged?(edge, value.translation.width)
          }
          .onEnded { _ in
            onTrimEnded?()
          }
      )
      .onHover { hovering in
        if hovering {
          NSCursor.resizeLeftRight.push()
        } else {
          NSCursor.pop()
        }
      }
      .accessibilityLabel(edge == .leading ? "Trim start" : "Trim end")
  }

  @ViewBuilder
  private var clipFill: some View {
    switch kind {
    case .audio:
      LinearGradient(
        colors: [
          Color(red: 44 / 255, green: 58 / 255, blue: 54 / 255),
          Color(red: 31 / 255, green: 41 / 255, blue: 37 / 255),
        ],
        startPoint: .top,
        endPoint: .bottom
      )
    case .video, .image:
      LinearGradient(
        colors: [
          Color(red: 51 / 255, green: 61 / 255, blue: 73 / 255),
          Color(red: 34 / 255, green: 42 / 255, blue: 51 / 255),
        ],
        startPoint: .top,
        endPoint: .bottom
      )
    }
  }

  @ViewBuilder
  private var decoration: some View {
    switch kind {
    case .video:
      filmSprockets
    case .image:
      stillDecoration
    case .audio:
      waveform
    }
  }

  private var filmSprockets: some View {
    ZStack {
      Rectangle()
        .fill(
          .linearGradient(
            colors: [Color.white.opacity(0.1), .clear],
            startPoint: .leading,
            endPoint: .trailing
          )
        )
        .mask(
          Rectangle()
            .stroke(style: StrokeStyle(lineWidth: 1, dash: [1, 17]))
        )
      VStack {
        sprocketRow
        Spacer()
        sprocketRow
      }
    }
  }

  private var sprocketRow: some View {
    Rectangle()
      .fill(Color.black.opacity(0.5))
      .frame(height: 4)
      .mask(
        Rectangle()
          .stroke(style: StrokeStyle(lineWidth: 4, dash: [3, 6]))
      )
  }

  private var stillDecoration: some View {
    ZStack(alignment: .topTrailing) {
      Color.white.opacity(0.06)
      Circle()
        .fill(Color.white.opacity(0.35))
        .frame(width: 10, height: 10)
        .padding(.top, 7)
        .padding(.trailing, 13)
    }
  }

  private var waveform: some View {
    HStack(spacing: 1) {
      ForEach(Array(bars.enumerated()), id: \.offset) { _, height in
        Capsule()
          .fill(H3Color.clipAudio.opacity(0.85))
          .frame(width: 2, height: max(4, 28 * height))
      }
    }
    .padding(.horizontal, 6)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }

  private var bars: [CGFloat] {
    let count = max(8, Int(metrics.x(for: duration) / 4))
    var seed = title.unicodeScalars.reduce(into: UInt64(2_166_136_261)) { partial, scalar in
      partial = partial &* 16_777_619 &+ UInt64(scalar.value)
    }
    return (0..<count).map { _ in
      seed = seed &* 1_664_525 &+ 1_013_904_223
      let value = Double((seed >> 16) & 255) / 255
      return CGFloat(0.18 + value * 0.82)
    }
  }

  private var accent: Color {
    kind == .audio ? H3Color.clipAudio : H3Color.clipVideo
  }
}

private struct TimelineDisabledHatch: View {
  var body: some View {
    Canvas { context, size in
      let spacing: CGFloat = 8
      var x: CGFloat = -size.height
      while x < size.width + size.height {
        var path = Path()
        path.move(to: CGPoint(x: x, y: size.height))
        path.addLine(to: CGPoint(x: x + size.height, y: 0))
        context.stroke(path, with: .color(.black.opacity(0.22)), lineWidth: 1)
        x += spacing
      }
    }
    .background(Color.black.opacity(0.46))
    .allowsHitTesting(false)
  }
}

private struct TimelineTrimHandle: View {
  var body: some View {
    ZStack {
      Color.clear
        .frame(width: 12, height: .infinity)
      Capsule()
        .fill(H3Color.accent.opacity(0.92))
        .frame(width: 2, height: 18)
        .shadow(color: .black.opacity(0.45), radius: 1, y: 1)
    }
    .frame(width: 12)
    .frame(maxHeight: .infinity)
    .padding(.vertical, -4)
    .contentShape(Rectangle())
  }
}
