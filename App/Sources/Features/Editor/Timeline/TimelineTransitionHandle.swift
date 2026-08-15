import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct TimelineTransitionHandle: View {
  var duration: TimeInterval
  var kind: VisualTransitionKind?
  var isSelected: Bool
  var metrics: TimelineMetrics
  var laneHeight: CGFloat
  var onSelect: () -> Void
  var onContextMenu: ((CGPoint) -> Void)?
  var onDurationChanged: ((CGFloat) -> Void)?
  var onDurationEnded: (() -> Void)?

  static let buttonSize: CGFloat = 28

  private var overlapWidth: CGFloat {
    kind == nil ? Self.buttonSize : max(Self.buttonSize, metrics.x(for: duration))
  }

  var body: some View {
    ZStack {
      if kind != nil {
        overlapBand
      }
      Button(action: onSelect) {
        Image(systemName: kind?.symbol ?? "plus")
          .font(.system(size: kind == nil ? 14 : 13, weight: .bold))
          .foregroundStyle(Color.white)
          .frame(width: Self.buttonSize, height: Self.buttonSize)
          .background(kind == nil ? H3Color.controlFill : H3Color.accent)
          .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
          .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
              .stroke(
                isSelected ? Color.white.opacity(0.85) : H3Color.line,
                lineWidth: isSelected ? 1.5 : 1
              )
          }
          .shadow(color: .black.opacity(0.45), radius: 4, y: 2)
      }
      .buttonStyle(.plain)
      .accessibilityIdentifier("timeline-transition")
      .accessibilityLabel(kind == nil ? "Add transition" : "Edit transition")
    }
    .frame(
      width: overlapWidth,
      height: kind == nil ? Self.buttonSize : laneHeight
    )
    .contentShape(Rectangle())
    .overlay {
      GeometryReader { proxy in
        SecondaryClickProbe { local in
          onContextMenu?(
            CGPoint(
              x: proxy.frame(in: .named("editor-root")).minX + local.x,
              y: proxy.frame(in: .named("editor-root")).minY + local.y
            )
          )
        }
      }
    }
    .gesture(
      kind == nil
        ? nil
        : DragGesture(minimumDistance: 2)
          .onChanged { value in
            onDurationChanged?(value.translation.width)
          }
          .onEnded { _ in
            onDurationEnded?()
          }
    )
    .onTapGesture(perform: onSelect)
  }

  private var overlapBand: some View {
    ZStack {
      RoundedRectangle(cornerRadius: 4, style: .continuous)
        .fill(H3Color.accent.opacity(0.22))
      DiagonalOverlapMark()
        .stroke(Color.white.opacity(0.28), lineWidth: 1)
      RoundedRectangle(cornerRadius: 4, style: .continuous)
        .stroke(H3Color.accent.opacity(isSelected ? 0.95 : 0.7), lineWidth: isSelected ? 1.5 : 1)
    }
    .frame(width: overlapWidth, height: laneHeight - 8)
  }
}

private struct DiagonalOverlapMark: Shape {
  func path(in rect: CGRect) -> Path {
    var path = Path()
    path.move(to: CGPoint(x: rect.minX + 4, y: rect.maxY - 4))
    path.addLine(to: CGPoint(x: rect.maxX - 4, y: rect.minY + 4))
    return path
  }
}
