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

  private var controlSize: CGFloat {
    kind == nil ? Self.buttonSize : 22
  }

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
          .font(.system(size: kind == nil ? 14 : 11, weight: .bold))
          .foregroundStyle(Color.white)
          .frame(width: controlSize, height: controlSize)
          .background(kind == nil ? H3Color.controlFill : Color.black.opacity(0.78))
          .clipShape(RoundedRectangle(cornerRadius: kind == nil ? 8 : 6, style: .continuous))
          .overlay {
            RoundedRectangle(cornerRadius: kind == nil ? 8 : 6, style: .continuous)
              .stroke(
                isSelected
                  ? Color.white.opacity(0.95)
                  : (kind == nil ? H3Color.line : H3Color.accent.opacity(0.95)),
                lineWidth: isSelected ? 1.5 : 1
              )
          }
          .shadow(color: .black.opacity(0.45), radius: 4, y: 2)
      }
      .buttonStyle(.plain)
      .offset(
        x: kind == nil ? 0 : -overlapWidth / 2 + controlSize / 2 + 5,
        y: kind == nil ? 0 : -laneHeight / 2 + controlSize / 2 + 5
      )
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
        .fill(
          LinearGradient(
            colors: [H3Color.accent.opacity(0.52), H3Color.accent.opacity(0.25)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
          )
        )
      TransitionOverlapMark()
        .stroke(Color.white.opacity(0.58), lineWidth: 1)
        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
      HStack(spacing: 0) {
        Rectangle().fill(Color.white.opacity(0.9)).frame(width: 2)
        Spacer(minLength: 0)
        Rectangle().fill(Color.white.opacity(0.9)).frame(width: 2)
      }
      if overlapWidth >= 68, let kind {
        Text(kind.label.uppercased())
          .font(.system(size: 7, weight: .bold, design: .monospaced))
          .tracking(0.6)
          .foregroundStyle(Color.white.opacity(0.9))
          .lineLimit(1)
          .padding(.horizontal, 5)
          .padding(.bottom, 4)
          .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
      }
      RoundedRectangle(cornerRadius: 4, style: .continuous)
        .stroke(
          isSelected ? Color.white.opacity(0.95) : H3Color.accent,
          lineWidth: isSelected ? 2 : 1.5
        )
    }
    .frame(width: overlapWidth, height: laneHeight - 2)
    .shadow(color: H3Color.accent.opacity(isSelected ? 0.6 : 0.35), radius: 3)
  }
}

private struct TransitionOverlapMark: Shape {
  func path(in rect: CGRect) -> Path {
    var path = Path()
    var x = rect.minX - rect.height
    while x <= rect.maxX {
      path.move(to: CGPoint(x: x, y: rect.maxY))
      path.addLine(to: CGPoint(x: x + rect.height, y: rect.minY))
      x += 10
    }
    path.move(to: CGPoint(x: rect.minX + 2, y: rect.minY + 2))
    path.addLine(to: CGPoint(x: rect.maxX - 2, y: rect.maxY - 2))
    return path
  }
}
