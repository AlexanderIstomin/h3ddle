import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct CanvasGizmoOverlay: View {
  var layout: CanvasGizmoGeometry.Layout
  var isEnabled: Bool
  var onDuplicate: () -> Void
  var onToggleEnabled: () -> Void
  var onChromeChange: ([CGRect]) -> Void

  var body: some View {
    ZStack(alignment: .topLeading) {
      outline
      stem
      ForEach(Array(layout.corners.keys), id: \.self) { corner in
        if let point = layout.corners[corner] {
          handle(at: point)
        }
      }
      handle(at: layout.rotateHandle, circular: true)
      chrome
    }
    .allowsHitTesting(true)
  }

  private var outline: some View {
    Path { path in
      guard let first = layout.quad.first else { return }
      path.move(to: CGPoint(x: first.x, y: first.y))
      for point in layout.quad.dropFirst() {
        path.addLine(to: CGPoint(x: point.x, y: point.y))
      }
      path.closeSubpath()
    }
    .stroke(H3Color.accent, lineWidth: 1)
    .allowsHitTesting(false)
  }

  private var stem: some View {
    Path { path in
      path.move(to: CGPoint(x: layout.rotateBase.x, y: layout.rotateBase.y))
      path.addLine(to: CGPoint(x: layout.rotateHandle.x, y: layout.rotateHandle.y))
    }
    .stroke(H3Color.accent, lineWidth: 1)
    .allowsHitTesting(false)
  }

  private func handle(at point: (x: Double, y: Double), circular: Bool = false) -> some View {
    let size = CanvasGizmoGeometry.handleSize
    return Group {
      if circular {
        Circle()
          .fill(Color.white)
          .overlay(Circle().stroke(H3Color.accent, lineWidth: 1))
      } else {
        RoundedRectangle(cornerRadius: 1, style: .continuous)
          .fill(Color.white)
          .overlay(
            RoundedRectangle(cornerRadius: 1, style: .continuous)
              .stroke(H3Color.accent, lineWidth: 1)
          )
      }
    }
    .frame(width: size, height: size)
    .position(x: point.x, y: point.y)
    .allowsHitTesting(false)
  }

  private var chrome: some View {
    HStack(spacing: 4) {
      gizmoButton(
        symbol: "plus.square.on.square",
        label: "Duplicate",
        identifier: "gizmo-duplicate",
        action: onDuplicate
      )
      gizmoButton(
        symbol: isEnabled ? "power" : "power",
        label: isEnabled ? "Disable" : "Enable",
        identifier: "gizmo-toggle-enabled",
        action: onToggleEnabled
      )
    }
    .padding(4)
    .background(H3Color.surface.opacity(0.94))
    .overlay {
      RoundedRectangle(cornerRadius: 8, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    .background {
      GeometryReader { proxy in
        Color.clear.preference(
          key: GizmoChromeKey.self,
          value: [proxy.frame(in: .named("program-canvas"))]
        )
      }
    }
    .onPreferenceChange(GizmoChromeKey.self, perform: onChromeChange)
    .position(chromeCenter)
    .accessibilityElement(children: .contain)
  }

  private var chromeCenter: CGPoint {
    CGPoint(
      x: layout.aabb.x + layout.aabb.width / 2,
      y: max(18, layout.aabb.y - 22)
    )
  }

  private func gizmoButton(
    symbol: String,
    label: String,
    identifier: String,
    action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      Image(systemName: symbol)
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(H3Color.textPrimary)
        .frame(width: 24, height: 22)
        .background(H3Color.controlFill)
        .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
    }
    .buttonStyle(.plain)
    .help(label)
    .accessibilityLabel(label)
    .accessibilityIdentifier(identifier)
  }
}

private struct GizmoChromeKey: PreferenceKey {
  static let defaultValue: [CGRect] = []

  static func reduce(value: inout [CGRect], nextValue: () -> [CGRect]) {
    value.append(contentsOf: nextValue())
  }
}

extension CanvasCorner: Identifiable {
  public var id: String { rawValue }
}
