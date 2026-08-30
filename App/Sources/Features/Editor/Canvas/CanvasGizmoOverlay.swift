import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct CanvasGizmoOverlay: View {
  var layout: CanvasGizmoGeometry.Layout
  var badge: String? = nil

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
      if let badge {
        valueBadge(badge)
      }
    }
    .allowsHitTesting(false)
  }

  private func valueBadge(_ text: String) -> some View {
    let center = CanvasGizmoGeometry.centroid(of: layout)
    return Text(text)
      .font(.system(size: 11, weight: .semibold, design: .monospaced))
      .foregroundStyle(.white)
      .padding(.horizontal, 7)
      .padding(.vertical, 2)
      .background {
        RoundedRectangle(cornerRadius: 5, style: .continuous)
          .fill(Color(red: 20 / 255, green: 18 / 255, blue: 16 / 255).opacity(0.92))
          .overlay {
            RoundedRectangle(cornerRadius: 5, style: .continuous)
              .stroke(Color.white.opacity(0.14), lineWidth: 1)
          }
      }
      .position(x: center.x, y: center.y - 30)
      .allowsHitTesting(false)
      .accessibilityIdentifier("canvas-gizmo-badge")
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
    .stroke(H3Color.accent, lineWidth: 1.5)
    .allowsHitTesting(false)
  }

  private func handle(at point: (x: Double, y: Double), circular: Bool = false) -> some View {
    let size = circular ? CanvasGizmoGeometry.rotateDiscSize : CanvasGizmoGeometry.handleSize
    return Group {
      if circular {
        ZStack {
          Circle()
            .fill(H3Color.accent)
          Image(systemName: "arrow.clockwise")
            .font(.system(size: 9, weight: .bold))
            .foregroundStyle(Color.white)
            .offset(y: -0.6)
          Circle()
            .strokeBorder(Color.white, lineWidth: 1.75)
        }
        .clipShape(Circle())
      } else {
        RoundedRectangle(cornerRadius: 2, style: .continuous)
          .fill(Color.white)
          .overlay(
            RoundedRectangle(cornerRadius: 2, style: .continuous)
              .stroke(H3Color.accent, lineWidth: 1.5)
          )
      }
    }
    .frame(width: size, height: size)
    .position(x: point.x, y: point.y)
    .allowsHitTesting(false)
  }
}

extension CanvasCorner: @retroactive Identifiable {
  public var id: String { rawValue }
}
