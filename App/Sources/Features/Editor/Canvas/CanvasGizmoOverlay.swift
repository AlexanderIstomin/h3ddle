import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct CanvasGizmoOverlay: View {
  var layout: CanvasGizmoGeometry.Layout

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
    }
    .allowsHitTesting(false)
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
}

extension CanvasCorner: Identifiable {
  public var id: String { rawValue }
}
