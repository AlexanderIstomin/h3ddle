import CoreGraphics
import Foundation
import H3ddleCore

enum CanvasGizmoGeometry {
  static let padding = 32.0
  static let handleSize = 8.0
  static let cornerHandleHit: Double = 12
  static let rotateHandleHit: Double = 14
  static let rotateStem: Double = 22

  struct Layout {
    var quad: [(x: Double, y: Double)]
    var corners: [CanvasCorner: (x: Double, y: Double)]
    var rotateHandle: (x: Double, y: Double)
    var rotateBase: (x: Double, y: Double)
    var aabb: (x: Double, y: Double, width: Double, height: Double)
  }

  static func placement(
    item: VisualItem,
    source: (width: Double, height: Double),
    canvas: (width: Double, height: Double),
    override: CanvasObjectTransform? = nil
  ) -> CanvasLayout.Placement {
    CanvasLayout.placed(
      sourceWidth: source.width,
      sourceHeight: source.height,
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      transform: override ?? item.canvasTransform
    )
  }

  static func layout(
    placement: CanvasLayout.Placement,
    canvas: (width: Double, height: Double),
    viewSize: (width: Double, height: Double),
    aspect: Double,
    magnification: Double,
    offset: (x: Double, y: Double)
  ) -> Layout {
    let programQuad = placement.quad.map { point in
      (
        x: point.x / max(canvas.width, 1),
        y: point.y / max(canvas.height, 1)
      )
    }
    let viewQuad = programQuad.map { point in
      CanvasViewportMath.viewPoint(
        program: point,
        viewSize: viewSize,
        aspect: aspect,
        padding: padding,
        magnification: magnification,
        offset: offset
      )
    }
    let corners: [CanvasCorner: (x: Double, y: Double)] = [
      .bottomLeft: viewQuad[0],
      .bottomRight: viewQuad[1],
      .topRight: viewQuad[2],
      .topLeft: viewQuad[3],
    ]
    let topMidProgram = (
      x: (programQuad[2].x + programQuad[3].x) / 2,
      y: (programQuad[2].y + programQuad[3].y) / 2
    )
    let centerProgram = (
      x: placement.centerX / max(canvas.width, 1),
      y: placement.centerY / max(canvas.height, 1)
    )
    let stem = outward(
      from: centerProgram,
      through: topMidProgram,
      viewDistance: rotateStem,
      viewSize: viewSize,
      aspect: aspect,
      magnification: magnification,
      offset: offset
    )
    let xs = viewQuad.map(\.x)
    let ys = viewQuad.map(\.y)
    let minX = xs.min() ?? 0
    let maxX = xs.max() ?? 0
    let minY = ys.min() ?? 0
    let maxY = ys.max() ?? 0
    return Layout(
      quad: viewQuad,
      corners: corners,
      rotateHandle: stem.handle,
      rotateBase: stem.base,
      aabb: (minX, minY, maxX - minX, maxY - minY)
    )
  }

  static func hitCorner(
    at viewPoint: (x: Double, y: Double),
    in layout: Layout
  ) -> CanvasCorner? {
    layout.corners.min(by: { lhs, rhs in
      hypot(lhs.value.x - viewPoint.x, lhs.value.y - viewPoint.y)
        < hypot(rhs.value.x - viewPoint.x, rhs.value.y - viewPoint.y)
    }).flatMap { candidate in
      hypot(candidate.value.x - viewPoint.x, candidate.value.y - viewPoint.y) <= cornerHandleHit
        ? candidate.key
        : nil
    }
  }

  static func hitsRotate(at viewPoint: (x: Double, y: Double), in layout: Layout) -> Bool {
    hypot(layout.rotateHandle.x - viewPoint.x, layout.rotateHandle.y - viewPoint.y)
      <= rotateHandleHit
  }

  static func contains(
    program: (x: Double, y: Double),
    placement: CanvasLayout.Placement,
    canvas: (width: Double, height: Double),
    tolerance: Double
  ) -> Bool {
    let point = (program.x * canvas.width, program.y * canvas.height)
    return CanvasGestureMath.contains(
      point: point,
      quad: placement.quad,
      tolerance: tolerance
    )
  }

  private static func outward(
    from center: (x: Double, y: Double),
    through edge: (x: Double, y: Double),
    viewDistance: Double,
    viewSize: (width: Double, height: Double),
    aspect: Double,
    magnification: Double,
    offset: (x: Double, y: Double)
  ) -> (base: (x: Double, y: Double), handle: (x: Double, y: Double)) {
    let base = CanvasViewportMath.viewPoint(
      program: edge,
      viewSize: viewSize,
      aspect: aspect,
      padding: padding,
      magnification: magnification,
      offset: offset
    )
    let centerView = CanvasViewportMath.viewPoint(
      program: center,
      viewSize: viewSize,
      aspect: aspect,
      padding: padding,
      magnification: magnification,
      offset: offset
    )
    let axis = (x: base.x - centerView.x, y: base.y - centerView.y)
    let length = hypot(axis.x, axis.y)
    guard length > 0.001 else {
      return (base, (base.x, base.y - viewDistance))
    }
    let handle = (
      x: base.x + axis.x / length * viewDistance,
      y: base.y + axis.y / length * viewDistance
    )
    return (base, handle)
  }
}

extension CanvasLayout.Placement {
  var quad: [(x: Double, y: Double)] {
    CanvasLayout.projectedQuad(
      centerX: centerX,
      centerY: centerY,
      drawWidth: drawWidth,
      drawHeight: drawHeight,
      rotationRadians: rotationRadians
    )
  }
}
