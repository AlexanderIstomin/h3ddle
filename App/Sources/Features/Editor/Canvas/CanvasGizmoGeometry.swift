import CoreGraphics
import Foundation
import H3ddleCore

enum CanvasGizmoGeometry {
  static let padding = 32.0
  static let handleSize = 10.0
  static let cornerHitSize = 40.0
  static let rotateDiscSize = 20.0
  static let rotateHandleHit = 11.0
  static let rotateStem = 28.0

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

  static func centroid(of layout: Layout) -> (x: Double, y: Double) {
    let count = Double(max(layout.quad.count, 1))
    return (
      layout.quad.reduce(0) { $0 + $1.x } / count,
      layout.quad.reduce(0) { $0 + $1.y } / count
    )
  }

  static func hitCorner(
    at viewPoint: (x: Double, y: Double),
    in layout: Layout
  ) -> CanvasCorner? {
    let center = centroid(of: layout)
    let half = cornerHitSize / 2
    let hits = layout.corners.filter { _, corner in
      let placed = CanvasGestureMath.outsetHandle(
        point: corner,
        centroid: center,
        hitSize: (cornerHitSize, cornerHitSize),
        shapeSize: (handleSize, handleSize)
      )
      return abs(viewPoint.x - placed.hitX) <= half && abs(viewPoint.y - placed.hitY) <= half
    }
    return hits.min(by: { lhs, rhs in
      hypot(lhs.value.x - viewPoint.x, lhs.value.y - viewPoint.y)
        < hypot(rhs.value.x - viewPoint.x, rhs.value.y - viewPoint.y)
    })?.key
  }

  static func cornerIntent(
    at viewPoint: (x: Double, y: Double),
    corner: CanvasCorner,
    in layout: Layout
  ) -> CanvasHandleIntent {
    guard let point = layout.corners[corner] else { return .scale }
    return CanvasGestureMath.cornerIntent(
      pointer: viewPoint,
      corner: point,
      centroid: centroid(of: layout)
    )
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
