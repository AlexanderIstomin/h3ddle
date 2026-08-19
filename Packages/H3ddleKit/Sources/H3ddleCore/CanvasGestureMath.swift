import Foundation

public enum CanvasCorner: String, Sendable, CaseIterable {
  case topLeft
  case topRight
  case bottomRight
  case bottomLeft

  public var opposite: CanvasCorner {
    switch self {
    case .topLeft: .bottomRight
    case .topRight: .bottomLeft
    case .bottomRight: .topLeft
    case .bottomLeft: .topRight
    }
  }
}

public enum CanvasViewportMath: Sendable {
  /// Program `(0, 0)` is the monitor bottom-left, y-up. Translation `(0, 0)`
  /// is canvas center (`0.5, 0.5`). Returns `nil` when the unscaled point
  /// sits outside the aspect-fitted monitor (pad and letterbox are empty canvas).
  public static func programPoint(
    viewPoint: (x: Double, y: Double),
    viewSize: (width: Double, height: Double),
    aspect: Double,
    padding: Double,
    magnification: Double,
    offset: (x: Double, y: Double)
  ) -> (x: Double, y: Double)? {
    let monitor = monitorRect(viewSize: viewSize, aspect: aspect, padding: padding)
    let unscaled = unscaledViewPoint(
      viewPoint: viewPoint,
      viewSize: viewSize,
      magnification: magnification,
      offset: offset
    )
    guard contains(unscaled, in: monitor) else { return nil }
    guard monitor.width > 0, monitor.height > 0 else { return nil }
    return normalizedProgramPoint(unscaled, in: monitor)
  }

  /// Converts a view point without rejecting coordinates outside the monitor.
  /// This keeps an active canvas gesture responsive when a handle or pointer
  /// crosses the artwork boundary.
  public static func unboundedProgramPoint(
    viewPoint: (x: Double, y: Double),
    viewSize: (width: Double, height: Double),
    aspect: Double,
    padding: Double,
    magnification: Double,
    offset: (x: Double, y: Double)
  ) -> (x: Double, y: Double)? {
    let monitor = monitorRect(viewSize: viewSize, aspect: aspect, padding: padding)
    guard monitor.width > 0, monitor.height > 0 else { return nil }
    let unscaled = unscaledViewPoint(
      viewPoint: viewPoint,
      viewSize: viewSize,
      magnification: magnification,
      offset: offset
    )
    return normalizedProgramPoint(unscaled, in: monitor)
  }

  private static func normalizedProgramPoint(
    _ point: (x: Double, y: Double),
    in monitor: CanvasLayout.Rect
  ) -> (x: Double, y: Double) {
    return (
      (point.x - monitor.x) / monitor.width,
      1 - (point.y - monitor.y) / monitor.height
    )
  }

  public static func viewPoint(
    program: (x: Double, y: Double),
    viewSize: (width: Double, height: Double),
    aspect: Double,
    padding: Double,
    magnification: Double,
    offset: (x: Double, y: Double)
  ) -> (x: Double, y: Double) {
    let monitor = monitorRect(viewSize: viewSize, aspect: aspect, padding: padding)
    let unscaled = (
      x: monitor.x + program.x * monitor.width,
      y: monitor.y + (1 - program.y) * monitor.height
    )
    let center = (x: viewSize.width / 2, y: viewSize.height / 2)
    let mag = max(magnification, 0.000_1)
    return (
      center.x + (unscaled.x - center.x) * mag + offset.x,
      center.y + (unscaled.y - center.y) * mag + offset.y
    )
  }

  public static func monitorRect(
    viewSize: (width: Double, height: Double),
    aspect: Double,
    padding: Double
  ) -> CanvasLayout.Rect {
    let available = (
      width: max(0, viewSize.width - 2 * padding),
      height: max(0, viewSize.height - 2 * padding)
    )
    let fitted = aspectFit(available, aspect: aspect)
    let padded = (width: fitted.width + 2 * padding, height: fitted.height + 2 * padding)
    let originX = viewSize.width / 2 - padded.width / 2 + padding
    let originY = viewSize.height / 2 - padded.height / 2 + padding
    return CanvasLayout.Rect(
      x: originX,
      y: originY,
      width: fitted.width,
      height: fitted.height
    )
  }

  private static func unscaledViewPoint(
    viewPoint: (x: Double, y: Double),
    viewSize: (width: Double, height: Double),
    magnification: Double,
    offset: (x: Double, y: Double)
  ) -> (x: Double, y: Double) {
    let center = (x: viewSize.width / 2, y: viewSize.height / 2)
    let p1 = (x: viewPoint.x - offset.x, y: viewPoint.y - offset.y)
    let mag = max(magnification, 0.000_1)
    return (
      center.x + (p1.x - center.x) / mag,
      center.y + (p1.y - center.y) / mag
    )
  }

  private static func aspectFit(
    _ available: (width: Double, height: Double),
    aspect: Double
  ) -> (width: Double, height: Double) {
    let aspect = max(aspect, 0.000_1)
    guard available.width > 0, available.height > 0 else {
      return (0, 0)
    }
    if available.width / available.height > aspect {
      let height = available.height
      return (height * aspect, height)
    }
    let width = available.width
    return (width, width / aspect)
  }

  private static func contains(
    _ point: (x: Double, y: Double),
    in rect: CanvasLayout.Rect
  ) -> Bool {
    point.x >= rect.x - 1e-9
      && point.x <= rect.maxX + 1e-9
      && point.y >= rect.y - 1e-9
      && point.y <= rect.maxY + 1e-9
  }
}

public enum CanvasGestureMath: Sendable {
  public static let minimumScale = 0.05
  public static let rotationSnap = Double.pi / 12

  public static func moved(
    origin: CanvasObjectTransform,
    deltaProgramX: Double,
    deltaProgramY: Double
  ) -> CanvasObjectTransform {
    var next = origin
    next.translationX += deltaProgramX
    next.translationY += deltaProgramY
    return next
  }

  public static func scaled(
    origin: CanvasObjectTransform,
    grab: CanvasCorner,
    pointer: (x: Double, y: Double),
    canvasWidth: Double,
    canvasHeight: Double,
    sourceWidth: Double,
    sourceHeight: Double,
    aboutCenter: Bool
  ) -> CanvasObjectTransform {
    let placement = CanvasLayout.placed(
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight,
      transform: origin
    )
    let grabPoint = corner(
      grab,
      of: placement,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight
    )
    let pivot: (x: Double, y: Double)
    if aboutCenter {
      pivot = (placement.centerX, placement.centerY)
    } else {
      pivot = corner(
        grab.opposite,
        of: placement,
        canvasWidth: canvasWidth,
        canvasHeight: canvasHeight
      )
    }
    let pointerPixels = (
      x: pointer.x * canvasWidth,
      y: pointer.y * canvasHeight
    )
    let axis = (x: grabPoint.x - pivot.x, y: grabPoint.y - pivot.y)
    let axisLength2 = axis.x * axis.x + axis.y * axis.y
    guard axisLength2 > 1e-12 else { return origin }
    let projected =
      ((pointerPixels.x - pivot.x) * axis.x + (pointerPixels.y - pivot.y) * axis.y)
      / axisLength2
    var next = origin
    next.scale = max(origin.scale * projected, minimumScale)
    if aboutCenter { return next }

    let nextPlacement = CanvasLayout.placed(
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight,
      transform: next
    )
    let nextLocal = corner(
      grab.opposite,
      of: nextPlacement,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight
    )
    let newCenter = (
      x: pivot.x - (nextLocal.x - nextPlacement.centerX),
      y: pivot.y - (nextLocal.y - nextPlacement.centerY)
    )
    guard canvasWidth > 0, canvasHeight > 0 else { return next }
    next.translationX = (newCenter.x - canvasWidth / 2) / canvasWidth
    next.translationY = (newCenter.y - canvasHeight / 2) / canvasHeight
    return next
  }

  public static func rotated(
    origin: CanvasObjectTransform,
    start: (x: Double, y: Double),
    pointer: (x: Double, y: Double),
    canvasWidth: Double,
    canvasHeight: Double,
    sourceWidth: Double,
    sourceHeight: Double,
    snapToIncrements: Bool
  ) -> CanvasObjectTransform {
    let placement = CanvasLayout.placed(
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight,
      transform: origin
    )
    let startPixels = (x: start.x * canvasWidth, y: start.y * canvasHeight)
    let pointerPixels = (x: pointer.x * canvasWidth, y: pointer.y * canvasHeight)
    let startVector = (
      x: startPixels.x - placement.centerX,
      y: startPixels.y - placement.centerY
    )
    let pointerVector = (
      x: pointerPixels.x - placement.centerX,
      y: pointerPixels.y - placement.centerY
    )
    let startLength = hypot(startVector.x, startVector.y)
    let pointerLength = hypot(pointerVector.x, pointerVector.y)
    guard startLength > 1e-9, pointerLength > 1e-9 else { return origin }
    // y-up atan2 is counterclockwise from +x. Stored angle is clockwise.
    let cross = startVector.x * pointerVector.y - startVector.y * pointerVector.x
    let dot = startVector.x * pointerVector.x + startVector.y * pointerVector.y
    var radians = origin.rotationRadians - atan2(cross, dot)
    if snapToIncrements {
      radians = (radians / rotationSnap).rounded() * rotationSnap
    }
    var next = origin
    next.rotationRadians = radians
    return next
  }

  public static func contains(
    point: (x: Double, y: Double),
    quad: [(x: Double, y: Double)],
    tolerance: Double = 0
  ) -> Bool {
    guard quad.count == 4 else { return false }
    var sign = 0.0
    for index in quad.indices {
      let a = quad[index]
      let b = quad[(index + 1) % quad.count]
      let cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
      if abs(cross) <= tolerance { continue }
      let next = cross > 0 ? 1.0 : -1.0
      if sign == 0 {
        sign = next
      } else if next != sign {
        return false
      }
    }
    return true
  }

  private static func corner(
    _ corner: CanvasCorner,
    of placement: CanvasLayout.Placement,
    canvasWidth: Double,
    canvasHeight: Double
  ) -> (x: Double, y: Double) {
    _ = canvasWidth
    _ = canvasHeight
    let local: (x: Double, y: Double)
    switch corner {
    case .bottomLeft:
      local = (-placement.drawWidth / 2, -placement.drawHeight / 2)
    case .bottomRight:
      local = (placement.drawWidth / 2, -placement.drawHeight / 2)
    case .topRight:
      local = (placement.drawWidth / 2, placement.drawHeight / 2)
    case .topLeft:
      local = (-placement.drawWidth / 2, placement.drawHeight / 2)
    }
    let rotated = CanvasLayout.rotateClockwise(
      x: local.x,
      y: local.y,
      radians: placement.rotationRadians
    )
    return (placement.centerX + rotated.x, placement.centerY + rotated.y)
  }
}
