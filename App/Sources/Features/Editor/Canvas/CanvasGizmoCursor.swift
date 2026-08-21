import AppKit
import H3ddleCore

enum CanvasGizmoCursor {
  private static let cache = Cache()

  static func scale(radians: Double) -> NSCursor {
    let key = quantize(radians)
    return cache.scale(key) {
      make(kind: .scale, degrees: Double(key))
    }
  }

  static func rotate(radians: Double) -> NSCursor {
    let key = quantize(radians)
    return cache.rotate(key) {
      make(kind: .rotate, degrees: Double(key))
    }
  }

  /// Rotate glyph is perpendicular to the handle radial (view y-down `atan2`).
  static func rotateRadians(radial: Double) -> Double {
    radial + .pi / 2
  }

  private enum Kind {
    case scale
    case rotate
  }

  /// Pointer-thread cache; AppKit cursor creation is serialized by the lock.
  private final class Cache: @unchecked Sendable {
    private let lock = NSLock()
    private var scaleCursors: [Int: NSCursor] = [:]
    private var rotateCursors: [Int: NSCursor] = [:]

    func scale(_ key: Int, make: () -> NSCursor) -> NSCursor {
      store(key, in: &scaleCursors, make: make)
    }

    func rotate(_ key: Int, make: () -> NSCursor) -> NSCursor {
      store(key, in: &rotateCursors, make: make)
    }

    private func store(
      _ key: Int,
      in table: inout [Int: NSCursor],
      make: () -> NSCursor
    ) -> NSCursor {
      lock.lock()
      defer { lock.unlock() }
      if let existing = table[key] { return existing }
      let cursor = make()
      table[key] = cursor
      return cursor
    }
  }

  private static func quantize(_ radians: Double) -> Int {
    var degrees = radians * 180 / .pi
    degrees = degrees.truncatingRemainder(dividingBy: 360)
    if degrees < 0 { degrees += 360 }
    return (Int((degrees / 5).rounded()) * 5) % 360
  }

  private static func make(kind: Kind, degrees: Double) -> NSCursor {
    let image = NSImage(size: NSSize(width: 24, height: 24), flipped: true) { _ in
      let transform = NSAffineTransform()
      transform.translateX(by: 12, yBy: 12)
      transform.rotate(byDegrees: CGFloat(degrees))
      transform.translateX(by: -12, yBy: -12)
      transform.concat()
      guard let context = NSGraphicsContext.current?.cgContext else { return false }
      switch kind {
      case .scale: drawScale(in: context)
      case .rotate: drawRotate(in: context)
      }
      return true
    }
    return NSCursor(image: image, hotSpot: NSPoint(x: 12, y: 12))
  }

  private static let outline = CGColor(gray: 0.05, alpha: 1)
  private static let stroke = CGColor(gray: 0.97, alpha: 1)

  private static func drawScale(in context: CGContext) {
    strokeScale(in: context, width: 3.2, color: outline)
    strokeScale(in: context, width: 1.8, color: stroke)
  }

  private static func strokeScale(in context: CGContext, width: CGFloat, color: CGColor) {
    context.saveGState()
    context.setStrokeColor(color)
    context.setLineWidth(width)
    context.setLineCap(.round)
    context.setLineJoin(.round)
    context.move(to: CGPoint(x: 4.5, y: 12))
    context.addLine(to: CGPoint(x: 19.5, y: 12))
    context.move(to: CGPoint(x: 4.5, y: 12))
    context.addLine(to: CGPoint(x: 7.5, y: 9))
    context.move(to: CGPoint(x: 4.5, y: 12))
    context.addLine(to: CGPoint(x: 7.5, y: 15))
    context.move(to: CGPoint(x: 19.5, y: 12))
    context.addLine(to: CGPoint(x: 16.5, y: 9))
    context.move(to: CGPoint(x: 19.5, y: 12))
    context.addLine(to: CGPoint(x: 16.5, y: 15))
    context.strokePath()
    context.restoreGState()
  }

  private static func drawRotate(in context: CGContext) {
    strokeRotate(in: context, width: 3.2, color: outline)
    strokeRotate(in: context, width: 1.8, color: stroke)
  }

  private static func strokeRotate(in context: CGContext, width: CGFloat, color: CGColor) {
    context.saveGState()
    context.setStrokeColor(color)
    context.setLineWidth(width)
    context.setLineCap(.round)
    context.setLineJoin(.round)
    let center = CGPoint(x: 12, y: 12)
    let radius: CGFloat = 5.8
    let start = CGPoint(x: 6.2, y: 12)
    let end = CGPoint(x: 16.1, y: 16.1)
    context.move(to: start)
    context.addArc(
      center: center,
      radius: radius,
      startAngle: .pi,
      endAngle: atan2(end.y - center.y, end.x - center.x),
      clockwise: false
    )
    context.move(to: end)
    context.addLine(to: CGPoint(x: 15.7, y: 12.7))
    context.move(to: end)
    context.addLine(to: CGPoint(x: 19.4, y: 15.3))
    context.strokePath()
    context.restoreGState()
  }
}
