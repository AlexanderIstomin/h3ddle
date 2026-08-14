import Foundation

public enum CanvasFit: String, Codable, Sendable, Hashable, CaseIterable {
  /// Letterbox the media inside the canvas.
  case fit
  /// Crop the media so it covers the canvas.
  case cover
}

public struct VisualCanvasTransform: Hashable, Codable, Sendable {
  public var fit: CanvasFit
  /// Quarter-turns clockwise in `0...3`.
  public var rotationTurns: Int

  public static let identity = VisualCanvasTransform()

  public init(fit: CanvasFit = .fit, rotationTurns: Int = 0) {
    self.fit = fit
    self.rotationTurns = CanvasLayout.normalizedTurns(rotationTurns)
  }

  public func rotated(by turns: Int = 1) -> VisualCanvasTransform {
    VisualCanvasTransform(fit: fit, rotationTurns: rotationTurns + turns)
  }

  public var rotationRadians: Double {
    Double(rotationTurns) * .pi / 2
  }

  public var rotationDegrees: Double {
    Double(rotationTurns) * 90
  }
}

public enum CanvasLayout {
  public struct Rect: Equatable, Sendable {
    public var x: Double
    public var y: Double
    public var width: Double
    public var height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
      self.x = x
      self.y = y
      self.width = width
      self.height = height
    }

    public var midX: Double { x + width / 2 }
    public var midY: Double { y + height / 2 }
  }

  public static func normalizedTurns(_ turns: Int) -> Int {
    let remainder = turns % 4
    return remainder >= 0 ? remainder : remainder + 4
  }

  /// Bounding box of the media after rotation, centered on the canvas.
  public static func destination(
    sourceWidth: Double,
    sourceHeight: Double,
    canvasWidth: Double,
    canvasHeight: Double,
    transform: VisualCanvasTransform
  ) -> Rect {
    let source = orientedSize(
      width: sourceWidth,
      height: sourceHeight,
      rotationTurns: transform.rotationTurns
    )
    guard source.width > 0, source.height > 0, canvasWidth > 0, canvasHeight > 0 else {
      return Rect(x: 0, y: 0, width: max(0, canvasWidth), height: max(0, canvasHeight))
    }
    let scale: Double
    switch transform.fit {
    case .fit:
      scale = min(canvasWidth / source.width, canvasHeight / source.height)
    case .cover:
      scale = max(canvasWidth / source.width, canvasHeight / source.height)
    }
    let width = source.width * scale
    let height = source.height * scale
    return Rect(
      x: (canvasWidth - width) / 2,
      y: (canvasHeight - height) / 2,
      width: width,
      height: height
    )
  }

  /// Unrotated draw size that matches `destination` after `rotationTurns`.
  public static func unrotatedSize(
    destination: Rect,
    rotationTurns: Int
  ) -> (width: Double, height: Double) {
    if normalizedTurns(rotationTurns).isMultiple(of: 2) {
      return (destination.width, destination.height)
    }
    return (destination.height, destination.width)
  }

  public static func orientedSize(
    width: Double,
    height: Double,
    rotationTurns: Int
  ) -> (width: Double, height: Double) {
    if normalizedTurns(rotationTurns).isMultiple(of: 2) {
      return (width, height)
    }
    return (height, width)
  }
}
