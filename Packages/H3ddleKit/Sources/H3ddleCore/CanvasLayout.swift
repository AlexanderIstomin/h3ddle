import Foundation

public enum CanvasFit: String, Codable, Sendable, Hashable, CaseIterable {
  /// Letterbox the media inside the canvas.
  case fit
  /// Crop the media so it covers the canvas.
  case cover
}

/// UI-free placement + free transform. Translation is normalized to the
/// program canvas: +x is right, +y is up, 1.0 is one canvas width/height.
/// Rotation is clockwise radians, matching SwiftUI and the compositor sign
/// flip (bitmap contexts are y-up).
public struct CanvasObjectTransform: Hashable, Codable, Sendable {
  public var fit: CanvasFit
  public var translationX: Double
  public var translationY: Double
  public var scale: Double
  public var rotationRadians: Double

  public static let identity = CanvasObjectTransform()

  public init(
    fit: CanvasFit = .fit,
    translationX: Double = 0,
    translationY: Double = 0,
    scale: Double = 1,
    rotationRadians: Double = 0
  ) {
    self.fit = fit
    self.translationX = translationX
    self.translationY = translationY
    self.scale = max(scale, 0.01)
    self.rotationRadians = rotationRadians
  }

  /// Legacy / test convenience. Sets `rotationRadians = turns * π/2`.
  public init(fit: CanvasFit = .fit, rotationTurns: Int) {
    self.init(
      fit: fit,
      rotationRadians: Double(CanvasLayout.normalizedTurns(rotationTurns)) * .pi / 2
    )
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    fit = try container.decodeIfPresent(CanvasFit.self, forKey: .fit) ?? .fit
    translationX = try container.decodeIfPresent(Double.self, forKey: .translationX) ?? 0
    translationY = try container.decodeIfPresent(Double.self, forKey: .translationY) ?? 0
    scale = max(try container.decodeIfPresent(Double.self, forKey: .scale) ?? 1, 0.01)
    if let radians = try container.decodeIfPresent(Double.self, forKey: .rotationRadians) {
      rotationRadians = radians
    } else {
      let turns = CanvasLayout.normalizedTurns(
        try container.decodeIfPresent(Int.self, forKey: .rotationTurns) ?? 0
      )
      rotationRadians = Double(turns) * .pi / 2
    }
  }

  public func encode(to encoder: any Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(fit, forKey: .fit)
    try container.encode(translationX, forKey: .translationX)
    try container.encode(translationY, forKey: .translationY)
    try container.encode(scale, forKey: .scale)
    try container.encode(rotationRadians, forKey: .rotationRadians)
  }

  /// Nearest quarter-turn, hint only. Ignore on decode when radians exist.
  public var rotationTurns: Int {
    CanvasLayout.normalizedTurns(
      Int((rotationRadians / (.pi / 2)).rounded())
    )
  }

  public var rotationDegrees: Double {
    rotationRadians * 180 / .pi
  }

  public func rotated(by turns: Int = 1) -> CanvasObjectTransform {
    var next = self
    next.rotationRadians += Double(turns) * .pi / 2
    return next
  }

  enum CodingKeys: String, CodingKey {
    case fit
    case translationX
    case translationY
    case scale
    case rotationRadians
    case rotationTurns
  }
}

public typealias VisualCanvasTransform = CanvasObjectTransform

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
    public var maxX: Double { x + width }
    public var maxY: Double { y + height }
  }

  public struct Placement: Equatable, Sendable {
    public var centerX: Double
    public var centerY: Double
    /// Unrotated draw size in the same pixel space as the canvas.
    public var drawWidth: Double
    public var drawHeight: Double
    public var rotationRadians: Double
    /// Axis-aligned bounds after rotation. May exceed the canvas at odd angles.
    public var aabb: Rect

    public init(
      centerX: Double,
      centerY: Double,
      drawWidth: Double,
      drawHeight: Double,
      rotationRadians: Double,
      aabb: Rect
    ) {
      self.centerX = centerX
      self.centerY = centerY
      self.drawWidth = drawWidth
      self.drawHeight = drawHeight
      self.rotationRadians = rotationRadians
      self.aabb = aabb
    }
  }

  public static func normalizedTurns(_ turns: Int) -> Int {
    let remainder = turns % 4
    return remainder >= 0 ? remainder : remainder + 4
  }

  /// Clockwise rotation in a y-up plane.
  public static func rotateClockwise(
    x: Double,
    y: Double,
    radians: Double
  ) -> (x: Double, y: Double) {
    let cosine = Foundation.cos(radians)
    let sine = Foundation.sin(radians)
    return (x * cosine + y * sine, -x * sine + y * cosine)
  }

  /// Media placement. Fit/cover run on the unrotated source, then the fitted
  /// quad is rotated. `source` and `canvas` are the same pixel space.
  public static func placed(
    sourceWidth: Double,
    sourceHeight: Double,
    canvasWidth: Double,
    canvasHeight: Double,
    transform: CanvasObjectTransform
  ) -> Placement {
    guard sourceWidth > 0, sourceHeight > 0, canvasWidth > 0, canvasHeight > 0 else {
      return Placement(
        centerX: canvasWidth / 2,
        centerY: canvasHeight / 2,
        drawWidth: max(0, canvasWidth),
        drawHeight: max(0, canvasHeight),
        rotationRadians: transform.rotationRadians,
        aabb: Rect(x: 0, y: 0, width: max(0, canvasWidth), height: max(0, canvasHeight))
      )
    }
    let fitScale: Double
    switch transform.fit {
    case .fit:
      fitScale = min(canvasWidth / sourceWidth, canvasHeight / sourceHeight)
    case .cover:
      fitScale = max(canvasWidth / sourceWidth, canvasHeight / sourceHeight)
    }
    let drawWidth = sourceWidth * fitScale * transform.scale
    let drawHeight = sourceHeight * fitScale * transform.scale
    let centerX = canvasWidth / 2 + transform.translationX * canvasWidth
    let centerY = canvasHeight / 2 + transform.translationY * canvasHeight
    return Placement(
      centerX: centerX,
      centerY: centerY,
      drawWidth: drawWidth,
      drawHeight: drawHeight,
      rotationRadians: transform.rotationRadians,
      aabb: aabb(
        centerX: centerX,
        centerY: centerY,
        width: drawWidth,
        height: drawHeight,
        rotationRadians: transform.rotationRadians
      )
    )
  }

  /// Overlay placement. No fit/cover: source pixels are already in canvas
  /// space, then uniform scale, translation, and rotation apply.
  public static func overlayPlaced(
    sourceWidth: Double,
    sourceHeight: Double,
    canvasWidth: Double,
    canvasHeight: Double,
    transform: CanvasObjectTransform
  ) -> Placement {
    guard sourceWidth > 0, sourceHeight > 0, canvasWidth > 0, canvasHeight > 0 else {
      return Placement(
        centerX: canvasWidth / 2,
        centerY: canvasHeight / 2,
        drawWidth: max(0, sourceWidth),
        drawHeight: max(0, sourceHeight),
        rotationRadians: transform.rotationRadians,
        aabb: Rect(x: 0, y: 0, width: max(0, canvasWidth), height: max(0, canvasHeight))
      )
    }
    let drawWidth = sourceWidth * transform.scale
    let drawHeight = sourceHeight * transform.scale
    let centerX = canvasWidth / 2 + transform.translationX * canvasWidth
    let centerY = canvasHeight / 2 + transform.translationY * canvasHeight
    return Placement(
      centerX: centerX,
      centerY: centerY,
      drawWidth: drawWidth,
      drawHeight: drawHeight,
      rotationRadians: transform.rotationRadians,
      aabb: aabb(
        centerX: centerX,
        centerY: centerY,
        width: drawWidth,
        height: drawHeight,
        rotationRadians: transform.rotationRadians
      )
    )
  }

  /// Corners of a source-space rect after overlay placement, in canvas pixels.
  public static func overlayQuad(
    rect: Rect,
    sourceWidth: Double,
    sourceHeight: Double,
    placement: Placement
  ) -> [(x: Double, y: Double)] {
    let scaleX = placement.drawWidth / max(sourceWidth, 0.001)
    let scaleY = placement.drawHeight / max(sourceHeight, 0.001)
    let corners = [
      (rect.x, rect.y),
      (rect.maxX, rect.y),
      (rect.maxX, rect.maxY),
      (rect.x, rect.maxY),
    ]
    return corners.map { point in
      let localX = (point.0 - sourceWidth / 2) * scaleX
      let localY = (point.1 - sourceHeight / 2) * scaleY
      let rotated = rotateClockwise(
        x: localX,
        y: localY,
        radians: placement.rotationRadians
      )
      return (placement.centerX + rotated.x, placement.centerY + rotated.y)
    }
  }

  /// Bounding box of the media after fit, scale, translation, and rotation.
  public static func destination(
    sourceWidth: Double,
    sourceHeight: Double,
    canvasWidth: Double,
    canvasHeight: Double,
    transform: VisualCanvasTransform
  ) -> Rect {
    placed(
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight,
      transform: transform
    ).aabb
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

  public static func orientedSize(
    width: Double,
    height: Double,
    rotationRadians: Double
  ) -> (width: Double, height: Double) {
    let aabb = aabb(
      centerX: 0,
      centerY: 0,
      width: width,
      height: height,
      rotationRadians: rotationRadians
    )
    return (aabb.width, aabb.height)
  }

  /// Four corners of the unrotated draw rect, rotated around center.
  /// Order: bottom-left, bottom-right, top-right, top-left (y-up).
  public static func projectedQuad(
    centerX: Double,
    centerY: Double,
    drawWidth: Double,
    drawHeight: Double,
    rotationRadians: Double
  ) -> [(x: Double, y: Double)] {
    let locals: [(Double, Double)] = [
      (-drawWidth / 2, -drawHeight / 2),
      (drawWidth / 2, -drawHeight / 2),
      (drawWidth / 2, drawHeight / 2),
      (-drawWidth / 2, drawHeight / 2),
    ]
    return locals.map { local in
      let rotated = rotateClockwise(x: local.0, y: local.1, radians: rotationRadians)
      return (centerX + rotated.x, centerY + rotated.y)
    }
  }

  public static func aabb(
    centerX: Double,
    centerY: Double,
    width: Double,
    height: Double,
    rotationRadians: Double
  ) -> Rect {
    let quad = projectedQuad(
      centerX: centerX,
      centerY: centerY,
      drawWidth: width,
      drawHeight: height,
      rotationRadians: rotationRadians
    )
    let xs = quad.map(\.x)
    let ys = quad.map(\.y)
    let minX = xs.min() ?? centerX
    let maxX = xs.max() ?? centerX
    let minY = ys.min() ?? centerY
    let maxY = ys.max() ?? centerY
    return Rect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
  }
}
