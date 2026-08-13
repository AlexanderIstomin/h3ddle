import CoreGraphics
import Foundation

struct CanvasViewport: Equatable {
  static let minimumMagnification: CGFloat = 0.25
  static let maximumMagnification: CGFloat = 8

  var magnification: CGFloat = 1
  var offset: CGSize = .zero

  mutating func reset() {
    magnification = 1
    offset = .zero
  }

  mutating func zoom(by factor: CGFloat) {
    magnification = min(
      max(magnification * factor, Self.minimumMagnification),
      Self.maximumMagnification
    )
  }

  mutating func pan(by translation: CGSize) {
    offset.width += translation.width
    offset.height += translation.height
  }
}
