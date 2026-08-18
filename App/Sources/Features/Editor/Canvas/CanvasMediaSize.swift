import AVFoundation
import CoreGraphics
import Foundation
import H3ddleCore
import ImageIO

struct CanvasMediaDimensions: Equatable, Sendable {
  var width: Double
  var height: Double

  static let fallback = CanvasMediaDimensions(width: 1_920, height: 1_080)
}

actor CanvasMediaSize {
  static let shared = CanvasMediaSize()

  private var cache: [URL: CanvasMediaDimensions] = [:]

  func size(for asset: AssetReference) async -> CanvasMediaDimensions {
    if let cached = cache[asset.url] { return cached }
    let resolved: CanvasMediaDimensions
    switch asset.kind {
    case .image:
      resolved = Self.imageSize(at: asset.url) ?? .fallback
    case .video:
      resolved = await Self.videoSize(at: asset.url) ?? .fallback
    case .audio:
      resolved = .fallback
    }
    cache[asset.url] = resolved
    return resolved
  }

  nonisolated private static func imageSize(at url: URL) -> CanvasMediaDimensions? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
      let height = properties[kCGImagePropertyPixelHeight] as? NSNumber
    else {
      return nil
    }
    return CanvasMediaDimensions(width: width.doubleValue, height: height.doubleValue)
  }

  nonisolated private static func videoSize(at url: URL) async -> CanvasMediaDimensions? {
    let asset = AVURLAsset(url: url)
    guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return nil }
    guard let naturalSize = try? await track.load(.naturalSize),
      let preferredTransform = try? await track.load(.preferredTransform)
    else { return nil }
    let size = naturalSize.applying(preferredTransform)
    let width = abs(Double(size.width))
    let height = abs(Double(size.height))
    guard width > 1, height > 1 else { return nil }
    return CanvasMediaDimensions(width: width, height: height)
  }
}
