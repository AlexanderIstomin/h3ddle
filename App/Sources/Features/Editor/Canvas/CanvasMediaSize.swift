import AVFoundation
import CoreGraphics
import Foundation
import H3ddleCore
import ImageIO

@MainActor
enum CanvasMediaSize {
  private static var cache: [URL: (width: Double, height: Double)] = [:]

  static func size(for asset: AssetReference) -> (width: Double, height: Double) {
    if let cached = cache[asset.url] { return cached }
    let resolved: (width: Double, height: Double)
    switch asset.kind {
    case .image:
      resolved = imageSize(at: asset.url) ?? (1920, 1080)
    case .video:
      resolved = videoSize(at: asset.url) ?? (1920, 1080)
    case .audio:
      resolved = (1920, 1080)
    }
    cache[asset.url] = resolved
    return resolved
  }

  private static func imageSize(at url: URL) -> (width: Double, height: Double)? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
      let height = properties[kCGImagePropertyPixelHeight] as? NSNumber
    else {
      return nil
    }
    return (width.doubleValue, height.doubleValue)
  }

  private static func videoSize(at url: URL) -> (width: Double, height: Double)? {
    let asset = AVURLAsset(url: url)
    guard let track = asset.tracks(withMediaType: .video).first else { return nil }
    let size = track.naturalSize.applying(track.preferredTransform)
    let width = abs(Double(size.width))
    let height = abs(Double(size.height))
    guard width > 1, height > 1 else { return nil }
    return (width, height)
  }
}
