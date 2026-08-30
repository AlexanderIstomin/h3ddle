import Foundation
import ImageIO

public enum UpscalingImageProbe {
  public static func pixelSize(at url: URL) throws -> UpscalingPixelSize {
    guard url.isFileURL, FileManager.default.isReadableFile(atPath: url.path),
      let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
        as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
      let height = properties[kCGImagePropertyPixelHeight] as? NSNumber
    else {
      throw UpscalingError.sourceNotReadable
    }
    let size = UpscalingPixelSize(width: width.intValue, height: height.intValue)
    guard size.isValid else { throw UpscalingError.invalidDimensions }
    return size
  }
}
