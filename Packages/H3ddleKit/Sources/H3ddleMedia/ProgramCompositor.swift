import AVFoundation
import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import H3ddleCore
import ImageIO
import VideoToolbox

/// Places the current visual onto the program canvas. Preview and export share this path.
public final class ProgramCompositor: @unchecked Sendable {
  public let width: Int
  public let height: Int
  /// Authoring canvas from `ProjectSettings`. Defaults to the buffer so
  /// existing tests keep `composeScale == 1`.
  public let layoutWidth: Int
  public let layoutHeight: Int
  public let background: (CGFloat, CGFloat, CGFloat)
  public let backgroundAlpha: CGFloat

  public var composeScale: Double {
    Double(width) / Double(max(layoutWidth, 1))
  }

  private let images = NSCache<NSURL, CGImage>()
  private let textImages = NSCache<CacheKey<TextRasterKey>, CGImage>()
  private let generators = NSCache<NSURL, AVAssetImageGenerator>()
  private let ciContext = CIContext(options: [CIContextOption.useSoftwareRenderer: false])

  public init(
    width: Int,
    height: Int,
    background: (CGFloat, CGFloat, CGFloat) = (0, 0, 0),
    backgroundAlpha: CGFloat = 1,
    layoutWidth: Int? = nil,
    layoutHeight: Int? = nil
  ) {
    self.width = max(1, width)
    self.height = max(1, height)
    self.layoutWidth = max(1, layoutWidth ?? width)
    self.layoutHeight = max(1, layoutHeight ?? height)
    self.background = background
    self.backgroundAlpha = min(max(backgroundAlpha, 0), 1)
    images.countLimit = 12
    images.totalCostLimit = 256 * 1_024 * 1_024
    textImages.countLimit = 64
    textImages.totalCostLimit = 64 * 1_024 * 1_024
    generators.countLimit = 8
  }

  public convenience init(settings: ProjectSettings) {
    self.init(
      width: settings.width,
      height: settings.height,
      background: settings.background,
      layoutWidth: settings.width,
      layoutHeight: settings.height
    )
  }

  public convenience init(
    width: Int,
    height: Int,
    background: ProjectBackground,
    fillsBackground: Bool = true,
    layoutWidth: Int? = nil,
    layoutHeight: Int? = nil
  ) {
    let color = background.compositorColor
    let alpha: CGFloat = (!fillsBackground && background.isClear) ? 0 : 1
    self.init(
      width: width,
      height: height,
      background: color,
      backgroundAlpha: alpha,
      layoutWidth: layoutWidth,
      layoutHeight: layoutHeight
    )
  }

  public func pixelBuffer(
    for frame: ProgramPreviewFrame,
    videoFrame: CGImage? = nil
  ) async -> CVPixelBuffer? {
    if let mix = frame.transition {
      let outgoing = await sourceImage(for: mix.outgoing)
      let incoming: CGImage?
      if case .video = mix.incoming, let videoFrame {
        incoming = videoFrame
      } else {
        incoming = await sourceImage(for: mix.incoming)
      }
      let outgoingPlaced = rasterize(
        outgoing,
        transform: mix.outgoingTransform,
        effects: mix.outgoingEffects,
        time: frame.time
      )
      let incomingPlaced = rasterize(
        incoming,
        transform: mix.incomingTransform,
        effects: mix.incomingEffects,
        time: frame.time
      )
      guard
        let buffer = pixelBuffer(
          placing: outgoingPlaced,
          transform: .identity,
          incoming: incomingPlaced,
          incomingTransform: .identity,
          progress: mix.progress,
          kind: mix.kind
        )
      else {
        return nil
      }
      draw(frame.overlays, onto: buffer)
      return buffer
    }
    let image: CGImage?
    if case .video = frame.visual, let videoFrame {
      image = videoFrame
    } else {
      image = await sourceImage(for: frame.visual)
    }
    guard let buffer = pixelBuffer(placing: image, transform: frame.visualTransform) else {
      return nil
    }
    apply(frame.visualEffects, to: buffer, at: frame.time)
    draw(frame.overlays, onto: buffer)
    return buffer
  }

  public static func makeImage(from buffer: CVPixelBuffer) -> CGImage? {
    var image: CGImage?
    let status = VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &image)
    if status == noErr, let image { return image }
    return copyImage(from: buffer)
  }

  private static func copyImage(from buffer: CVPixelBuffer) -> CGImage? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    var bytes = [UInt8](repeating: 0, count: stride * height)
    memcpy(&bytes, base, bytes.count)
    guard
      let context = CGContext(
        data: &bytes,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: stride,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
          | CGBitmapInfo.byteOrder32Little.rawValue
      )
    else {
      return nil
    }
    return context.makeImage()
  }

  public func pixelBuffer(
    placing image: CGImage?,
    transform: VisualCanvasTransform
  ) -> CVPixelBuffer? {
    guard let prepared = prepareCanvas() else { return nil }
    defer { CVPixelBufferUnlockBaseAddress(prepared.buffer, []) }
    if let image {
      draw(image, in: prepared.context, transform: transform)
    }
    return prepared.buffer
  }

  public func pixelBuffer(
    placing outgoing: CGImage?,
    transform outgoingTransform: VisualCanvasTransform,
    incoming: CGImage?,
    incomingTransform: VisualCanvasTransform,
    progress: Double,
    kind: VisualTransitionKind
  ) -> CVPixelBuffer? {
    guard let prepared = prepareCanvas() else { return nil }
    defer { CVPixelBufferUnlockBaseAddress(prepared.buffer, []) }
    let progress = min(max(progress, 0), 1)
    switch kind {
    case .dissolve:
      paint(outgoing, transform: outgoingTransform, in: prepared.context, alpha: 1)
      paint(incoming, transform: incomingTransform, in: prepared.context, alpha: progress)
    case .fade:
      if progress < 0.5 {
        paint(outgoing, transform: outgoingTransform, in: prepared.context, alpha: 1 - progress * 2)
      } else {
        paint(incoming, transform: incomingTransform, in: prepared.context, alpha: progress * 2 - 1)
      }
    case .wipe:
      paint(outgoing, transform: outgoingTransform, in: prepared.context, alpha: 1)
      prepared.context.saveGState()
      prepared.context.clip(
        to: CGRect(x: 0, y: 0, width: CGFloat(progress) * CGFloat(width), height: CGFloat(height))
      )
      paint(incoming, transform: incomingTransform, in: prepared.context, alpha: 1)
      prepared.context.restoreGState()
    }
    return prepared.buffer
  }

  private func prepareCanvas() -> (buffer: CVPixelBuffer, context: CGContext)? {
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width,
      height,
      kCVPixelFormatType_32BGRA,
      [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
      ] as CFDictionary,
      &buffer
    )
    guard status == kCVReturnSuccess, let buffer else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    guard let data = CVPixelBufferGetBaseAddress(buffer),
      let context = CGContext(
        data: data,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
          | CGBitmapInfo.byteOrder32Little.rawValue
      )
    else {
      CVPixelBufferUnlockBaseAddress(buffer, [])
      return nil
    }
    fillBackground(context)
    return (buffer, context)
  }

  private func fillBackground(_ context: CGContext) {
    context.setFillColor(
      red: background.0,
      green: background.1,
      blue: background.2,
      alpha: backgroundAlpha
    )
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
  }

  private func paint(
    _ image: CGImage?,
    transform: VisualCanvasTransform,
    in context: CGContext,
    alpha: CGFloat
  ) {
    guard let image, alpha > 0.000_1 else { return }
    context.saveGState()
    context.setAlpha(min(max(alpha, 0), 1))
    draw(image, in: context, transform: transform)
    context.restoreGState()
  }

  private func sourceImage(for visual: ProgramVisualPresentation) async -> CGImage? {
    switch visual {
    case .empty:
      return nil
    case .image(let asset):
      return stillImage(at: asset.url)
    case .video(let asset, let localTime, _):
      return await videoImage(at: asset.url, time: localTime)
    }
  }

  private func stillImage(at url: URL) -> CGImage? {
    let key = url as NSURL
    if let cached = images.object(forKey: key) { return cached }
    guard FileManager.default.fileExists(atPath: url.path),
      let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      return nil
    }
    images.setObject(image, forKey: key, cost: Self.cacheCost(of: image))
    return image
  }

  private func videoImage(at url: URL, time: TimeInterval) async -> CGImage? {
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    let key = url as NSURL
    let generator: AVAssetImageGenerator
    if let cached = generators.object(forKey: key) {
      generator = cached
    } else {
      let created = AVAssetImageGenerator(asset: AVURLAsset(url: url))
      created.appliesPreferredTrackTransform = true
      created.requestedTimeToleranceBefore = .zero
      created.requestedTimeToleranceAfter = .zero
      generators.setObject(created, forKey: key)
      generator = created
    }
    let cmTime = CMTime(seconds: max(0, time), preferredTimescale: 600)
    return try? await generator.image(at: cmTime).image
  }

  private func draw(
    _ image: CGImage,
    in context: CGContext,
    transform: VisualCanvasTransform
  ) {
    let placement = CanvasLayout.placed(
      sourceWidth: Double(image.width),
      sourceHeight: Double(image.height),
      canvasWidth: Double(width),
      canvasHeight: Double(height),
      transform: transform
    )
    context.saveGState()
    context.translateBy(x: placement.centerX, y: placement.centerY)
    // Bitmap contexts are y-up; match SwiftUI's clockwise rotationEffect.
    context.rotate(by: -CGFloat(placement.rotationRadians))
    context.draw(
      image,
      in: CGRect(
        x: -placement.drawWidth / 2,
        y: -placement.drawHeight / 2,
        width: placement.drawWidth,
        height: placement.drawHeight
      )
    )
    context.restoreGState()
  }

  private func draw(_ overlays: [ProgramTextPresentation], onto buffer: CVPixelBuffer) {
    guard !overlays.isEmpty else { return }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let data = CVPixelBufferGetBaseAddress(buffer),
      let context = CGContext(
        data: data,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
          | CGBitmapInfo.byteOrder32Little.rawValue
      )
    else {
      return
    }
    for overlay in overlays {
      draw(overlay, in: context)
    }
  }

  private func draw(_ overlay: ProgramTextPresentation, in context: CGContext) {
    let pixelScale = composeScale * overlay.transform.scale
    let bucket = Int((pixelScale * 20).rounded())
    let key = TextRasterKey(
      itemID: overlay.item.id,
      text: overlay.item.text,
      style: overlay.item.style,
      layoutWidth: layoutWidth,
      layoutHeight: layoutHeight,
      scaleBucket: bucket
    )
    let cacheKey = CacheKey(key)
    let image: CGImage
    let layout: TextLayout
    if let cached = textImages.object(forKey: cacheKey) {
      image = cached
      layout = TextRasterizer.layout(
        overlay.item,
        layoutSize: (layoutWidth, layoutHeight)
      )
    } else if let raster = TextRasterizer.raster(
      overlay.item,
      layoutSize: (layoutWidth, layoutHeight),
      pixelScale: pixelScale
    ) {
      textImages.setObject(
        raster.image,
        forKey: cacheKey,
        cost: Self.cacheCost(of: raster.image)
      )
      image = raster.image
      layout = raster.layout
    } else {
      return
    }
    let drawWidth = layout.expandedSize.width * pixelScale
    let drawHeight = layout.expandedSize.height * pixelScale
    let centerX = Double(width) / 2 + overlay.transform.translationX * Double(width)
    let centerY = Double(height) / 2 + overlay.transform.translationY * Double(height)
    context.saveGState()
    context.translateBy(x: centerX, y: centerY)
    context.rotate(by: -CGFloat(overlay.transform.rotationRadians))
    context.draw(
      image,
      in: CGRect(
        x: -drawWidth / 2,
        y: -drawHeight / 2,
        width: drawWidth,
        height: drawHeight
      )
    )
    context.restoreGState()
  }

  private static func cacheCost(of image: CGImage) -> Int {
    let (cost, overflow) = image.bytesPerRow.multipliedReportingOverflow(by: image.height)
    return overflow ? Int.max : cost
  }

  private func rasterize(
    _ image: CGImage?,
    transform: VisualCanvasTransform,
    effects: [VisualEffectInstance],
    time: TimeInterval
  ) -> CGImage? {
    guard let buffer = pixelBuffer(placing: image, transform: transform) else { return nil }
    apply(effects, to: buffer, at: time)
    return ProgramCompositor.makeImage(from: buffer)
  }

  private func apply(
    _ effects: [VisualEffectInstance],
    to buffer: CVPixelBuffer,
    at time: TimeInterval
  ) {
    let enabled = effects.filter(\.isEnabled)
    guard !enabled.isEmpty else { return }
    var image = CIImage(cvPixelBuffer: buffer)
    let canvas = CGRect(x: 0, y: 0, width: width, height: height)
    for effect in enabled {
      image = filter(effect, input: image, time: time).cropped(to: canvas)
    }
    ciContext.render(image, to: buffer)
  }

  private func filter(
    _ effect: VisualEffectInstance,
    input: CIImage,
    time: TimeInterval
  ) -> CIImage {
    switch effect.kind {
    case .colorGrade:
      var image = input
      if let filter = CIFilter(name: "CIExposureAdjust") {
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(effect.value("exposure"), forKey: kCIInputEVKey)
        image = filter.outputImage ?? image
      }
      if let filter = CIFilter(name: "CIColorControls") {
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(effect.value("contrast", default: 1), forKey: kCIInputContrastKey)
        filter.setValue(effect.value("saturation", default: 1), forKey: kCIInputSaturationKey)
        image = filter.outputImage ?? image
      }
      return image
    case .vignette:
      guard let filter = CIFilter(name: "CIVignette") else { return input }
      filter.setValue(input, forKey: kCIInputImageKey)
      filter.setValue(effect.value("amount") * 2, forKey: kCIInputIntensityKey)
      filter.setValue(1 + effect.value("feather") * 2, forKey: kCIInputRadiusKey)
      return filter.outputImage ?? input
    case .filmGrain:
      return filmGrain(input, effect: effect, time: time)
    case .sharpen:
      guard let filter = CIFilter(name: "CISharpenLuminance") else { return input }
      filter.setValue(input, forKey: kCIInputImageKey)
      filter.setValue(effect.value("amount") * 2, forKey: kCIInputSharpnessKey)
      return filter.outputImage ?? input
    case .blur:
      guard let filter = CIFilter(name: "CIGaussianBlur") else { return input }
      filter.setValue(input, forKey: kCIInputImageKey)
      filter.setValue(effect.value("radius"), forKey: kCIInputRadiusKey)
      return (filter.outputImage ?? input).cropped(to: input.extent)
    case .bloom:
      guard let filter = CIFilter(name: "CIBloom") else { return input }
      filter.setValue(input, forKey: kCIInputImageKey)
      filter.setValue(effect.value("amount"), forKey: kCIInputIntensityKey)
      filter.setValue(4 + effect.value("threshold") * 20, forKey: kCIInputRadiusKey)
      return (filter.outputImage ?? input).cropped(to: input.extent)
    case .chromaKey:
      return chromaKey(input, effect: effect)
    }
  }

  private func filmGrain(
    _ input: CIImage,
    effect: VisualEffectInstance,
    time: TimeInterval
  ) -> CIImage {
    let amount = effect.value("amount")
    let cells = 16 + effect.value("size", default: 0.5) * 384
    let frame = floor(time * 60)
    if let kernel = CompositorKernels.grain,
      let output = kernel.apply(
        extent: input.extent,
        roiCallback: { _, rect in rect },
        arguments: [input, amount, cells, frame, Double(width), Double(height)]
      )
    {
      return output
    }
    return hashedFilmGrain(input, amount: amount, cells: cells, frame: frame)
  }

  /// Same hash as the Metal kernel so grain chatters per frame if compile fails.
  private func hashedFilmGrain(
    _ input: CIImage,
    amount: Double,
    cells: Double,
    frame: Double
  ) -> CIImage {
    guard amount > 0.000_1,
      let cgImage = ciContext.createCGImage(input, from: input.extent)
    else {
      return input
    }
    let pixelWidth = cgImage.width
    let pixelHeight = cgImage.height
    let stride = pixelWidth * 4
    var pixels = [UInt8](repeating: 0, count: stride * pixelHeight)
    guard
      let context = CGContext(
        data: &pixels,
        width: pixelWidth,
        height: pixelHeight,
        bitsPerComponent: 8,
        bytesPerRow: stride,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      )
    else {
      return input
    }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
    let cellCount = max(8, cells)
    for y in 0..<pixelHeight {
      for x in 0..<pixelWidth {
        let offset = y * stride + x * 4
        let red = Double(pixels[offset]) / 255
        let green = Double(pixels[offset + 1]) / 255
        let blue = Double(pixels[offset + 2]) / 255
        let luma = red * 0.2126 + green * 0.7152 + blue * 0.0722
        let mid = luma * (1 - luma) * 4
        let cellX = floor(Double(x) / Double(max(pixelWidth, 1)) * cellCount) + frame
        let cellY = floor(Double(y) / Double(max(pixelHeight, 1)) * cellCount) + frame
        let hash = sin(cellX * 12.9898 + cellY * 78.233) * 43758.5453
        let noise = (hash - floor(hash)) - 0.5
        let delta = noise * amount * mid
        pixels[offset] = UInt8(min(max(red + delta, 0), 1) * 255)
        pixels[offset + 1] = UInt8(min(max(green + delta, 0), 1) * 255)
        pixels[offset + 2] = UInt8(min(max(blue + delta, 0), 1) * 255)
      }
    }
    guard let grained = context.makeImage() else { return input }
    return CIImage(cgImage: grained).cropped(to: input.extent)
  }

  private func chromaKey(_ input: CIImage, effect: VisualEffectInstance) -> CIImage {
    let targetBlue = effect.value("hue") >= 0.5 ? 1.0 : 0.0
    let softness = max(0.04, effect.value("softness"))
    if let kernel = CompositorKernels.chroma,
      let output = kernel.apply(extent: input.extent, arguments: [input, targetBlue, softness])
    {
      return output
    }
    return hashedChromaKey(input, targetBlue: targetBlue > 0.5, softness: softness)
  }

  private func hashedChromaKey(
    _ input: CIImage,
    targetBlue: Bool,
    softness: Double
  ) -> CIImage {
    guard let cgImage = ciContext.createCGImage(input, from: input.extent) else { return input }
    let pixelWidth = cgImage.width
    let pixelHeight = cgImage.height
    let stride = pixelWidth * 4
    var pixels = [UInt8](repeating: 0, count: stride * pixelHeight)
    guard
      let context = CGContext(
        data: &pixels,
        width: pixelWidth,
        height: pixelHeight,
        bitsPerComponent: 8,
        bytesPerRow: stride,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      )
    else {
      return input
    }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
    let edge = 0.04 + softness * 0.35
    let satHigh = 0.18 + softness * 0.25
    for y in 0..<pixelHeight {
      for x in 0..<pixelWidth {
        let offset = y * stride + x * 4
        let red = Double(pixels[offset]) / 255
        let green = Double(pixels[offset + 1]) / 255
        let blue = Double(pixels[offset + 2]) / 255
        let alpha = Double(pixels[offset + 3]) / 255
        let primary = targetBlue ? blue : green
        let others = targetBlue ? max(red, green) : max(red, blue)
        let dominance = primary - others
        let sat = max(red, max(green, blue)) - min(red, min(green, blue))
        let mask =
          1
          - smoothstep(0.02, edge, dominance) * smoothstep(0.04, satHigh, sat)
        pixels[offset] = UInt8(min(max(red * mask, 0), 1) * 255)
        pixels[offset + 1] = UInt8(min(max(green * mask, 0), 1) * 255)
        pixels[offset + 2] = UInt8(min(max(blue * mask, 0), 1) * 255)
        pixels[offset + 3] = UInt8(min(max(alpha * mask, 0), 1) * 255)
      }
    }
    guard let keyed = context.makeImage() else { return input }
    return CIImage(cgImage: keyed).cropped(to: input.extent)
  }

  private func smoothstep(_ edge0: Double, _ edge1: Double, _ x: Double) -> Double {
    let t = min(max((x - edge0) / max(edge1 - edge0, 0.000_1), 0), 1)
    return t * t * (3 - 2 * t)
  }
}

private struct TextRasterKey: Hashable {
  var itemID: UUID
  var text: String
  var style: TextStyle
  var layoutWidth: Int
  var layoutHeight: Int
  var scaleBucket: Int
}

private final class CacheKey<Value: Hashable>: NSObject {
  let value: Value

  init(_ value: Value) {
    self.value = value
  }

  override var hash: Int { value.hashValue }

  override func isEqual(_ object: Any?) -> Bool {
    guard let other = object as? CacheKey<Value> else { return false }
    return value == other.value
  }
}

extension ProjectBackground {
  public var compositorColor: (CGFloat, CGFloat, CGFloat) {
    if isClear { return (0, 0, 0) }
    let cleaned = rawValue.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
    var value: UInt64 = 0
    Scanner(string: cleaned).scanHexInt64(&value)
    return (
      CGFloat((value >> 16) & 0xFF) / 255,
      CGFloat((value >> 8) & 0xFF) / 255,
      CGFloat(value & 0xFF) / 255
    )
  }
}
