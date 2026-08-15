import CoreGraphics
import CoreText
import Foundation
import H3ddleCore
import os

public struct TextLayout: Sendable {
  public var expandedSize: (width: Double, height: Double)
  public var lineFragments: [CanvasLayout.Rect]
  public var glyphBounds: [CanvasLayout.Rect]
}

public enum FontResolver {
  public static func font(for style: TextStyle) -> CTFont {
    let size = CGFloat(max(style.fontSize, 1))
    if let name = style.fontPostScriptName as CFString?,
      let named = CTFontCreateWithName(name, size, nil) as CTFont?
    {
      return named
    }
    if let matched = matchFamily(style, size: size) {
      return matched
    }
    Logger(subsystem: "com.h3ddle.app", category: "h3ddle.media.text").info(
      "Font fallback for \(style.fontFamily, privacy: .public)"
    )
    return CTFontCreateUIFontForLanguage(.system, size, nil)
      ?? CTFontCreateWithName("Helvetica" as CFString, size, nil)
  }

  private static func matchFamily(_ style: TextStyle, size: CGFloat) -> CTFont? {
    guard let names = CTFontManagerCopyAvailablePostScriptNames() as? [String] else {
      return nil
    }
    let targetWeight = weightTrait(style.fontWeight)
    var best: (font: CTFont, score: Double)?
    for name in names {
      let base = CTFontCreateWithName(name as CFString, size, nil)
      let family = CTFontCopyFamilyName(base) as String
      guard family.caseInsensitiveCompare(style.fontFamily) == .orderedSame
        || name.caseInsensitiveCompare(style.fontFamily) == .orderedSame
      else {
        continue
      }
      let traits = CTFontCopyTraits(base) as NSDictionary
      let weight = (traits[kCTFontWeightTrait] as? NSNumber)?.doubleValue ?? 0
      let slant = (traits[kCTFontSlantTrait] as? NSNumber)?.doubleValue ?? 0
      let italicScore = style.italic ? abs(slant - 0.07) : abs(slant)
      let score = abs(weight - targetWeight) + italicScore
      if best == nil || score < best!.score {
        best = (base, score)
      }
    }
    return best?.font
  }

  private static func weightTrait(_ weight: Int) -> Double {
    switch min(max(weight, 100), 900) {
    case ...150: return -0.8
    case ...250: return -0.6
    case ...350: return -0.4
    case ...450: return 0
    case ...550: return 0.23
    case ...650: return 0.4
    case ...750: return 0.56
    case ...850: return 0.62
    default: return 0.8
    }
  }
}

public enum TextRasterizer {
  public static let characterCap = 8_192

  public static func layout(
    _ item: TextItem,
    layoutSize: (width: Int, height: Int)
  ) -> TextLayout {
    metrics(item, layoutSize: layoutSize).layout
  }

  public static func raster(
    _ item: TextItem,
    layoutSize: (width: Int, height: Int),
    pixelScale: Double
  ) -> (image: CGImage, layout: TextLayout)? {
    let prepared = metrics(item, layoutSize: layoutSize)
    let scale = max(pixelScale, 0.05)
    let pixelWidth = max(1, Int((prepared.layout.expandedSize.width * scale).rounded(.up)))
    let pixelHeight = max(1, Int((prepared.layout.expandedSize.height * scale).rounded(.up)))
    var data = [UInt8](repeating: 0, count: pixelWidth * pixelHeight * 4)
    guard
      let context = CGContext(
        data: &data,
        width: pixelWidth,
        height: pixelHeight,
        bitsPerComponent: 8,
        bytesPerRow: pixelWidth * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
          | CGBitmapInfo.byteOrder32Little.rawValue
      )
    else {
      return nil
    }
    context.scaleBy(x: scale, y: scale)
    let style = item.style
    let inset = outset(style)
    if style.backgroundColor.a > 0.001 {
      context.setFillColor(cgColor(style.backgroundColor))
      let rect = CGRect(
        x: 0,
        y: 0,
        width: prepared.layout.expandedSize.width,
        height: prepared.layout.expandedSize.height
      )
      let path = CGPath(
        roundedRect: rect,
        cornerWidth: min(style.backgroundCornerRadius, rect.width / 2),
        cornerHeight: min(style.backgroundCornerRadius, rect.height / 2),
        transform: nil
      )
      context.addPath(path)
      context.fillPath()
    }
    context.saveGState()
    context.translateBy(x: 0, y: prepared.layout.expandedSize.height)
    context.scaleBy(x: 1, y: -1)
    context.textMatrix = .identity
    context.translateBy(x: inset, y: inset)
    if style.shadowBlur > 0 || style.shadowOffsetX != 0 || style.shadowOffsetY != 0 {
      context.setShadow(
        offset: CGSize(width: style.shadowOffsetX, height: -style.shadowOffsetY),
        blur: style.shadowBlur,
        color: cgColor(style.shadowColor)
      )
    }
    CTFrameDraw(prepared.frame, context)
    if style.strokeWidth > 0 {
      context.setShadow(offset: .zero, blur: 0, color: nil)
      context.setStrokeColor(cgColor(style.strokeColor))
      context.setLineWidth(style.strokeWidth)
      context.setTextDrawingMode(.stroke)
      CTFrameDraw(prepared.frame, context)
    }
    context.restoreGState()
    guard let image = context.makeImage() else { return nil }
    return (image, prepared.layout)
  }

  private struct Prepared {
    var layout: TextLayout
    var frame: CTFrame
  }

  private static func metrics(
    _ item: TextItem,
    layoutSize: (width: Int, height: Int)
  ) -> Prepared {
    let text = clipped(item.text)
    let font = FontResolver.font(for: item.style)
    var alignment = ctAlignment(item.style.alignment)
    var lineHeight = fontSize(item.style) * CGFloat(item.style.lineHeight)
    let paragraph = withUnsafePointer(to: &alignment) { alignmentPtr in
      withUnsafePointer(to: &lineHeight) { linePtr in
        var settings = [
          CTParagraphStyleSetting(
            spec: .alignment,
            valueSize: MemoryLayout<CTTextAlignment>.size,
            value: alignmentPtr
          ),
          CTParagraphStyleSetting(
            spec: .minimumLineHeight,
            valueSize: MemoryLayout<CGFloat>.size,
            value: linePtr
          ),
        ]
        return CTParagraphStyleCreate(&settings, settings.count)
      }
    }
    let attributed = CFAttributedStringCreateMutable(nil, 0)!
    CFAttributedStringReplaceString(attributed, CFRange(location: 0, length: 0), text as CFString)
    let range = CFRange(location: 0, length: CFAttributedStringGetLength(attributed))
    CFAttributedStringSetAttribute(attributed, range, kCTFontAttributeName, font)
    CFAttributedStringSetAttribute(attributed, range, kCTForegroundColorAttributeName, cgColor(item.style.fill))
    if item.style.letterSpacing != 0 {
      var kern = CGFloat(item.style.letterSpacing)
      let number = CFNumberCreate(nil, .cgFloatType, &kern)
      CFAttributedStringSetAttribute(attributed, range, kCTKernAttributeName, number)
    }
    CFAttributedStringSetAttribute(attributed, range, kCTParagraphStyleAttributeName, paragraph)
    let setter = CTFramesetterCreateWithAttributedString(attributed)
    let layoutWidth = max(layoutSize.width, 1)
    let wrapWidth: CGFloat
    if item.style.wrap == .wrap {
      wrapWidth = CGFloat(item.style.boxWidth ?? 0.8 * Double(layoutWidth))
    } else {
      wrapWidth = CGFloat(layoutWidth) * 8
    }
    let suggested = CTFramesetterSuggestFrameSizeWithConstraints(
      setter,
      CFRange(location: 0, length: 0),
      nil,
      CGSize(width: wrapWidth, height: CGFloat.greatestFiniteMagnitude),
      nil
    )
    let content = CGSize(
      width: max(1, suggested.width.rounded(.up)),
      height: max(1, suggested.height.rounded(.up))
    )
    let path = CGPath(
      rect: CGRect(origin: .zero, size: content),
      transform: nil
    )
    let frame = CTFramesetterCreateFrame(
      setter,
      CFRange(location: 0, length: 0),
      path,
      nil
    )
    let inset = outset(item.style)
    let glyphs = glyphRects(frame: frame, contentHeight: Double(content.height))
    let lines = lineRects(frame: frame, contentHeight: Double(content.height))
    return Prepared(
      layout: TextLayout(
        expandedSize: (
          Double(content.width) + inset * 2,
          Double(content.height) + inset * 2
        ),
        lineFragments: lines,
        glyphBounds: glyphs
      ),
      frame: frame
    )
  }

  private static func glyphRects(frame: CTFrame, contentHeight: Double) -> [CanvasLayout.Rect] {
    guard let lines = CTFrameGetLines(frame) as? [CTLine] else { return [] }
    var origins = [CGPoint](repeating: .zero, count: lines.count)
    CTFrameGetLineOrigins(frame, CFRange(location: 0, length: 0), &origins)
    var bounds: [CanvasLayout.Rect] = []
    for (index, line) in lines.enumerated() {
      let origin = origins[index]
      let runs = CTLineGetGlyphRuns(line) as? [CTRun] ?? []
      for run in runs {
        let count = CTRunGetGlyphCount(run)
        var positions = [CGPoint](repeating: .zero, count: count)
        CTRunGetPositions(run, CFRange(location: 0, length: 0), &positions)
        var advances = [CGSize](repeating: .zero, count: count)
        CTRunGetAdvances(run, CFRange(location: 0, length: 0), &advances)
        for glyph in 0..<count {
          let width = max(advances[glyph].width, 1)
          let height = max(CTFontGetAscent(runFont(run)) + CTFontGetDescent(runFont(run)), 1)
          let x = Double(origin.x + positions[glyph].x)
          let yFromTop = Double(origin.y + positions[glyph].y)
          let y = contentHeight - yFromTop - Double(height)
          bounds.append(CanvasLayout.Rect(x: x, y: max(0, y), width: Double(width), height: Double(height)))
        }
      }
    }
    return bounds
  }

  private static func lineRects(frame: CTFrame, contentHeight: Double) -> [CanvasLayout.Rect] {
    guard let lines = CTFrameGetLines(frame) as? [CTLine] else { return [] }
    var origins = [CGPoint](repeating: .zero, count: lines.count)
    CTFrameGetLineOrigins(frame, CFRange(location: 0, length: 0), &origins)
    return zip(lines, origins).map { line, origin in
      let bounds = CTLineGetBoundsWithOptions(line, [.useOpticalBounds])
      let y = contentHeight - Double(origin.y + bounds.height)
      return CanvasLayout.Rect(
        x: Double(origin.x + bounds.minX),
        y: max(0, y),
        width: Double(max(bounds.width, 1)),
        height: Double(max(bounds.height, 1))
      )
    }
  }

  private static func runFont(_ run: CTRun) -> CTFont {
    let attributes = CTRunGetAttributes(run) as NSDictionary
    if let font = attributes[kCTFontAttributeName] {
      return font as! CTFont
    }
    return CTFontCreateUIFontForLanguage(.system, 12, nil)
      ?? CTFontCreateWithName("Helvetica" as CFString, 12, nil)
  }

  private static func outset(_ style: TextStyle) -> Double {
    style.backgroundPadding + style.strokeWidth
      + max(0, style.shadowBlur) + max(abs(style.shadowOffsetX), abs(style.shadowOffsetY))
  }

  private static func clipped(_ text: String) -> String {
    if text.utf16.count <= characterCap { return text }
    let end = text.utf16.index(text.utf16.startIndex, offsetBy: characterCap)
    return String(text.utf16[..<end]) ?? String(text.prefix(characterCap))
  }

  private static func fontSize(_ style: TextStyle) -> CGFloat {
    CGFloat(max(style.fontSize, 1))
  }

  private static func ctAlignment(_ alignment: TextAlignment) -> CTTextAlignment {
    switch alignment {
    case .leading: .left
    case .center: .center
    case .trailing: .right
    }
  }

  private static func cgColor(_ color: TextColor) -> CGColor {
    CGColor(
      red: color.r,
      green: color.g,
      blue: color.b,
      alpha: color.a
    )
  }
}
