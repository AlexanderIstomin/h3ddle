import AppKit
import CoreGraphics
import CoreVideo
import H3ddleCore
import H3ddleMedia
import Observation

@MainActor
@Observable
final class ProgramFramePresenter {
  var image: NSImage?

  private var compositor: ProgramCompositor
  private var queued: Request?
  private var isRendering = false
  private var visualBuffer: CVPixelBuffer?
  private var renderGeneration = 0

  init() {
    compositor = ProgramCompositor(width: 2, height: 2)
  }

  func render(
    frame: ProgramPreviewFrame,
    canvas: CGSize,
    scale: CGFloat,
    background: ProjectBackground,
    videoFrame: CGImage?,
    layoutWidth: Int? = nil,
    layoutHeight: Int? = nil
  ) {
    let raster = Self.raster(size: canvas, scale: scale)
    guard raster.width > 1, raster.height > 1 else { return }
    prepareCompositor(
      raster: raster,
      background: background,
      layoutWidth: layoutWidth,
      layoutHeight: layoutHeight
    )
    queued = Request(frame: frame, videoFrame: videoFrame)
    startIfNeeded()
  }

  @discardableResult
  func blitOverlays(_ overlays: [ProgramTextPresentation]) -> Bool {
    renderGeneration += 1
    queued = nil
    guard let visual = visualBuffer,
      let composed = compositor.addingOverlays(overlays, onto: visual)
    else {
      return false
    }
    apply(ProgramCompositor.makeImage(from: composed))
    return true
  }

  func clear() {
    queued = nil
    renderGeneration += 1
    visualBuffer = nil
    image = nil
  }

  private func prepareCompositor(
    raster: (width: Int, height: Int),
    background: ProjectBackground,
    layoutWidth: Int?,
    layoutHeight: Int?
  ) {
    let fillsBackground = !background.isClear
    let color = background.compositorColor
    let layoutW = layoutWidth ?? raster.width
    let layoutH = layoutHeight ?? raster.height
    if compositor.width != raster.width || compositor.height != raster.height
      || compositor.layoutWidth != layoutW || compositor.layoutHeight != layoutH
      || compositor.backgroundAlpha != (fillsBackground ? 1 : 0)
      || compositor.background != color
    {
      compositor = ProgramCompositor(
        width: raster.width,
        height: raster.height,
        background: background,
        fillsBackground: fillsBackground,
        layoutWidth: layoutW,
        layoutHeight: layoutH
      )
      visualBuffer = nil
    }
  }

  private func startIfNeeded() {
    guard !isRendering, let request = queued else { return }
    queued = nil
    isRendering = true
    let token = renderGeneration
    let compositor = compositor
    var visualOnly = request.frame
    visualOnly.overlays = []
    let overlays = request.frame.overlays
    Task.detached {
      let visual = await compositor.pixelBuffer(
        for: visualOnly,
        videoFrame: request.videoFrame
      )
      let composed = visual.flatMap { compositor.addingOverlays(overlays, onto: $0) } ?? visual
      let next = composed.flatMap(ProgramCompositor.makeImage)
      await MainActor.run {
        if token == self.renderGeneration {
          self.visualBuffer = visual
          self.apply(next)
        }
        self.isRendering = false
        self.startIfNeeded()
      }
    }
  }

  private func apply(_ image: CGImage?) {
    guard let image else { return }
    self.image = NSImage(
      cgImage: image,
      size: NSSize(width: image.width, height: image.height)
    )
  }

  private struct Request {
    var frame: ProgramPreviewFrame
    var videoFrame: CGImage?
  }

  static func raster(size: CGSize, scale: CGFloat) -> (width: Int, height: Int) {
    let pixelWidth = max(size.width, 0) * max(scale, 1)
    let pixelHeight = max(size.height, 0) * max(scale, 1)
    let longest = max(pixelWidth, pixelHeight)
    let cap: CGFloat = 1_920
    let factor = longest > cap ? cap / longest : 1
    var width = Int((pixelWidth * factor).rounded(.toNearestOrAwayFromZero))
    var height = Int((pixelHeight * factor).rounded(.toNearestOrAwayFromZero))
    if !width.isMultiple(of: 2) { width += 1 }
    if !height.isMultiple(of: 2) { height += 1 }
    if width < 2 || height < 2 { return (1_280, 720) }
    return (width, height)
  }
}
