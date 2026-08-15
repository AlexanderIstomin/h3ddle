import CoreVideo
import Foundation
import H3ddleCore
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import H3ddleMedia

@Suite("Program compositor")
struct ProgramCompositorTests {
  @Test("An empty frame fills the canvas with the background")
  func emptyFrameIsBackground() {
    let compositor = ProgramCompositor(width: 16, height: 12, background: (0.2, 0.4, 0.6))
    guard let buffer = compositor.pixelBuffer(placing: nil, transform: .identity) else {
      Issue.record("Expected a canvas buffer")
      return
    }
    #expect(CVPixelBufferGetWidth(buffer) == 16)
    #expect(CVPixelBufferGetHeight(buffer) == 12)
    let sample = rgb(buffer, x: 8, yFromBottom: 6)
    #expect(abs(Int(sample.r) - 51) < 2)
    #expect(abs(Int(sample.g) - 102) < 2)
    #expect(abs(Int(sample.b) - 153) < 2)
  }

  @Test("Fit letterboxes a wide still on a square canvas")
  func fitLetterboxes() throws {
    let fixture = try solidImage(width: 32, height: 16, red: 220, green: 40, blue: 20)
    defer { fixture.remove() }
    let compositor = ProgramCompositor(width: 32, height: 32, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: fixture.image,
        transform: VisualCanvasTransform(fit: .fit)
      )
    else {
      Issue.record("Expected a composed buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 2), r: 0, g: 0, b: 0))
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 30), r: 0, g: 0, b: 0))
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 16), r: 220, g: 40, b: 20))
  }

  @Test("makeImage snapshots a composed buffer")
  func makeImageSnapshotsBuffer() {
    let compositor = ProgramCompositor(width: 12, height: 8, background: (0.8, 0.1, 0.1))
    guard let buffer = compositor.pixelBuffer(placing: nil, transform: .identity) else {
      Issue.record("Expected a canvas buffer")
      return
    }
    let image = ProgramCompositor.makeImage(from: buffer)
    #expect(image?.width == 12)
    #expect(image?.height == 8)
  }

  @Test("Metal film grain and chroma kernels compile")
  func metalKernelsCompile() {
    if CompositorKernels.grain == nil || CompositorKernels.chroma == nil {
      Issue.record("Kernel compile failed: \(CompositorKernels.compileError ?? "unknown")")
    }
    #expect(CompositorKernels.grain != nil)
    #expect(CompositorKernels.chroma != nil)
  }

  @Test("A color grade exposure lift brightens the canvas")
  func colorGradeBrightens() async throws {
    let imageURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-grade-\(UUID().uuidString).png")
    try ExportTestMedia.writePNG(to: imageURL, width: 16, height: 16, red: 40, green: 40, blue: 40)
    defer { try? FileManager.default.removeItem(at: imageURL) }
    var project = H3ddleProject()
    let image = AssetReference(kind: .image, displayName: "Still", url: imageURL, duration: 1)
    project.addAsset(image)
    let item = try project.timeline.appendVisual(image)
    guard var effect = project.timeline.addVisualEffect(item.id, kind: .colorGrade) else {
      Issue.record("Expected an effect instance")
      return
    }
    effect.parameters["exposure"] = 1.5
    project.timeline.setVisualEffect(item.id, effect: effect)

    let frame = ProgramPreview.frame(at: 0.2, project: project)
    let compositor = ProgramCompositor(width: 16, height: 16, background: (0, 0, 0))
    guard let buffer = await compositor.pixelBuffer(for: frame) else {
      Issue.record("Expected a graded buffer")
      return
    }
    let sample = rgb(buffer, x: 8, yFromBottom: 8)
    #expect(Int(sample.r) > 50)
  }

  @Test("Film grain changes from one frame to the next")
  func filmGrainAnimates() async throws {
    let imageURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-grain-\(UUID().uuidString).png")
    try ExportTestMedia.writePNG(
      to: imageURL,
      width: 32,
      height: 32,
      red: 120,
      green: 120,
      blue: 120
    )
    defer { try? FileManager.default.removeItem(at: imageURL) }
    var project = H3ddleProject()
    let image = AssetReference(kind: .image, displayName: "Still", url: imageURL, duration: 2)
    project.addAsset(image)
    let item = try project.timeline.appendVisual(image)
    guard var effect = project.timeline.addVisualEffect(item.id, kind: .filmGrain) else {
      Issue.record("Expected a grain instance")
      return
    }
    effect.parameters["amount"] = 1
    project.timeline.setVisualEffect(item.id, effect: effect)

    let compositor = ProgramCompositor(width: 32, height: 32, background: (0, 0, 0))
    let first = ProgramPreview.frame(at: 0.0, project: project)
    let second = ProgramPreview.frame(at: 0.2, project: project)
    guard let a = await compositor.pixelBuffer(for: first),
      let b = await compositor.pixelBuffer(for: second)
    else {
      Issue.record("Expected grain buffers")
      return
    }
    var different = 0
    for x in stride(from: 4, to: 28, by: 3) {
      let left = rgb(a, x: x, yFromBottom: 16)
      let right = rgb(b, x: x, yFromBottom: 16)
      if left.r != right.r || left.g != right.g { different += 1 }
    }
    #expect(different > 0)
  }

  @Test("Chroma key punches a green still through to transparency")
  func chromaKeyRemovesGreen() async throws {
    let imageURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-chroma-\(UUID().uuidString).png")
    try ExportTestMedia.writePNG(
      to: imageURL,
      width: 16,
      height: 16,
      red: 30,
      green: 190,
      blue: 40
    )
    defer { try? FileManager.default.removeItem(at: imageURL) }
    var project = H3ddleProject()
    let image = AssetReference(kind: .image, displayName: "Green", url: imageURL, duration: 1)
    project.addAsset(image)
    let item = try project.timeline.appendVisual(image)
    guard var effect = project.timeline.addVisualEffect(item.id, kind: .chromaKey) else {
      Issue.record("Expected a chroma instance")
      return
    }
    effect.parameters["hue"] = 0
    effect.parameters["softness"] = 0.35
    project.timeline.setVisualEffect(item.id, effect: effect)

    let compositor = ProgramCompositor(
      width: 16,
      height: 16,
      background: .clear,
      fillsBackground: false
    )
    let frame = ProgramPreview.frame(at: 0.1, project: project)
    guard let buffer = await compositor.pixelBuffer(for: frame) else {
      Issue.record("Expected a keyed buffer")
      return
    }
    #expect(alpha(buffer, x: 8, yFromBottom: 8) < 40)
  }

  @Test("Cover crops a wide still so the canvas is filled")
  func coverFills() throws {
    let fixture = try solidImage(width: 32, height: 16, red: 20, green: 180, blue: 80)
    defer { fixture.remove() }
    let compositor = ProgramCompositor(width: 32, height: 32, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: fixture.image,
        transform: VisualCanvasTransform(fit: .cover)
      )
    else {
      Issue.record("Expected a composed buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 1, yFromBottom: 1), r: 20, g: 180, b: 80))
    #expect(isNear(rgb(buffer, x: 30, yFromBottom: 30), r: 20, g: 180, b: 80))
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 16), r: 20, g: 180, b: 80))
  }

  @Test("A 90 degree fit swaps the letterbox axis")
  func rotateFitsOnTheOtherAxis() throws {
    let fixture = try solidImage(width: 32, height: 16, red: 40, green: 80, blue: 220)
    defer { fixture.remove() }
    let compositor = ProgramCompositor(width: 32, height: 32, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: fixture.image,
        transform: VisualCanvasTransform(fit: .fit, rotationTurns: 1)
      )
    else {
      Issue.record("Expected a composed buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 2, yFromBottom: 16), r: 0, g: 0, b: 0))
    #expect(isNear(rgb(buffer, x: 30, yFromBottom: 16), r: 0, g: 0, b: 0))
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 16), r: 40, g: 80, b: 220))
  }

  @Test("Preview frames of a still use the same placement")
  func stillPreviewMatchesDirectPlacement() async throws {
    let folder = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-comp-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }

    let imageURL = folder.appendingPathComponent("still.png")
    try ExportTestMedia.writePNG(to: imageURL, width: 32, height: 16, red: 200, green: 10, blue: 10)
    var project = H3ddleProject()
    project.settings.apply(resolution: .extreme)
    let image = AssetReference(kind: .image, displayName: "Still", url: imageURL, duration: 1)
    project.addAsset(image)
    let item = try project.timeline.appendVisual(image)
    project.timeline.setVisualCanvasFit(item.id, .fit)

    let frame = ProgramPreview.frame(at: 0.2, project: project)
    let compositor = ProgramCompositor(width: 32, height: 32, background: (0, 0, 0))
    guard let buffer = await compositor.pixelBuffer(for: frame) else {
      Issue.record("Expected a preview buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 2), r: 0, g: 0, b: 0))
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 16), r: 200, g: 10, b: 10))
  }

  @Test("A clear background can stay transparent")
  func clearBackgroundIsTransparent() {
    let compositor = ProgramCompositor(
      width: 8,
      height: 8,
      background: .clear,
      fillsBackground: false
    )
    guard let buffer = compositor.pixelBuffer(placing: nil, transform: .identity) else {
      Issue.record("Expected a canvas buffer")
      return
    }
    #expect(alpha(buffer, x: 4, yFromBottom: 4) < 8)
  }

  @Test("Translation moves the fitted still")
  func translationMovesStill() throws {
    let fixture = try solidImage(width: 32, height: 16, red: 220, green: 40, blue: 20)
    defer { fixture.remove() }
    let compositor = ProgramCompositor(width: 32, height: 32, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: fixture.image,
        transform: CanvasObjectTransform(fit: .fit, translationY: 0.25)
      )
    else {
      Issue.record("Expected a translated buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 2), r: 220, g: 40, b: 20))
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 24), r: 0, g: 0, b: 0))
  }

  @Test("Uniform scale 2 fills a letterboxed still")
  func scaleFillsLetterbox() throws {
    let fixture = try solidImage(width: 32, height: 16, red: 20, green: 180, blue: 80)
    defer { fixture.remove() }
    let compositor = ProgramCompositor(width: 32, height: 32, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: fixture.image,
        transform: CanvasObjectTransform(fit: .fit, scale: 2)
      )
    else {
      Issue.record("Expected a scaled buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 2), r: 20, g: 180, b: 80))
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 30), r: 20, g: 180, b: 80))
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 16), r: 20, g: 180, b: 80))
  }

  @Test("A 45 degree fit keeps the center lit")
  func fortyFiveKeepsCenter() throws {
    let fixture = try solidImage(width: 32, height: 16, red: 40, green: 80, blue: 220)
    defer { fixture.remove() }
    let compositor = ProgramCompositor(width: 32, height: 32, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: fixture.image,
        transform: CanvasObjectTransform(fit: .fit, rotationRadians: .pi / 4)
      )
    else {
      Issue.record("Expected a rotated buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 16, yFromBottom: 16), r: 40, g: 80, b: 220))
  }

  @Test("Layout size defaults to the buffer so composeScale is 1")
  func layoutDefaultsToBuffer() {
    let compositor = ProgramCompositor(width: 64, height: 36, background: (0, 0, 0))
    #expect(compositor.layoutWidth == 64)
    #expect(compositor.layoutHeight == 36)
    #expect(abs(compositor.composeScale - 1) < 0.000_1)
  }

  @Test("An injected video frame is placed instead of the file")
  func injectedVideoFrameIsUsed() async throws {
    let fixture = try solidImage(width: 16, height: 16, red: 10, green: 200, blue: 40)
    defer { fixture.remove() }
    var project = H3ddleProject()
    let video = AssetReference(
      kind: .video,
      displayName: "Shot",
      url: URL(fileURLWithPath: "/tmp/missing-compositor.mp4"),
      duration: 2
    )
    project.addAsset(video)
    try project.timeline.appendVisual(video)
    let frame = ProgramPreview.frame(at: 0.2, project: project)
    let compositor = ProgramCompositor(width: 16, height: 16, background: (0, 0, 0))
    guard let buffer = await compositor.pixelBuffer(for: frame, videoFrame: fixture.image) else {
      Issue.record("Expected a composed buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 8, yFromBottom: 8), r: 10, g: 200, b: 40))
  }

  @Test("Dissolve 0 keeps the outgoing still")
  func dissolveStartIsOutgoing() throws {
    let outgoing = try solidImage(width: 16, height: 16, red: 220, green: 20, blue: 20)
    let incoming = try solidImage(width: 16, height: 16, red: 20, green: 20, blue: 220)
    defer {
      outgoing.remove()
      incoming.remove()
    }
    let compositor = ProgramCompositor(width: 16, height: 16, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: outgoing.image,
        transform: .identity,
        incoming: incoming.image,
        incomingTransform: .identity,
        progress: 0,
        kind: .dissolve
      )
    else {
      Issue.record("Expected a mix buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 8, yFromBottom: 8), r: 220, g: 20, b: 20))
  }

  @Test("A mid fade is near the background")
  func fadeMidpointIsBackground() throws {
    let outgoing = try solidImage(width: 16, height: 16, red: 220, green: 20, blue: 20)
    let incoming = try solidImage(width: 16, height: 16, red: 20, green: 20, blue: 220)
    defer {
      outgoing.remove()
      incoming.remove()
    }
    let compositor = ProgramCompositor(width: 16, height: 16, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: outgoing.image,
        transform: .identity,
        incoming: incoming.image,
        incomingTransform: .identity,
        progress: 0.5,
        kind: .fade
      )
    else {
      Issue.record("Expected a mix buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 8, yFromBottom: 8), r: 0, g: 0, b: 0))
  }

  @Test("A mid wipe keeps outgoing on the right")
  func wipeRevealsFromTheLeft() throws {
    let outgoing = try solidImage(width: 16, height: 16, red: 220, green: 20, blue: 20)
    let incoming = try solidImage(width: 16, height: 16, red: 20, green: 20, blue: 220)
    defer {
      outgoing.remove()
      incoming.remove()
    }
    let compositor = ProgramCompositor(width: 16, height: 16, background: (0, 0, 0))
    guard
      let buffer = compositor.pixelBuffer(
        placing: outgoing.image,
        transform: .identity,
        incoming: incoming.image,
        incomingTransform: .identity,
        progress: 0.5,
        kind: .wipe
      )
    else {
      Issue.record("Expected a mix buffer")
      return
    }
    #expect(isNear(rgb(buffer, x: 2, yFromBottom: 8), r: 20, g: 20, b: 220))
    #expect(isNear(rgb(buffer, x: 14, yFromBottom: 8), r: 220, g: 20, b: 20))
  }

  private func solidImage(
    width: Int,
    height: Int,
    red: UInt8,
    green: UInt8,
    blue: UInt8
  ) throws -> (image: CGImage, remove: () -> Void) {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-comp-src-\(UUID().uuidString).png")
    try ExportTestMedia.writePNG(
      to: url,
      width: width,
      height: height,
      red: red,
      green: green,
      blue: blue
    )
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      try? FileManager.default.removeItem(at: url)
      throw MediaExportError.failed("Could not load a compositor fixture.")
    }
    return (image, { try? FileManager.default.removeItem(at: url) })
  }

  private func rgb(_ buffer: CVPixelBuffer, x: Int, yFromBottom: Int) -> (
    r: UInt8, g: UInt8, b: UInt8
  ) {
    let pixel = bgra(buffer, x: x, yFromBottom: yFromBottom)
    return (pixel.r, pixel.g, pixel.b)
  }

  private func alpha(_ buffer: CVPixelBuffer, x: Int, yFromBottom: Int) -> UInt8 {
    bgra(buffer, x: x, yFromBottom: yFromBottom).a
  }

  private func bgra(_ buffer: CVPixelBuffer, x: Int, yFromBottom: Int) -> (
    r: UInt8, g: UInt8, b: UInt8, a: UInt8
  ) {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let offset = yFromBottom * bytesPerRow + x * 4
    return (base[offset + 2], base[offset + 1], base[offset], base[offset + 3])
  }

  private func isNear(
    _ sample: (r: UInt8, g: UInt8, b: UInt8),
    r: UInt8,
    g: UInt8,
    b: UInt8
  ) -> Bool {
    abs(Int(sample.r) - Int(r)) < 8
      && abs(Int(sample.g) - Int(g)) < 8
      && abs(Int(sample.b) - Int(b)) < 8
  }
}
