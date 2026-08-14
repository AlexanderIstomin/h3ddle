import Testing

@testable import H3ddleCore

@Suite("Canvas layout")
struct CanvasLayoutTests {
  @Test("Fit letterboxes and cover crops")
  func fitAndCover() {
    let fit = CanvasLayout.destination(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: VisualCanvasTransform(fit: .fit)
    )
    #expect(abs(fit.width - 200) < 0.000_1)
    #expect(abs(fit.height - 100) < 0.000_1)
    #expect(abs(fit.x) < 0.000_1)
    #expect(abs(fit.y - 50) < 0.000_1)

    let cover = CanvasLayout.destination(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: VisualCanvasTransform(fit: .cover)
    )
    #expect(abs(cover.width - 400) < 0.000_1)
    #expect(abs(cover.height - 200) < 0.000_1)
    #expect(abs(cover.x + 100) < 0.000_1)
    #expect(abs(cover.y) < 0.000_1)
  }

  @Test("A 90 degree turn swaps the fitted box")
  func rotationSwapsFittedBox() {
    let dest = CanvasLayout.destination(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: VisualCanvasTransform(fit: .fit, rotationTurns: 1)
    )
    #expect(abs(dest.width - 100) < 0.000_1)
    #expect(abs(dest.height - 200) < 0.000_1)
    #expect(abs(dest.x - 50) < 0.000_1)
    #expect(abs(dest.y) < 0.000_1)

    let draw = CanvasLayout.unrotatedSize(destination: dest, rotationTurns: 1)
    #expect(abs(draw.width - 200) < 0.000_1)
    #expect(abs(draw.height - 100) < 0.000_1)
  }

  @Test("Turns wrap into 0...3")
  func normalizesTurns() {
    #expect(CanvasLayout.normalizedTurns(4) == 0)
    #expect(CanvasLayout.normalizedTurns(-1) == 3)
    #expect(VisualCanvasTransform(rotationTurns: 5).rotationTurns == 1)
  }
}
