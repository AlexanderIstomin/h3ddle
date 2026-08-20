import Foundation
import Testing

@testable import H3ddleCore

@Suite("Canvas gesture math")
struct CanvasGestureMathTests {
  @Test("A 10 pt drag on a 200x100 monitor changes translation by (0.05, -0.10)")
  func moveIsOneToOneAtUnityMagnification() {
    let next = CanvasGestureMath.moved(
      origin: .identity,
      deltaProgramX: 10 / 200,
      deltaProgramY: -10 / 100
    )
    #expect(abs(next.translationX - 0.05) < 0.000_1)
    #expect(abs(next.translationY + 0.10) < 0.000_1)
  }

  @Test("A 10 pt view drag at mag 2 is half a monitor drag")
  func moveAccountsForMagnification() throws {
    let start = CanvasViewportMath.programPoint(
      viewPoint: (200, 150),
      viewSize: (400, 300),
      aspect: 16 / 9,
      padding: 32,
      magnification: 2,
      offset: (10, -4)
    )
    let dragged = CanvasViewportMath.programPoint(
      viewPoint: (210, 150),
      viewSize: (400, 300),
      aspect: 16 / 9,
      padding: 32,
      magnification: 2,
      offset: (10, -4)
    )
    let startPoint = try #require(start)
    let draggedPoint = try #require(dragged)
    let next = CanvasGestureMath.moved(
      origin: .identity,
      deltaProgramX: draggedPoint.x - startPoint.x,
      deltaProgramY: draggedPoint.y - startPoint.y
    )
    let monitor = CanvasViewportMath.monitorRect(
      viewSize: (400, 300),
      aspect: 16 / 9,
      padding: 32
    )
    #expect(abs(next.translationX - 10 / 2 / monitor.width) < 1e-6)
  }

  @Test("Viewport fixture: program (0.5, 0.5) is (210, 146) at mag 2")
  func viewportFixtureCenter() throws {
    let view = CanvasViewportMath.viewPoint(
      program: (0.5, 0.5),
      viewSize: (400, 300),
      aspect: 16 / 9,
      padding: 32,
      magnification: 2,
      offset: (10, -4)
    )
    #expect(abs(view.x - 210) < 1e-6)
    #expect(abs(view.y - 146) < 1e-6)

    let program = try #require(
      CanvasViewportMath.programPoint(
        viewPoint: view,
        viewSize: (400, 300),
        aspect: 16 / 9,
        padding: 32,
        magnification: 2,
        offset: (10, -4)
      )
    )
    #expect(abs(program.x - 0.5) < 1e-6)
    #expect(abs(program.y - 0.5) < 1e-6)
  }

  @Test("Viewport fixture corners invert")
  func viewportFixtureCorners() throws {
    let cases: [((Double, Double), (Double, Double))] = [
      ((0, 0), (-126, 335)),
      ((0, 1), (-126, -43)),
      ((1, 1), (546, -43)),
    ]
    for (program, expectedView) in cases {
      let view = CanvasViewportMath.viewPoint(
        program: program,
        viewSize: (400, 300),
        aspect: 16 / 9,
        padding: 32,
        magnification: 2,
        offset: (10, -4)
      )
      #expect(abs(view.x - expectedView.0) < 1e-6)
      #expect(abs(view.y - expectedView.1) < 1e-6)
      let recovered = try #require(
        CanvasViewportMath.programPoint(
          viewPoint: view,
          viewSize: (400, 300),
          aspect: 16 / 9,
          padding: 32,
          magnification: 2,
          offset: (10, -4)
        )
      )
      #expect(abs(recovered.x - program.0) < 1e-6)
      #expect(abs(recovered.y - program.1) < 1e-6)
    }
  }

  @Test("A point in the 32 pt pad is empty canvas")
  func paddingIsOutsideMonitor() {
    #expect(
      CanvasViewportMath.programPoint(
        viewPoint: (16, 150),
        viewSize: (400, 300),
        aspect: 16 / 9,
        padding: 32,
        magnification: 1,
        offset: (0, 0)
      ) == nil
    )
  }

  @Test("Unbounded conversion preserves points beyond the monitor for canvas gestures")
  func unboundedPointExtendsPastMonitor() throws {
    let point = try #require(
      CanvasViewportMath.unboundedProgramPoint(
        viewPoint: (16, 150),
        viewSize: (400, 300),
        aspect: 16 / 9,
        padding: 32,
        magnification: 1,
        offset: (0, 0)
      )
    )
    #expect(point.x < 0)
    #expect(abs(point.y - 0.5) < 1e-6)
  }

  @Test("Scale from top-right about bottom-left keeps the pivot")
  func scaleAboutOppositeCorner() {
    let origin = CanvasObjectTransform()
    let start = CanvasLayout.placed(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: origin
    )
    let pivot = CanvasLayout.projectedQuad(
      centerX: start.centerX,
      centerY: start.centerY,
      drawWidth: start.drawWidth,
      drawHeight: start.drawHeight,
      rotationRadians: start.rotationRadians
    )[0]
    let grab = CanvasLayout.projectedQuad(
      centerX: start.centerX,
      centerY: start.centerY,
      drawWidth: start.drawWidth,
      drawHeight: start.drawHeight,
      rotationRadians: start.rotationRadians
    )[2]
    let pointer = (
      x: pivot.x + (grab.x - pivot.x) * 2,
      y: pivot.y + (grab.y - pivot.y) * 2
    )
    let next = CanvasGestureMath.scaled(
      origin: origin,
      grab: .topRight,
      pointer: (pointer.x / 200, pointer.y / 200),
      canvasWidth: 200,
      canvasHeight: 200,
      sourceWidth: 200,
      sourceHeight: 100,
      aboutCenter: false
    )
    #expect(abs(next.scale - 2) < 0.02)
    let placed = CanvasLayout.placed(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: next
    )
    let newPivot = CanvasLayout.projectedQuad(
      centerX: placed.centerX,
      centerY: placed.centerY,
      drawWidth: placed.drawWidth,
      drawHeight: placed.drawHeight,
      rotationRadians: placed.rotationRadians
    )[0]
    #expect(hypot(newPivot.x - pivot.x, newPivot.y - pivot.y) < 0.5)
  }

  @Test("Command scale about center leaves translation unchanged")
  func scaleAboutCenterKeepsTranslation() {
    let origin = CanvasObjectTransform(translationX: 0.1, translationY: -0.05)
    let placed = CanvasLayout.placed(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: origin
    )
    let grab = CanvasLayout.projectedQuad(
      centerX: placed.centerX,
      centerY: placed.centerY,
      drawWidth: placed.drawWidth,
      drawHeight: placed.drawHeight,
      rotationRadians: placed.rotationRadians
    )[2]
    let pointer = (
      x: placed.centerX + (grab.x - placed.centerX) * 1.5,
      y: placed.centerY + (grab.y - placed.centerY) * 1.5
    )
    let next = CanvasGestureMath.scaled(
      origin: origin,
      grab: .topRight,
      pointer: (pointer.x / 200, pointer.y / 200),
      canvasWidth: 200,
      canvasHeight: 200,
      sourceWidth: 200,
      sourceHeight: 100,
      aboutCenter: true
    )
    #expect(abs(next.translationX - 0.1) < 1e-6)
    #expect(abs(next.translationY + 0.05) < 1e-6)
    #expect(abs(next.scale - 1.5) < 0.02)
  }

  @Test("Rotate 90 degrees about center leaves translation unchanged")
  func rotateNinetyAboutCenter() {
    let origin = CanvasObjectTransform()
    let placed = CanvasLayout.placed(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: origin
    )
    let start = (x: (placed.centerX + 40) / 200, y: placed.centerY / 200)
    // Clockwise 90° in y-up: (40, 0) → (0, -40)
    let pointer = (x: placed.centerX / 200, y: (placed.centerY - 40) / 200)
    let next = CanvasGestureMath.rotated(
      origin: origin,
      start: start,
      pointer: pointer,
      canvasWidth: 200,
      canvasHeight: 200,
      sourceWidth: 200,
      sourceHeight: 100,
      snapToIncrements: false
    )
    #expect(abs(next.rotationRadians - .pi / 2) < 0.001)
    #expect(abs(next.translationX) < 1e-6)
    #expect(abs(next.translationY) < 1e-6)

    let rotated = CanvasLayout.placed(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: next
    )
    #expect(abs(rotated.aabb.width - 100) < 0.001)
    #expect(abs(rotated.aabb.height - 200) < 0.001)
  }

  @Test("Shift snaps rotation to 15 degrees")
  func rotateSnapsToFifteen() {
    let origin = CanvasObjectTransform()
    let placed = CanvasLayout.placed(
      sourceWidth: 200,
      sourceHeight: 100,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: origin
    )
    let start = (x: (placed.centerX + 40) / 200, y: placed.centerY / 200)
    // About 20° clockwise → snaps to 15°.
    let angle = 20.0 * .pi / 180
    let pointer = (
      x: (placed.centerX + 40 * Foundation.cos(angle)) / 200,
      y: (placed.centerY - 40 * Foundation.sin(angle)) / 200
    )
    let next = CanvasGestureMath.rotated(
      origin: origin,
      start: start,
      pointer: pointer,
      canvasWidth: 200,
      canvasHeight: 200,
      sourceWidth: 200,
      sourceHeight: 100,
      snapToIncrements: true
    )
    #expect(abs(next.rotationRadians - .pi / 12) < 0.001)
  }

  @Test("Overlay scale skips media fit")
  func overlayScaleSkipsFit() {
    let origin = CanvasObjectTransform()
    let placed = CanvasLayout.overlayPlaced(
      sourceWidth: 40,
      sourceHeight: 20,
      canvasWidth: 200,
      canvasHeight: 200,
      transform: origin
    )
    let grab = (
      x: (placed.centerX + placed.drawWidth / 2) / 200,
      y: (placed.centerY + placed.drawHeight / 2) / 200
    )
    let pointer = (x: grab.x + 0.05, y: grab.y + 0.025)
    let fitted = CanvasGestureMath.scaled(
      origin: origin,
      grab: .topRight,
      pointer: pointer,
      canvasWidth: 200,
      canvasHeight: 200,
      sourceWidth: 40,
      sourceHeight: 20,
      aboutCenter: true,
      usesMediaFit: true
    )
    let overlay = CanvasGestureMath.scaled(
      origin: origin,
      grab: .topRight,
      pointer: pointer,
      canvasWidth: 200,
      canvasHeight: 200,
      sourceWidth: 40,
      sourceHeight: 20,
      aboutCenter: true,
      usesMediaFit: false
    )
    #expect(overlay.scale != fitted.scale)
    #expect(abs(overlay.translationX) < 1e-6)
    #expect(abs(overlay.translationY) < 1e-6)
  }
}
