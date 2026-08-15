import AppKit
import SwiftUI

struct CanvasPointerSample: Equatable {
  var viewPoint: CGPoint
  var option: Bool
  var shift: Bool
  var command: Bool
}

enum CanvasPointerEvent {
  case down(CanvasPointerSample)
  case dragged(CanvasPointerSample)
  case up(CanvasPointerSample)
  case doubleClick
  case rightClick(CGPoint)
}

/// Claims left-drag, Option-drag, and right-click. Returns nil over SwiftUI
/// chrome and for pinch / scroll so those stay on the SwiftUI layer.
struct CanvasInteractionProbe: NSViewRepresentable {
  var chromeRects: [CGRect]
  var onEvent: (CanvasPointerEvent) -> Void

  func makeNSView(context: Context) -> Probe {
    let view = Probe()
    view.chromeRects = chromeRects
    view.onEvent = onEvent
    return view
  }

  func updateNSView(_ view: Probe, context: Context) {
    view.chromeRects = chromeRects
    view.onEvent = onEvent
  }

  final class Probe: NSView {
    var chromeRects: [CGRect] = []
    var onEvent: ((CanvasPointerEvent) -> Void)?

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    override func hitTest(_ point: NSPoint) -> NSView? {
      if chromeRects.contains(where: { $0.insetBy(dx: -2, dy: -2).contains(point) }) {
        return nil
      }
      guard let event = NSApp.currentEvent else { return nil }
      switch event.type {
      case .rightMouseDown, .rightMouseUp:
        return self
      case .leftMouseDown, .leftMouseDragged, .leftMouseUp:
        return self
      default:
        return nil
      }
    }

    override func mouseDown(with event: NSEvent) {
      if event.modifierFlags.contains(.control) {
        onEvent?(.rightClick(convert(event.locationInWindow, from: nil)))
        return
      }
      if event.clickCount >= 2 {
        onEvent?(.doubleClick)
        return
      }
      onEvent?(.down(sample(event)))
    }

    override func mouseDragged(with event: NSEvent) {
      onEvent?(.dragged(sample(event)))
    }

    override func mouseUp(with event: NSEvent) {
      onEvent?(.up(sample(event)))
    }

    override func rightMouseDown(with event: NSEvent) {
      onEvent?(.rightClick(convert(event.locationInWindow, from: nil)))
    }

    private func sample(_ event: NSEvent) -> CanvasPointerSample {
      CanvasPointerSample(
        viewPoint: convert(event.locationInWindow, from: nil),
        option: event.modifierFlags.contains(.option),
        shift: event.modifierFlags.contains(.shift),
        command: event.modifierFlags.contains(.command)
      )
    }
  }
}
