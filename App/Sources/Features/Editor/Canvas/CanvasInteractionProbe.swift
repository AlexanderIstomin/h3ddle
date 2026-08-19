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
  var onEvent: (CanvasPointerEvent) -> Void
  var cursorAt: (CGPoint) -> NSCursor?

  func makeNSView(context: Context) -> Probe {
    let view = Probe()
    view.onEvent = onEvent
    view.cursorAt = cursorAt
    return view
  }

  func updateNSView(_ view: Probe, context: Context) {
    view.onEvent = onEvent
    view.cursorAt = cursorAt
  }

  final class Probe: NSView {
    var onEvent: ((CanvasPointerEvent) -> Void)?
    var cursorAt: ((CGPoint) -> NSCursor?)?
    private var pointerIsInside = false
    private var trackingArea: NSTrackingArea?

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    override func updateTrackingAreas() {
      super.updateTrackingAreas()
      if let trackingArea {
        removeTrackingArea(trackingArea)
      }
      let area = NSTrackingArea(
        rect: .zero,
        options: [.activeInKeyWindow, .inVisibleRect, .mouseEnteredAndExited, .mouseMoved],
        owner: self,
        userInfo: nil
      )
      addTrackingArea(area)
      trackingArea = area
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
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

    override func mouseEntered(with event: NSEvent) {
      pointerIsInside = true
      updateCursor(for: event)
    }

    override func mouseMoved(with event: NSEvent) {
      updateCursor(for: event)
    }

    override func mouseExited(with event: NSEvent) {
      pointerIsInside = false
      NSCursor.arrow.set()
    }

    private func updateCursor(for event: NSEvent) {
      let point = convert(event.locationInWindow, from: nil)
      (cursorAt?(point) ?? .arrow).set()
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
