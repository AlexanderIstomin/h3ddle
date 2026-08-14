import AppKit
import SwiftUI

/// Forwards only control-click / right-click so SwiftUI tap and drag stay intact.
struct SecondaryClickProbe: NSViewRepresentable {
  var onSecondaryClick: (CGPoint) -> Void

  func makeNSView(context: Context) -> Probe {
    let view = Probe()
    view.onSecondaryClick = onSecondaryClick
    return view
  }

  func updateNSView(_ view: Probe, context: Context) {
    view.onSecondaryClick = onSecondaryClick
  }

  final class Probe: NSView {
    var onSecondaryClick: ((CGPoint) -> Void)?

    override func hitTest(_ point: NSPoint) -> NSView? {
      guard let event = NSApp.currentEvent else { return nil }
      switch event.type {
      case .rightMouseDown, .rightMouseUp:
        return self
      case .leftMouseDown, .leftMouseUp:
        return event.modifierFlags.contains(.control) ? self : nil
      default:
        return nil
      }
    }

    override func rightMouseDown(with event: NSEvent) {
      deliver(event)
    }

    override func mouseDown(with event: NSEvent) {
      guard event.modifierFlags.contains(.control) else { return }
      deliver(event)
    }

    private func deliver(_ event: NSEvent) {
      onSecondaryClick?(convert(event.locationInWindow, from: nil))
    }
  }
}
