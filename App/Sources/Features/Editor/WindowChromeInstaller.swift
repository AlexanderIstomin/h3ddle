import AppKit
import SwiftUI

/// Keeps a real AppKit title bar for traffic lights and double-click zoom,
/// painted in the same chrome color as the editor header below it.
struct WindowChromeInstaller: NSViewRepresentable {
  func makeNSView(context: Context) -> NSView {
    let view = NSView()
    view.isHidden = true
    apply(to: view.window)
    return view
  }

  func updateNSView(_ view: NSView, context: Context) {
    apply(to: view.window)
  }

  private func apply(to window: NSWindow?) {
    guard let window else { return }
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.titlebarSeparatorStyle = .none
    window.styleMask.insert([.titled, .closable, .miniaturizable, .resizable])
    window.styleMask.remove(.fullSizeContentView)
    window.isMovableByWindowBackground = false
    window.backgroundColor = NSColor(
      red: 17 / 255,
      green: 20 / 255,
      blue: 26 / 255,
      alpha: 1
    )
  }
}
