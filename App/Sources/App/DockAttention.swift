import AppKit

@MainActor
enum DockAttention {
  static func markGenerationFinished() {
    apply(showsDot: true)
    if !NSApp.isActive {
      NSApp.requestUserAttention(.informationalRequest)
    }
  }

  static func clear() {
    apply(showsDot: false)
  }

  private static func apply(showsDot: Bool) {
    let tile = NSApp.dockTile
    guard showsDot else {
      tile.contentView = nil
      tile.badgeLabel = nil
      tile.display()
      return
    }

    let view = DockIconDotView(frame: NSRect(x: 0, y: 0, width: 128, height: 128))
    view.icon = NSApp.applicationIconImage
    tile.contentView = view
    tile.display()
  }
}

private final class DockIconDotView: NSView {
  var icon: NSImage?

  override func draw(_ dirtyRect: NSRect) {
    icon?.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1)
    let diameter = bounds.width * 0.22
    let inset = bounds.width * 0.06
    let rect = NSRect(
      x: bounds.maxX - diameter - inset,
      y: bounds.maxY - diameter - inset,
      width: diameter,
      height: diameter
    )
    NSColor(red: 229 / 255, green: 72 / 255, blue: 77 / 255, alpha: 1).setFill()
    NSBezierPath(ovalIn: rect).fill()
    NSColor.white.withAlphaComponent(0.22).setStroke()
    let stroke = NSBezierPath(ovalIn: rect.insetBy(dx: 0.5, dy: 0.5))
    stroke.lineWidth = 1
    stroke.stroke()
  }
}
