import AppKit

/// What the Dock icon says about a generation that is running or has just
/// finished. A generation takes minutes, so the Dock is where the answer has
/// to be: the window is usually behind something else by then.
@MainActor
enum DockAttention {
  /// Last percentage drawn, so a run that reports progress many times a
  /// second redraws the tile only when the number would actually change.
  private static var shownPercent: Int?
  /// Whether a generation is in flight. Coming back to the window is not a
  /// reason to stop reporting one, and treating it as one is how the badge
  /// came to be invisible: activating the app cleared the tile, and a short
  /// run finished before the next progress event could put it back.
  private static var isRunning = false

  /// A percentage on the Dock icon while a generation runs.
  ///
  /// The badge rather than a drawn ring: at Dock size a ring reads as
  /// decoration, where "42%" answers the question the user actually walked
  /// away with. It also survives the icon being scaled down to the menu bar
  /// or the app switcher, which a thin arc does not.
  static func showProgress(_ fraction: Double) {
    let percent = Int((min(max(fraction, 0), 1) * 100).rounded())
    isRunning = true
    guard percent != shownPercent else { return }
    shownPercent = percent
    let tile = NSApp.dockTile
    tile.contentView = nil
    tile.badgeLabel = "\(percent)%"
    tile.display()
  }

  static func markGenerationFinished() {
    isRunning = false
    apply(showsDot: true)
    if !NSApp.isActive {
      NSApp.requestUserAttention(.informationalRequest)
    }
  }

  /// A run that was cancelled or failed: the percentage is stale and there is
  /// nothing to announce.
  static func markGenerationStopped() {
    isRunning = false
    apply(showsDot: false)
  }

  /// Called when the user comes back to the app. It dismisses the marker left
  /// by a finished run, and deliberately leaves a running one alone.
  static func dismissFinishedMarker() {
    guard !isRunning else { return }
    apply(showsDot: false)
  }

  static func clear() {
    isRunning = false
    apply(showsDot: false)
  }

  private static func apply(showsDot: Bool) {
    shownPercent = nil
    let tile = NSApp.dockTile
    guard showsDot else {
      tile.contentView = nil
      tile.badgeLabel = nil
      tile.display()
      return
    }

    // The percentage has served its purpose; leaving it beside the dot would
    // read as a run still at 100 rather than one that finished.
    tile.badgeLabel = nil
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
