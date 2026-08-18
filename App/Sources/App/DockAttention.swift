import AppKit
import os

/// What the Dock icon says about a generation that is running or has just
/// finished. A generation takes minutes, so the Dock is where the answer has
/// to be: the window is usually behind something else by then.
///
/// The rule this file learned the hard way: **do not touch the tile unless
/// there is something to put on it.** Clearing a tile that was never written
/// to is not a no-op — it installs an empty custom tile over the one the
/// system had already filled with the app icon, and then neither the icon nor
/// a later badge appears. Everything below is gated on `hasWritten` for that
/// reason.
@MainActor
enum DockAttention {
  private static let log = Logger(subsystem: "com.h3ddle.app", category: "dock")

  /// Whether anything here has ever changed the tile. Until it has, the tile
  /// belongs to the system and is left alone.
  private static var hasWritten = false
  /// Last percentage drawn, so a run reporting many times a second redraws
  /// only when the number would actually change.
  private static var shownPercent: Int?
  /// Whether a generation is in flight. Coming back to the window is not a
  /// reason to stop reporting one.
  private static var isRunning = false

  /// A percentage on the Dock icon while a generation runs.
  ///
  /// The badge rather than a drawn ring: at Dock size a ring reads as
  /// decoration, where "42%" answers the question the user actually walked
  /// away with, and it survives being scaled into the app switcher.
  static func showProgress(_ fraction: Double) {
    let percent = Int((min(max(fraction, 0), 1) * 100).rounded())
    isRunning = true
    guard percent != shownPercent else { return }
    shownPercent = percent
    let tile = NSApp.dockTile
    if tile.contentView != nil {
      tile.contentView = nil
    }
    tile.badgeLabel = "\(percent)%"
    // `badgeLabel` changes the tile's state, but the Dock is not required to
    // repaint that state until asked. Redraw after both the custom view and
    // label have reached their final values so the percentage appears on the
    // first update as well as after a finished-marker view.
    tile.display()
    hasWritten = true
    log.notice("Dock badge -> \(percent, privacy: .public)%")
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

  /// Called when the user comes back to the app. Dismisses the marker a
  /// finished run left, and deliberately leaves a running one alone.
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
    // Nothing has been drawn, so there is nothing to undo, and undoing it
    // anyway is what cost the app its Dock icon.
    guard showsDot || hasWritten else { return }
    let tile = NSApp.dockTile
    guard showsDot else {
      hasWritten = false
      tile.badgeLabel = nil
      // Order matters: drop the custom view first, then let the tile fall
      // back to the application icon on its own.
      if tile.contentView != nil {
        tile.contentView = nil
      }
      tile.display()
      log.notice("Dock tile released back to the system")
      return
    }

    // The percentage has served its purpose; leaving it beside the dot would
    // read as a run still at 100 rather than one that finished.
    tile.badgeLabel = nil
    let view = DockIconDotView(frame: NSRect(x: 0, y: 0, width: 128, height: 128))
    view.icon = NSApp.applicationIconImage
    tile.contentView = view
    tile.display()
    hasWritten = true
    log.notice("Dock tile shows the finished marker")
  }
}

private final class DockIconDotView: NSView {
  var icon: NSImage?

  override func draw(_ dirtyRect: NSRect) {
    // Without the icon there is nothing to badge, and drawing the dot alone
    // would replace the app's icon with a red circle on nothing.
    guard let icon else { return }
    icon.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1)
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
