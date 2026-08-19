import AppKit

/// Displays generation progress and completion state on the Dock icon.
///
/// The rule this file learned the hard way: **do not touch the tile unless
/// there is something to put on it.** Clearing a tile that was never written
/// to is not a no-op — it installs an empty custom tile over the one the
/// system had already filled with the app icon, and then neither the icon nor
/// a later badge appears. Everything below is gated on `hasWritten` for that
/// reason.
@MainActor
enum DockAttention {
  /// Whether anything here has ever changed the tile. Until it has, the tile
  /// belongs to the system and is left alone.
  private static var hasWritten = false
  /// Last percentage drawn, so frequent engine events only redraw the Dock
  /// when the visible value changes.
  private static var shownPercent: Int?
  /// Whether a generation is in flight. Coming back to the window should not
  /// clear state belonging to that run.
  private static var isRunning = false

  static func showProgress(_ fraction: Double) {
    let percent = Int((min(max(fraction, 0), 1) * 100).rounded())
    isRunning = true
    guard percent != shownPercent else { return }
    shownPercent = percent

    let tile = NSApp.dockTile
    let view: DockIconProgressView
    if let current = tile.contentView as? DockIconProgressView {
      view = current
    } else {
      view = DockIconProgressView(frame: NSRect(origin: .zero, size: tile.size))
      view.autoresizingMask = [.width, .height]
      view.icon = NSApp.applicationIconImage
      tile.contentView = view
    }
    view.percent = percent
    tile.badgeLabel = nil
    tile.display()
    hasWritten = true
  }

  static func markGenerationFinished() {
    isRunning = false
    apply(showsDot: true)
    if !NSApp.isActive {
      NSApp.requestUserAttention(.informationalRequest)
    }
  }

  /// A run that was cancelled or failed has nothing to announce.
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
      return
    }

    tile.badgeLabel = nil
    let view = DockIconDotView(frame: NSRect(x: 0, y: 0, width: 128, height: 128))
    view.icon = NSApp.applicationIconImage
    tile.contentView = view
    tile.display()
    hasWritten = true
  }
}

private final class DockIconProgressView: NSView {
  var icon: NSImage?
  var percent = 0

  override func draw(_ dirtyRect: NSRect) {
    guard let icon else { return }
    icon.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1)

    let scale = bounds.width / 128
    let font = NSFont.monospacedDigitSystemFont(
      ofSize: max(10, 20 * scale),
      weight: .bold
    )
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = .center
    let attributes: [NSAttributedString.Key: Any] = [
      .font: font,
      .foregroundColor: NSColor.white,
      .paragraphStyle: paragraph,
    ]
    let label = "\(percent)%" as NSString
    let textSize = label.size(withAttributes: attributes)
    let verticalPadding = max(2, 3 * scale)
    let horizontalPadding = max(4, 7 * scale)
    let height = textSize.height + verticalPadding * 2
    let width = max(height, textSize.width + horizontalPadding * 2)
    let inset = max(2, 5 * scale)
    let badge = NSRect(
      x: bounds.maxX - width - inset,
      y: bounds.maxY - height - inset,
      width: width,
      height: height
    )

    NSColor(red: 229 / 255, green: 72 / 255, blue: 77 / 255, alpha: 1).setFill()
    NSBezierPath(roundedRect: badge, xRadius: height / 2, yRadius: height / 2).fill()
    label.draw(
      in: NSRect(
        x: badge.minX,
        y: badge.midY - textSize.height / 2,
        width: badge.width,
        height: textSize.height
      ),
      withAttributes: attributes
    )
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
