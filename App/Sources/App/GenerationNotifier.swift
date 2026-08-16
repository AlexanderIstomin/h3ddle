import Foundation
import H3ddleGeneration
import UserNotifications
import os

/// Tells the user a generation has landed, because by then they are usually
/// in another app: a video takes minutes and there is nothing to watch.
///
/// Failures are announced too. Walking back to a window that has been idle
/// for ten minutes without saying why is worse than being interrupted.
///
/// macOS suppresses banners for the frontmost app of its own accord, so a
/// notification posted while the user is watching the run lands quietly in
/// Notification Centre rather than over the thing they are looking at.
@MainActor
enum GenerationNotifier {
  private static let log = Logger(subsystem: "com.h3ddle.app", category: "notifications")
  private static var askedForAuthorization = false

  /// Asked for on the first generation rather than at launch: a permission
  /// prompt before the user has done anything is the one they refuse.
  static func requestAuthorizationIfNeeded() {
    guard !askedForAuthorization, hostsNotifications else { return }
    askedForAuthorization = true
    UNUserNotificationCenter.current()
      .requestAuthorization(options: [.alert, .sound]) { granted, error in
        if let error {
          log.info("Notification authorization failed: \(error.localizedDescription, privacy: .public)")
        } else {
          log.info("Notification authorization granted: \(granted, privacy: .public)")
        }
      }
  }

  static func generationFinished(kind: GenerationKind, seconds: TimeInterval) {
    let noun =
      switch kind {
      case .video: "Video"
      case .image: "Image"
      case .audio: "Audio"
      }
    post(
      title: "\(noun) ready",
      body: "Generated in \(AppModel.formatElapsed(seconds))."
    )
  }

  static func generationFailed(message: String) {
    post(title: "Generation failed", body: message)
  }

  private static func post(title: String, body: String) {
    guard hostsNotifications else { return }
    let content = UNMutableNotificationContent()
    content.title = title
    content.body = body
    content.sound = .default
    // Immediate delivery: nil trigger fires as soon as it is scheduled.
    let request = UNNotificationRequest(
      identifier: UUID().uuidString, content: content, trigger: nil
    )
    UNUserNotificationCenter.current().add(request) { error in
      if let error {
        log.info("Could not post a notification: \(error.localizedDescription, privacy: .public)")
      }
    }
  }

  /// `UNUserNotificationCenter.current()` traps outright when the process has
  /// no bundle identifier, which is what a UI test runner driving the app
  /// under some configurations looks like. Checking costs nothing and turns a
  /// crash into silence.
  private static var hostsNotifications: Bool {
    Bundle.main.bundleIdentifier != nil
  }
}
