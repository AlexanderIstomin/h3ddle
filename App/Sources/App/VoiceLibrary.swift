import Foundation

/// A voice the user has kept, so a clip is chosen once rather than at every
/// generation.
///
/// The clip is copied into the app's own storage rather than referenced where
/// it was found: a voice that stops working because a file moved is worse than
/// the copy, and a few seconds of speech is a few hundred kilobytes.
struct SavedVoice: Identifiable, Codable, Equatable {
  let id: UUID
  var name: String
  /// File name inside the voices directory, not a path: the container moves
  /// between builds and machines, and the name does not.
  var fileName: String

  init(id: UUID = UUID(), name: String, fileName: String) {
    self.id = id
    self.name = name
    self.fileName = fileName
  }
}

/// Where saved voices live and how they are added.
///
/// The engine takes a clip and derives the 1024 numbers it actually conditions
/// on. Those numbers could be stored instead — they are the whole voice, at
/// four kilobytes — but computing them needs the speech package loaded, and
/// keeping the clip means a voice survives a package that is not installed yet.
enum VoiceLibrary {
  static var directory: URL {
    URL.applicationSupportDirectory
      .appendingPathComponent("H3ddle", isDirectory: true)
      .appendingPathComponent("Voices", isDirectory: true)
  }

  static func url(for voice: SavedVoice) -> URL {
    directory.appendingPathComponent(voice.fileName, isDirectory: false)
  }

  /// Copies a clip in and returns the voice that names it, or nil when the
  /// copy fails — a voice pointing at nothing would fail later, at generation,
  /// where the cause is much less obvious.
  static func add(clip source: URL, named name: String) -> SavedVoice? {
    let id = UUID()
    let fileName = id.uuidString + "." + (source.pathExtension.isEmpty
      ? "wav" : source.pathExtension)
    let destination = directory.appendingPathComponent(fileName, isDirectory: false)
    do {
      try FileManager.default.createDirectory(
        at: directory, withIntermediateDirectories: true)
      try FileManager.default.copyItem(at: source, to: destination)
    } catch {
      return nil
    }
    return SavedVoice(id: id, name: name, fileName: fileName)
  }

  static func remove(_ voice: SavedVoice) {
    try? FileManager.default.removeItem(at: url(for: voice))
  }

  /// A name from the file, tidied: "morgan-read-2.wav" becomes "Morgan read 2".
  static func suggestedName(for clip: URL) -> String {
    let base = clip.deletingPathExtension().lastPathComponent
      .replacingOccurrences(of: "_", with: " ")
      .replacingOccurrences(of: "-", with: " ")
      .trimmingCharacters(in: .whitespaces)
    guard let first = base.first else { return "Voice" }
    return base.isEmpty ? "Voice" : first.uppercased() + base.dropFirst()
  }
}
