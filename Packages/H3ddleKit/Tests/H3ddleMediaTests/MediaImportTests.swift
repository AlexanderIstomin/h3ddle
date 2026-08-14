import Foundation
import H3ddleCore
import Testing

@testable import H3ddleMedia

@Suite("Media import")
struct MediaImportTests {
  @Test("Extensions map to the timeline kinds")
  func classifiesExtensions() {
    #expect(MediaImport.kind(for: URL(fileURLWithPath: "/tmp/still.PNG")) == .image)
    #expect(MediaImport.kind(for: URL(fileURLWithPath: "/tmp/clip.mp4")) == .video)
    #expect(MediaImport.kind(for: URL(fileURLWithPath: "/tmp/score.wav")) == .audio)
    #expect(MediaImport.kind(for: URL(fileURLWithPath: "/tmp/notes.txt")) == nil)
  }

  @Test("A still is imported at three seconds with no native audio")
  func imageProbe() async throws {
    let folder = try temporaryFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = folder.appendingPathComponent("photo.png")
    try ExportTestMedia.writePNG(to: source, width: 24, height: 16, red: 20, green: 80, blue: 160)

    let imported = try await MediaImport.makeAsset(
      from: source,
      onto: .visual,
      copyingInto: folder.appendingPathComponent("library", isDirectory: true)
    )
    #expect(imported.asset.kind == .image)
    #expect(imported.asset.duration == MediaImport.stillDuration)
    #expect(imported.asset.displayName == "photo")
    #expect(!imported.includesNativeAudio)
    #expect(imported.asset.url != source)
    #expect(FileManager.default.fileExists(atPath: imported.asset.url.path))
  }

  @Test("Audio duration is probed from the file")
  func audioProbe() async throws {
    let folder = try temporaryFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = folder.appendingPathComponent("tone.wav")
    try ExportTestMedia.writeTone(to: source, duration: 1.25)

    let imported = try await MediaImport.makeAsset(
      from: source,
      onto: .audio,
      copyingInto: folder.appendingPathComponent("library", isDirectory: true)
    )
    #expect(imported.asset.kind == .audio)
    #expect(abs(imported.asset.duration - 1.25) < 0.05)
    #expect(!imported.includesNativeAudio)
  }

  @Test("A still cannot be placed on the audio lane")
  func rejectsWrongLane() async throws {
    let folder = try temporaryFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = folder.appendingPathComponent("photo.png")
    try ExportTestMedia.writePNG(to: source, width: 8, height: 8, red: 1, green: 2, blue: 3)

    do {
      _ = try await MediaImport.makeAsset(
        from: source,
        onto: .audio,
        copyingInto: folder.appendingPathComponent("library", isDirectory: true)
      )
      Issue.record("Expected a wrong-lane error")
    } catch MediaImportError.wrongLane {
    } catch {
      Issue.record("Unexpected error \(error)")
    }
  }

  @Test("Drop partition keeps only files that belong on the lane")
  func partitionsDropPayload() {
    let image = URL(fileURLWithPath: "/tmp/still.png")
    let video = URL(fileURLWithPath: "/tmp/clip.mp4")
    let audio = URL(fileURLWithPath: "/tmp/score.wav")
    let text = URL(fileURLWithPath: "/tmp/notes.txt")

    let visual = MediaImport.partition([image, video, audio, text], onto: .visual)
    #expect(visual.accepted == [image, video])
    #expect(visual.rejected == [audio, text])
    #expect(MediaImport.containsCompatible([audio, image], onto: .visual))
    #expect(!MediaImport.containsCompatible([audio, text], onto: .visual))

    let audioLane = MediaImport.partition([image, audio], onto: .audio)
    #expect(audioLane.accepted == [audio])
    #expect(audioLane.rejected == [image])
  }

  @Test("Unsupported files are rejected")
  func rejectsUnsupported() async {
    do {
      _ = try await MediaImport.probe(URL(fileURLWithPath: "/tmp/notes.txt"))
      Issue.record("Expected an unsupported-type error")
    } catch MediaImportError.unsupportedType {
    } catch {
      Issue.record("Unexpected error \(error)")
    }
  }

  private func temporaryFolder() throws -> URL {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-import-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }
}