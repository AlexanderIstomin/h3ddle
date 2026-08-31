import Foundation
import H3ddleCore
import Testing

@testable import H3ddleMedia

@Suite("Embedded generation metadata")
struct EmbeddedGenerationMetadataTests {
  private let payload = Data(#"{"version":1,"prompt":"a woman"}"#.utf8)

  @Test("PNG iTXt metadata round-trips without changing pixels")
  func pngRoundTrip() throws {
    let url = temporaryURL(extension: "png")
    defer { try? FileManager.default.removeItem(at: url) }
    let png = Data(
      base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )!
    try png.write(to: url)

    try EmbeddedGenerationMetadata.write(payload, to: url)

    #expect(try EmbeddedGenerationMetadata.read(from: url) == payload)
    #expect((try Data(contentsOf: url)).prefix(8) == png.prefix(8))
  }

  @Test("WAV private metadata chunk round-trips")
  func wavRoundTrip() throws {
    let url = temporaryURL(extension: "wav")
    defer { try? FileManager.default.removeItem(at: url) }
    try Data([
      0x52, 0x49, 0x46, 0x46, 0x04, 0x00, 0x00, 0x00,
      0x57, 0x41, 0x56, 0x45,
    ]).write(to: url)

    try EmbeddedGenerationMetadata.write(payload, to: url)

    #expect(try EmbeddedGenerationMetadata.read(from: url) == payload)
  }

  @Test("MP4 UUID metadata box round-trips without remuxing")
  func mp4RoundTrip() throws {
    let url = temporaryURL(extension: "mp4")
    defer { try? FileManager.default.removeItem(at: url) }
    // One small, structurally valid top-level ftyp-shaped box is sufficient
    // for exercising the lossless top-level box scanner.
    try Data([0, 0, 0, 8, 0x66, 0x74, 0x79, 0x70]).write(to: url)

    try EmbeddedGenerationMetadata.write(payload, to: url)

    #expect(try EmbeddedGenerationMetadata.read(from: url) == payload)
    #expect((try Data(contentsOf: url))[4..<8] == Data("ftyp".utf8))
  }

  private func temporaryURL(extension ext: String) -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-metadata-\(UUID().uuidString)")
      .appendingPathExtension(ext)
  }
}
