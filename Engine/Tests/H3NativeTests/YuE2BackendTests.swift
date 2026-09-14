import Foundation
import H3Native
import Testing

@Suite("YuE2 native backend")
struct YuE2BackendTests {
  private func invoke(
    shaders: String? = "h3_shaders.metal", lyrics: String = "A little song",
    seconds: Double = 10, steps: Int32 = 32, output: String = "/missing-yue2-output.wav",
    cancel: Bool = false
  ) -> (status: Int32, duration: Double, error: String) {
    var error = [CChar](repeating: 0, count: 512)
    var duration = 99.0
    let callback: h3ddle_zimage_step? = cancel ? { _, _, _, _ in 0 } : nil
    let run: (UnsafePointer<CChar>?) -> Int32 = { pointer in
      h3ddle_yue2_generate(
        "/missing-yue2-package", pointer, "Acoustic folk", lyrics, seconds,
        steps, 42, output, callback, nil, &duration, &error, error.count)
    }
    let status = shaders.map { $0.withCString(run) } ?? run(nil)
    return (status, duration, String(decoding: error.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self))
  }

  @Test("Missing Metal cannot select an interpreter or CPU fallback")
  func requiresMetal() {
    let result = invoke(shaders: nil)
    #expect(result.status == 0)
    #expect(result.duration == 0)
    #expect(!result.error.isEmpty)
  }

  @Test("Unsupported solver counts fail before loading weights", arguments: [0, 1, 7, 24, 33, -1])
  func invalidSteps(_ steps: Int32) {
    let result = invoke(steps: steps)
    #expect(result.status == 0)
    #expect(result.error.contains("8/16/32"))
  }

  @Test("Invalid clip lengths fail before loading weights", arguments: [0.0, 361.0, Double.nan, Double.infinity])
  func invalidLength(_ seconds: Double) {
    let result = invoke(seconds: seconds)
    #expect(result.status == 0)
    #expect(result.error.contains("1–360"))
  }

  @Test("Instrumental and long requests reach model loading", arguments: [1.0, 31.0, 120.0, 360.0])
  func optionalLyrics(_ seconds: Double) {
    let result = invoke(lyrics: "", seconds: seconds, cancel: true)
    #expect(result.status == 0)
    #expect(result.duration == 0)
    #expect(result.error.isEmpty) // The loading callback cancelled a valid request.
  }

  @Test("Cancellation stops before model loading")
  func cancellation() {
    let result = invoke(cancel: true)
    #expect(result.status == 0)
    #expect(result.duration == 0)
    #expect(!result.error.contains("tokenizer"))
  }

  @Test("A pre-existing output is preserved")
  func preservesOutput() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
    let original = Data("existing audio".utf8)
    try original.write(to: file)
    defer { try? FileManager.default.removeItem(at: file) }
    let result = invoke(output: file.path)
    #expect(result.status == 0)
    #expect(result.error.contains("already exists"))
    #expect(try Data(contentsOf: file) == original)
  }
}
