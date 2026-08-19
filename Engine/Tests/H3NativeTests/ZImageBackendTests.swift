import H3Native
import Testing

@Suite("Z-Image native backend")
struct ZImageBackendTests {
  private func invoke(shaders: UnsafePointer<CChar>?) -> (status: Int32, error: String) {
    var rgb = [UInt8](repeating: 0, count: 3 * 128 * 128)
    var error = [CChar](repeating: 0, count: 256)
    let status = "/missing-zimage-package".withCString { package in
      "a test prompt".withCString { prompt in
        rgb.withUnsafeMutableBufferPointer { output in
          error.withUnsafeMutableBufferPointer { problem in
            h3ddle_zimage_generate(
              package, shaders, prompt, 128, 128, 1, 42,
              nil, 1.0, 0, output.baseAddress,
              nil, nil, nil, problem.baseAddress, problem.count)
          }
        }
      }
    }
    let message = error.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
    return (status, String(decoding: message, as: UTF8.self))
  }

  @Test("Production generation rejects a missing Metal backend")
  func missingShadersCannotSelectCPU() {
    let result = invoke(shaders: nil)
    #expect(result.status == 0)
    #expect(result.error ==
      "Z-Image generation requires Metal shaders; CPU fallback is disabled")
  }

  @Test("An empty shader path cannot select CPU either")
  func emptyShadersCannotSelectCPU() {
    let result = "".withCString { invoke(shaders: $0) }
    #expect(result.status == 0)
    #expect(result.error ==
      "Z-Image generation requires Metal shaders; CPU fallback is disabled")
  }
}
