import Foundation
import H3ddleCore

public struct FakeGenerationProvider: GenerationProvider {
  private let stepDelay: Duration

  public init(stepDelay: Duration = .seconds(1)) {
    self.stepDelay = stepDelay
  }

  public func events(
    for request: GenerationRequest
  ) -> AsyncThrowingStream<GenerationEvent, any Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          guard !request.prompt.isEmpty else {
            throw GenerationError.emptyPrompt
          }

          let phases = ["Preparing", "Generating", "Finalizing"]
          for (index, phase) in phases.enumerated() {
            try await Task.sleep(for: stepDelay)
            try Task.checkCancellation()
            continuation.yield(
              .progress(
                phase: phase,
                fractionComplete: Double(index + 1) / Double(phases.count)
              )
            )
          }

          let duration = request.kind == .image ? max(request.duration, 3) : request.duration
          let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3ddle-fake-\(UUID().uuidString)")
            .appendingPathExtension(request.kind == .audio ? "wav" : request.kind == .image ? "png" : "mp4")
          if request.kind == .audio {
            try FakeAudioFixture.writeTone(to: url, duration: duration)
          }
          let asset = AssetReference(
            kind: request.kind.mediaKind,
            displayName: "Generated \(request.kind.rawValue.capitalized)",
            url: url,
            duration: duration
          )
          continuation.yield(.completed(asset))
          continuation.finish()
        } catch is CancellationError {
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }

      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }
}

enum FakeAudioFixture {
  static func writeTone(to url: URL, duration: TimeInterval) throws {
    let sampleRate = 22_050
    let sampleCount = max(1, Int((duration * Double(sampleRate)).rounded()))
    var data = Data()
    data.reserveCapacity(44 + sampleCount * 2)

    func appendUInt32(_ value: UInt32) {
      var value = value.littleEndian
      withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    func appendUInt16(_ value: UInt16) {
      var value = value.littleEndian
      withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }

    data.append(contentsOf: Array("RIFF".utf8))
    appendUInt32(UInt32(36 + sampleCount * 2))
    data.append(contentsOf: Array("WAVE".utf8))
    data.append(contentsOf: Array("fmt ".utf8))
    appendUInt32(16)
    appendUInt16(1)
    appendUInt16(1)
    appendUInt32(UInt32(sampleRate))
    appendUInt32(UInt32(sampleRate * 2))
    appendUInt16(2)
    appendUInt16(16)
    data.append(contentsOf: Array("data".utf8))
    appendUInt32(UInt32(sampleCount * 2))

    let twoPi = 2.0 * Double.pi
    for index in 0..<sampleCount {
      let sample = sin(twoPi * 220 * Double(index) / Double(sampleRate)) * 0.22
      let quantized = Int16((sample * Double(Int16.max)).rounded())
      appendUInt16(UInt16(bitPattern: quantized))
    }

    try data.write(to: url, options: .atomic)
  }
}
