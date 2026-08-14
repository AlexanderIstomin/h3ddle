import Foundation
import Testing

@testable import H3ddleEngineProtocol

@Suite("Engine JSON-lines protocol")
struct EngineProtocolTests {
  @Test("Commands round trip without losing job identity")
  func commandRoundTrip() throws {
    let jobID = UUID()
    let command = EngineCommand(
      requestID: UUID(),
      jobID: jobID,
      kind: .generate,
      generation: EngineGenerationRequest(
        kind: .video,
        prompt: "A quiet coastal road at dawn",
        duration: 5,
        outputURL: URL(fileURLWithPath: "/tmp/output.mp4")
      )
    )

    let encoded = try EngineLineCodec.encode(command)
    let decoded = try EngineLineCodec.decode(EngineCommand.self, from: encoded)

    #expect(encoded.last == 0x0A)
    #expect(decoded == command)
    #expect(decoded.jobID == jobID)
  }

  @Test("Generation requests carry an explicit quality preset")
  func generationQualityRoundTrip() throws {
    let request = EngineGenerationRequest(
      kind: .video,
      prompt: "A quiet coastal road at dawn",
      duration: 5,
      quality: .standard,
      outputURL: URL(fileURLWithPath: "/tmp/output.mp4")
    )

    let encoded = try EngineLineCodec.encode(request)
    let decoded = try EngineLineCodec.decode(EngineGenerationRequest.self, from: encoded)

    #expect(decoded.quality == .standard)
    #expect(
      EngineGenerationRequest(
        kind: .video,
        prompt: "p",
        duration: 1,
        outputURL: URL(fileURLWithPath: "/tmp/o.mp4")
      ).quality == .preview
    )
  }

  @Test("Denoising-step overrides are optional and clamped to the engine range")
  func denoisingStepOverride() throws {
    let output = URL(fileURLWithPath: "/tmp/output.mp4")
    let defaulted = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, outputURL: output
    )
    #expect(defaulted.denoisingSteps == nil)

    let clampedLow = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, denoisingSteps: 1, outputURL: output
    )
    #expect(clampedLow.denoisingSteps == 2)

    let clampedHigh = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, denoisingSteps: 5000, outputURL: output
    )
    #expect(clampedHigh.denoisingSteps == 1000)

    let request = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, denoisingSteps: 7, outputURL: output
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(request)
    )
    #expect(decoded.denoisingSteps == 7)
  }

  @Test("Layer and core-reuse overrides are optional and clamped to engine ranges")
  func layerAndCoreReuseOverrides() throws {
    let output = URL(fileURLWithPath: "/tmp/output.mp4")
    let defaulted = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, outputURL: output
    )
    #expect(defaulted.activeDiTLayers == nil)
    #expect(defaulted.coreReuse == nil)

    let clamped = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1,
      activeDiTLayers: 10, coreReuse: 9, outputURL: output
    )
    #expect(clamped.activeDiTLayers == 35)
    #expect(clamped.coreReuse == 6)

    let request = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1,
      activeDiTLayers: 45, coreReuse: 4, outputURL: output
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(request)
    )
    #expect(decoded.activeDiTLayers == 45)
    #expect(decoded.coreReuse == 4)
  }

  @Test("Quality presets stay on validated h3.c combinations")
  func qualityPresetTable() {
    #expect(EngineGenerationQuality.preview.canvasSize == 256)
    #expect(EngineGenerationQuality.preview.denoisingSteps == 4)
    #expect(EngineGenerationQuality.preview.activeDiTLayers == 50)
    #expect(EngineGenerationQuality.preview.denoiseReuse == 1)

    #expect(EngineGenerationQuality.standard.canvasSize == 512)
    #expect(EngineGenerationQuality.standard.denoisingSteps == 20)
    #expect(EngineGenerationQuality.standard.activeDiTLayers == 45)
    #expect(EngineGenerationQuality.standard.denoiseReuse == 2)

    #expect(EngineGenerationQuality.high.canvasSize == 768)
    #expect(EngineGenerationQuality.high.denoisingSteps == 20)
    #expect(EngineGenerationQuality.high.activeDiTLayers == 50)
    #expect(EngineGenerationQuality.high.denoiseReuse == 1)

    for quality in EngineGenerationQuality.allCases {
      // H3 canvases must be multiples of 32; reuse above 1 must never pair
      // with the minimum step budget, where every pass has to run the model.
      #expect(quality.canvasSize.isMultiple(of: 32))
      #expect(quality.denoisingSteps >= 20 || quality.denoiseReuse == 1)
    }
  }

  @Test("Audio generation requests round trip as soundtrack jobs")
  func audioGenerationRequest() throws {
    let request = EngineGenerationRequest(
      kind: .audio,
      prompt: "Soft rain",
      duration: 2,
      outputURL: URL(fileURLWithPath: "/tmp/rain.m4a")
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(request)
    )
    #expect(decoded.kind == .audio)
  }

  @Test("Image generation requests round trip as stills")
  func imageGenerationRequest() throws {
    let request = EngineGenerationRequest(
      kind: .image,
      prompt: "A red fox",
      duration: 4,
      outputURL: URL(fileURLWithPath: "/tmp/still.png")
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(request)
    )
    #expect(decoded.kind == .image)
    #expect(decoded.duration == 4)
  }

  @Test("Denoising preview defaults off and survives a round trip")
  func previewDenoiseRoundTrip() throws {
    let output = URL(fileURLWithPath: "/tmp/output.mp4")
    let defaulted = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, outputURL: output
    )
    #expect(!defaulted.previewDenoise)

    let enabled = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, previewDenoise: true, outputURL: output
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(enabled)
    )
    #expect(decoded.previewDenoise)

    let legacy = Data(
      """
      {"duration":1,"kind":"video","outputURL":"file:///tmp/o.mp4","prompt":"p","quality":"preview"}
      """.utf8
    )
    let decodedLegacy = try EngineLineCodec.decode(
      EngineGenerationRequest.self, from: legacy
    )
    #expect(!decodedLegacy.previewDenoise)
  }

  @Test("Beta schedule defaults off and survives a round trip")
  func betaScheduleRoundTrip() throws {
    let output = URL(fileURLWithPath: "/tmp/output.mp4")
    #expect(
      !EngineGenerationRequest(kind: .video, prompt: "p", duration: 1, outputURL: output)
        .useBetaSchedule
    )
    let enabled = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, useBetaSchedule: true, outputURL: output
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(enabled)
    )
    #expect(decoded.useBetaSchedule)

    let legacy = Data(
      """
      {"duration":1,"kind":"video","outputURL":"file:///tmp/o.mp4","prompt":"p","quality":"preview"}
      """.utf8
    )
    #expect(
      try !EngineLineCodec.decode(EngineGenerationRequest.self, from: legacy).useBetaSchedule
    )
  }

  @Test("Seeds default to nil and survive a round trip")
  func seedRoundTrip() throws {
    let output = URL(fileURLWithPath: "/tmp/output.mp4")
    #expect(
      EngineGenerationRequest(kind: .video, prompt: "p", duration: 1, outputURL: output)
        .seed == nil
    )
    let seeded = EngineGenerationRequest(
      kind: .video, prompt: "p", duration: 1, seed: 987_654_321, outputURL: output
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(seeded)
    )
    #expect(decoded.seed == 987_654_321)
  }

  @Test("Progress is clamped at the protocol boundary")
  func progressClamps() {
    let event = EngineEvent(
      requestID: UUID(),
      kind: .progress,
      fractionComplete: 1.4
    )

    #expect(event.fractionComplete == 1)
  }

  @Test("Capabilities distinguish embedded audio from audio-only output")
  func capabilitySemantics() throws {
    let capabilities = EngineCapabilities(
      engineName: "h3.c",
      engineVersion: "0.1",
      features: [.embeddedAudio, .videoGeneration, .embeddedAudio]
    )

    #expect(capabilities.features == [.videoGeneration, .embeddedAudio])
    #expect(capabilities.supports(.embeddedAudio))
    #expect(!capabilities.supports(.standaloneAudioGeneration))
  }

  @Test("Model reports use stable component ordering")
  func modelReportOrdering() {
    let directory = URL(fileURLWithPath: "/tmp/model", isDirectory: true)
    let report = EngineModelReport(
      modelDirectory: directory,
      components: [
        EngineModelComponent(
          kind: .audioVAE,
          bytes: 20,
          tensorBytes: 18,
          fileCount: 1,
          tensorCount: 2
        ),
        EngineModelComponent(
          kind: .textEncoder,
          bytes: 10,
          tensorBytes: 8,
          fileCount: 1,
          tensorCount: 2
        ),
      ],
      device: EngineDeviceReport(
        name: "Apple GPU",
        architecture: "Apple Silicon",
        physicalMemory: 32,
        recommendedWorkingSet: 24,
        unifiedMemory: true
      ),
      format: .optimizedINT8SingleFile,
      supportsGeneration: false
    )

    #expect(report.components.map(\.kind) == [.textEncoder, .audioVAE])
    #expect(report.totalBytes == 30)
    #expect(!report.hasReferenceTransformer)
    #expect(report.format == .optimizedINT8SingleFile)
    #expect(!report.supportsGeneration)
  }
}
