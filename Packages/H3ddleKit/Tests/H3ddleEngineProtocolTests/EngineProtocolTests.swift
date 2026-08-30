import Foundation
import Testing

@testable import H3ddleEngineProtocol
import H3ddleGeneration

@Suite("Engine JSON-lines protocol")
struct EngineProtocolTests {
  /// The request has hand-written CodingKeys and a hand-written decoder, so a
  /// new field can be added to the struct, compile, satisfy the memberwise
  /// initialiser, and still never cross the wire. That is exactly what
  /// happened to `image`: the service saw nil, fell through to H3's loader and
  /// failed with a model-layout error that said nothing about the real cause.
  /// Every optional settings group gets a round trip here for that reason.
  @Test("Per-model settings survive the wire, not just the initialiser")
  func optionalSettingsRoundTrip() throws {
    let request = EngineGenerationRequest(
      kind: .image,
      prompt: "A red apple on a wooden table",
      duration: 0,
      h3ModelProfile: .fastH3,
      canvasWidth: 1024,
      canvasHeight: 1024,
      speech: EngineSpeechOptions(temperature: 0.9),
      image: EngineImageOptions(model: .zImage, steps: 8),
      outputURL: URL(fileURLWithPath: "/tmp/apple.png"),
      checkpoint: EngineCheckpointOptions(
        fileURL: URL(fileURLWithPath: "/tmp/apple.h3ckpt"),
        fingerprint: String(repeating: "a", count: 64)
      )
    )

    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self, from: try EngineLineCodec.encode(request))

    #expect(decoded.image?.model == .zImage)
    #expect(decoded.h3ModelProfile == .fastH3)
    #expect(decoded.image?.steps == 8)
    #expect(decoded.speech?.temperature == 0.9)
    #expect(decoded.checkpoint?.fileURL.path == "/tmp/apple.h3ckpt")
    #expect(decoded.checkpoint?.fingerprint == String(repeating: "a", count: 64))
    #expect(decoded == request)
  }

  @Test("FastH3 model profile defaults safely and survives the wire")
  func fastH3ProfileRoundTrip() throws {
    let legacy = Data(
      #"{"kind":"video","prompt":"x","duration":1,"quality":"preview","fastStill":false,"blockCache":false,"previewDenoise":false,"useBetaSchedule":false,"outputURL":"file:///tmp/x.mp4"}"#.utf8)
    #expect(try JSONDecoder().decode(
      EngineGenerationRequest.self, from: legacy).h3ModelProfile == .standard)

    let request = EngineGenerationRequest(
      kind: .video, prompt: "x", duration: 5, denoisingSteps: 4,
      h3ModelProfile: .fastH3,
      outputURL: URL(fileURLWithPath: "/tmp/fasth3.mp4"))
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self, from: try EngineLineCodec.encode(request))
    #expect(decoded.h3ModelProfile == .fastH3)
    #expect(decoded == request)
  }

  @Test("FastH3 rejects plumbing previews outside its released shape")
  func fastH3PreviewContract() {
    #expect(!EngineFastH3PreviewContract.supports(width: 256, height: 256, frames: 124))
    #expect(!EngineFastH3PreviewContract.supports(width: 512, height: 512, frames: 22))
    #expect(EngineFastH3PreviewContract.supports(width: 832, height: 480, frames: 124))
    #expect(EngineFastH3PreviewContract.supports(width: 512, height: 512, frames: 362))
    #expect(!EngineFastH3PreviewContract.supports(width: 512, height: 512, frames: 379))
  }

  @Test("Video settings survive a round trip")
  func videoOptionsRoundTrip() throws {
    let request = EngineGenerationRequest(
      kind: .video,
      prompt: "a sailboat",
      duration: 2.708,
      canvasWidth: 512,
      canvasHeight: 512,
      video: EngineVideoOptions(model: .ltx, steps: 8),
      allowsLTXMemoryOvercommit: true,
      outputURL: URL(fileURLWithPath: "/tmp/clip.mp4")
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self, from: try EngineLineCodec.encode(request))
    #expect(decoded.video?.model == .ltx)
    #expect(decoded.video?.steps == 8)
    #expect(decoded.allowsLTXMemoryOvercommit)
    #expect(decoded == request)
  }

  @Test("H3 inpainting inputs survive a round trip")
  func videoInpaintingRoundTrip() throws {
    let inpainting = EngineVideoInpaintingOptions(
      sourceVideoURL: URL(fileURLWithPath: "/tmp/source.mov"),
      maskURL: URL(fileURLWithPath: "/tmp/mask.mov"),
      maskKind: .video,
      preserveSourceAudio: false
    )
    let request = EngineGenerationRequest(
      kind: .video,
      prompt: "replace the sign",
      duration: 5,
      referenceImageURLs: [URL(fileURLWithPath: "/tmp/sign.png")],
      video: EngineVideoOptions(model: .h3, inpainting: inpainting),
      outputURL: URL(fileURLWithPath: "/tmp/inpaint.mp4")
    )

    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self, from: try EngineLineCodec.encode(request))

    #expect(decoded.video?.inpainting == inpainting)
    #expect(decoded.video?.inpainting?.maskKind == .video)
    #expect(decoded.video?.inpainting?.preserveSourceAudio == false)
    #expect(decoded == request)
  }

  @Test("High-memory consent defaults to denied")
  func memoryOvercommitDefaultsToDenied() throws {
    let json = Data(
      #"{"kind":"video","prompt":"x","duration":1,"quality":"preview","fastStill":false,"blockCache":false,"previewDenoise":false,"useBetaSchedule":false,"outputURL":"file:///tmp/x.mp4"}"#.utf8)
    let decoded = try JSONDecoder().decode(EngineGenerationRequest.self, from: json)
    #expect(!decoded.allowsLTXMemoryOvercommit)
  }

  /// Absent settings mean H3, which is what `.video` meant before a second
  /// model existed — the same promise `.image` makes.
  @Test("A clip with no video settings still means H3")
  func absentVideoSettingsMeanH3() throws {
    let request = EngineGenerationRequest(
      kind: .video, prompt: "x", duration: 1,
      outputURL: URL(fileURLWithPath: "/tmp/x.mp4"))
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self, from: try EngineLineCodec.encode(request))
    #expect(decoded.video == nil)
  }

  /// The video VAE compresses 8x in time and cannot make the seven leading
  /// frames, so a clip is 8k+1 frames and nothing else. Rounding here rather
  /// than in the engine is what keeps the duration the app shows equal to the
  /// one it gets back.
  @Test("A duration rounds to a length the VAE can actually express")
  func videoFrameRounding() {
    #expect(EngineVideoOptions.frames(forSeconds: 0.708) == 17)
    #expect(EngineVideoOptions.frames(forSeconds: 2.708) == 65)
    #expect(EngineVideoOptions.frames(forSeconds: 4.042) == 97)
    // Never zero, and never a length the decoder would refuse.
    #expect(EngineVideoOptions.frames(forSeconds: 0) == 9)
    for seconds in stride(from: 0.1, through: 8.0, by: 0.1) {
      let frames = EngineVideoOptions.frames(forSeconds: seconds)
      #expect(frames >= 9)
      #expect((frames - 1) % EngineVideoOptions.frameStep == 0)
    }
  }

  /// Both sides have to be a multiple of the VAE's spatial factor, and that is
  /// the *only* rule — the model has no aspect ratio it insists on, and its own
  /// released example is 960x544 rather than a square.
  @Test("Only frames the video VAE can express are offered")
  func videoCanvasSupport() {
    #expect(EngineVideoOptions.supports(width: 960, height: 544))
    #expect(EngineVideoOptions.supports(width: 544, height: 960))
    #expect(EngineVideoOptions.supports(width: 512, height: 512))
    #expect(!EngineVideoOptions.supports(width: 500, height: 544))
    #expect(!EngineVideoOptions.supports(width: 960, height: 500))
    #expect(!EngineVideoOptions.supports(width: 0, height: 0))
    // Every tier the studio offers, at every aspect a project can be in, must
    // pass the engine's own rule. This is the check that would have caught a
    // "720p" that rendered 720 — which is not a multiple of 32.
    for tier in LTXResolution.allCases {
      for aspect in [16.0 / 9, 9.0 / 16, 1.0, 4.0 / 5, 3.0 / 2] {
        let frame = tier.frame(aspect: aspect)
        #expect(EngineVideoOptions.supports(width: frame.width,
                                            height: frame.height))
      }
    }
  }

  /// The tiers name a short edge and the aspect decides the rest, so one
  /// choice reads the same in a landscape project and a portrait one.
  @Test("A tier keeps its short edge and takes its shape from the project")
  func videoTierShape() {
    #expect(LTXResolution.p1080.frame(aspect: 16.0 / 9) == (1920, 1088))
    #expect(LTXResolution.p1080.frame(aspect: 9.0 / 16) == (1088, 1920))
    #expect(LTXResolution.p480.frame(aspect: 1.0) == (480, 480))
    // 720 and 1080 are not multiples of 32; the tiers keep the familiar name
    // and render the nearest frame the decoder can express.
    #expect(LTXResolution.p720.shortEdge == 704)
    #expect(LTXResolution.p1080.shortEdge == 1088)
  }

  /// The estimate protects unified memory before the native bridge allocates
  /// a complete float video or starts the 48-block sampler. This is the exact
  /// 10-second, landscape 720p, one-reference geometry from the reported run;
  /// it deliberately crosses a 16 GiB Mac's safe budget.
  @Test("LTX memory preflight scales with geometry and conditioning")
  func ltxMemoryPreflight() {
    let gibibyte: UInt64 = 1_073_741_824
    let baseline = EngineVideoOptions.estimatedLTXPeakMemoryBytes(
      width: 512, height: 512, frames: 65)
    let large = EngineVideoOptions.estimatedLTXPeakMemoryBytes(
      width: 1248, height: 704, frames: 241, conditioningPictures: 1)

    #expect(large > baseline * 2)
    #expect(large > EngineVideoOptions.safeLTXMemoryBudget(
      physicalMemory: 16 * gibibyte))
    #expect(large < EngineVideoOptions.safeLTXMemoryBudget(
      physicalMemory: 32 * gibibyte))
    #expect(
      EngineVideoOptions.estimatedLTXPeakMemoryBytes(
        width: 1248, height: 704, frames: 241, conditioningPictures: 2)
        > large)
  }

  /// Absent settings mean H3, which is what `.image` meant before a second
  /// model existed. A client that has never heard of v14 keeps working.
  @Test("A still with no image settings still means H3")
  func absentImageSettingsMeanH3() throws {
    let request = EngineGenerationRequest(
      kind: .image, prompt: "x", duration: 0,
      outputURL: URL(fileURLWithPath: "/tmp/x.png"))
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self, from: try EngineLineCodec.encode(request))
    #expect(decoded.image == nil)
  }

  @Test("Independent image and video packages require exclusive model memory")
  func independentModelsRequireExclusiveModelMemory() {
    let output = URL(fileURLWithPath: "/tmp/x.png")
    let zImage = EngineGenerationRequest(
      kind: .image, prompt: "x", duration: 0,
      image: EngineImageOptions(model: .zImage), outputURL: output)
    let h3 = EngineGenerationRequest(
      kind: .image, prompt: "x", duration: 0,
      image: EngineImageOptions(model: .h3), outputURL: output)
    let legacyH3 = EngineGenerationRequest(
      kind: .image, prompt: "x", duration: 0, outputURL: output)
    let ltx = EngineGenerationRequest(
      kind: .video, prompt: "x", duration: 1,
      video: EngineVideoOptions(model: .ltx),
      outputURL: URL(fileURLWithPath: "/tmp/x.mp4"))

    #expect(zImage.requiresExclusiveModelMemory)
    #expect(!h3.requiresExclusiveModelMemory)
    #expect(!legacyH3.requiresExclusiveModelMemory)
    #expect(ltx.requiresExclusiveModelMemory)
  }

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

  @Test("Frame anchors and reference images survive a round trip")
  func visualConditioningRoundTrip() throws {
    let request = EngineGenerationRequest(
      kind: .video,
      prompt: "Continue from the stills",
      duration: 1,
      firstFrameURL: URL(fileURLWithPath: "/tmp/start.png"),
      lastFrameURL: URL(fileURLWithPath: "/tmp/end.png"),
      outputURL: URL(fileURLWithPath: "/tmp/o.mp4")
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(request)
    )
    #expect(decoded.firstFrameURL?.path == "/tmp/start.png")
    #expect(decoded.lastFrameURL?.path == "/tmp/end.png")
    #expect(decoded.referenceImageURLs.isEmpty)

    let refs = EngineGenerationRequest(
      kind: .video,
      prompt: "Use Picture 1",
      duration: 1,
      referenceImageURLs: [URL(fileURLWithPath: "/tmp/a.png"), URL(fileURLWithPath: "/tmp/b.png")],
      outputURL: URL(fileURLWithPath: "/tmp/o.mp4")
    )
    let decodedRefs = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(refs)
    )
    #expect(decodedRefs.referenceImageURLs.map(\.lastPathComponent) == ["a.png", "b.png"])
  }

  /// A field that fails to encode reads as the default at the far end, and
  /// the default here is a full repaint — the picture silently ignored rather
  /// than an error anyone would notice.
  @Test("The strength of a source picture survives a round trip")
  func sourceStrengthRoundTrip() throws {
    let request = EngineGenerationRequest(
      kind: .image,
      prompt: "p",
      duration: 1,
      seed: 42,
      sourceStrength: 0.35,
      firstFrameURL: URL(fileURLWithPath: "/tmp/in.png"),
      outputURL: URL(fileURLWithPath: "/tmp/o.png")
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(request)
    )
    #expect(decoded.sourceStrength == 0.35)
    #expect(decoded.firstFrameURL?.lastPathComponent == "in.png")

    /* Absent is distinct from zero: nothing to work from, rather than a
     * picture to keep entirely. */
    let plain = EngineGenerationRequest(
      kind: .image, prompt: "p", duration: 1,
      outputURL: URL(fileURLWithPath: "/tmp/o.png"))
    let decodedPlain = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(plain)
    )
    #expect(decodedPlain.sourceStrength == nil)
  }

  @Test("Canvas size overrides survive a round trip")
  func canvasOverrideRoundTrip() throws {
    let request = EngineGenerationRequest(
      kind: .video,
      prompt: "p",
      duration: 1,
      canvasWidth: 1344,
      canvasHeight: 768,
      outputURL: URL(fileURLWithPath: "/tmp/o.mp4")
    )
    let decoded = try EngineLineCodec.decode(
      EngineGenerationRequest.self,
      from: EngineLineCodec.encode(request)
    )
    #expect(decoded.canvasWidth == 1344)
    #expect(decoded.canvasHeight == 768)
  }

  @Test("Quality presets stay on validated h3.c combinations")
  func qualityPresetTable() {
    #expect(EngineGenerationQuality.preview.canvasSize == 768)
    #expect(EngineGenerationQuality.preview.denoisingSteps == 4)
    #expect(EngineGenerationQuality.preview.activeDiTLayers == 50)
    #expect(EngineGenerationQuality.preview.denoiseReuse == 1)

    #expect(EngineGenerationQuality.standard.canvasSize == 768)
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
      // Work presets retain the native fallback. Studio resolution is carried
      // by explicit width and height, independently of these budgets.
      #expect(quality.canvasSize == H3NativeCanvas.shortEdge)
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

  @Test("Engine performance samples survive the wire")
  func performanceSampleRoundTrip() throws {
    let event = EngineEvent(
      requestID: UUID(),
      kind: .progress,
      performance: EnginePerformanceSample(physicalFootprintBytes: 12_345),
      phase: "denoise",
      fractionComplete: 0.5
    )

    let decoded = try EngineLineCodec.decode(
      EngineEvent.self,
      from: EngineLineCodec.encode(event)
    )
    #expect(decoded.performance?.physicalFootprintBytes == 12_345)
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
