import Foundation
import Testing

@testable import H3ddleGeneration

@Suite("Generation studio settings")
struct GenerationStudioSettingsTests {
  @Test("Named presets write the documented knob combinations")
  func namedPresets() {
    var settings = GenerationStudioSettings.makeDefault(seed: 1)
    settings.apply(preset: .standard)
    #expect(settings.preset == .standard)
    #expect(settings.knobs.canvas == .p480)
    #expect(settings.knobs.denoisingSteps == 20)
    #expect(settings.knobs.activeDiTLayers == 45)
  }

  @Test("Editing a knob selects Custom and remembers the snapshot")
  func editingSelectsCustom() {
    var settings = GenerationStudioSettings.makeDefault(seed: 1)
    settings.apply(preset: .preview)
    settings.updateKnobs { $0.denoisingSteps = 8 }
    #expect(settings.preset == .custom)
    #expect(settings.custom.denoisingSteps == 8)

    settings.apply(preset: .high)
    #expect(settings.knobs.canvas == .p768)
    #expect(settings.knobs.denoisingSteps == 20)

    settings.apply(preset: .custom)
    #expect(settings.knobs.denoisingSteps == 8)
    #expect(settings.knobs.canvas == .p352)
  }

  @Test("Matching a named preset snaps back to that preset")
  func matchingNamedPreset() {
    var settings = GenerationStudioSettings.makeDefault(seed: 1)
    settings.updateKnobs { $0.denoisingSteps = 9 }
    settings.updateKnobs { $0.denoisingSteps = 4 }
    #expect(settings.preset == .preview)
  }

  @Test("Native landscape flips for portrait")
  func nativeLandscapeFollowsOrientation() {
    #expect(GenerationCanvas.p1088.dimensions(aspect: 16.0 / 9) == (1920, 1088))
    #expect(GenerationCanvas.p1088.dimensions(aspect: 9.0 / 16) == (1088, 1920))
  }
}

@Suite("Canvas ladder")
struct GenerationCanvasTests {
  @Test("Tiers reproduce the published resolution table at 16:9")
  func matchesReferenceTable() {
    let wide = 16.0 / 9
    // 352p is the one row that differs: the published table targets
    // megapixels and lands on 608x352 (1.73:1), while fixing the short edge
    // gives 640x352 (1.82:1), which is nearer true 16:9. Both are legal.
    #expect(GenerationCanvas.p352.dimensions(aspect: wide) == (640, 352))
    #expect(GenerationCanvas.p480.dimensions(aspect: wide) == (864, 480))
    #expect(GenerationCanvas.p576.dimensions(aspect: wide) == (1024, 576))
    #expect(GenerationCanvas.p768.dimensions(aspect: wide) == (1376, 768))
    #expect(GenerationCanvas.p1088.dimensions(aspect: wide) == (1920, 1088))
  }

  @Test("The short edge is what the name promises, in every aspect")
  func shortEdgeIsFixed() {
    for canvas in GenerationCanvas.allCases {
      for aspect in [16.0 / 9, 1.0, 9.0 / 16, 4.0 / 5, 3.0 / 2] {
        let size = canvas.dimensions(aspect: aspect)
        #expect(min(size.width, size.height) == canvas.shortEdge)
      }
    }
  }

  @Test("Every dimension the engine receives is a legal multiple of 32")
  func dimensionsAreLegal() {
    for canvas in GenerationCanvas.allCases {
      for aspect in [16.0 / 9, 1.0, 9.0 / 16, 4.0 / 5, 3.0 / 2, 2.39] {
        let size = canvas.dimensions(aspect: aspect)
        #expect(size.width % 32 == 0)
        #expect(size.height % 32 == 0)
        #expect(size.width >= 32 && size.height >= 32)
      }
    }
  }

  @Test("Labels name the short edge, not the square case")
  func labels() {
    #expect(GenerationCanvas.p352.label == "352p")
    #expect(GenerationCanvas.p1088.label == "1088p")
  }

  @Test("Settings saved under the old square names still load")
  func legacyDecoding() throws {
    let decoder = JSONDecoder()
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"square256\"".utf8)) == .p480)
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"native1344\"".utf8)) == .p1088)
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"p768\"".utf8)) == .p768)
  }

  @Test("A degenerate aspect falls back to square rather than crashing")
  func degenerateAspect() {
    #expect(GenerationCanvas.p480.dimensions(aspect: 0) == (480, 480))
    #expect(GenerationCanvas.p480.dimensions(aspect: .nan) == (480, 480))
  }

  /// Transcribed from the released pipeline. These are the numbers the model
  /// was trained on, and the app was outside both of them.
  @Test("H3's canvas matches the reference's adapt_canvas")
  func h3CanvasMatchesReference() {
    // 16:9 nominal is 1365x768, over the area cap, and scales to 1344x768.
    #expect(H3Canvas.dimensions(aspect: 16.0 / 9) == (1344, 768))
    #expect(H3Canvas.dimensions(aspect: 9.0 / 16) == (768, 1344))
    #expect(H3Canvas.dimensions(aspect: 1.0) == (768, 768))
    // The short edge never depends on what the caller asked for.
    for aspect in [16.0 / 9, 4.0 / 5, 1.0, 9.0 / 16, 3.0 / 2] {
      let size = H3Canvas.dimensions(aspect: aspect)
      #expect(min(size.width, size.height) == H3Canvas.shortEdge)
      #expect(size.width * size.height <= H3Canvas.maximumPixels)
      #expect(size.width % 32 == 0 && size.height % 32 == 0)
    }
  }

  @Test("Durations land on the 17k+5 grid inside the trained range")
  func h3DurationStaysInRange() {
    #expect(H3Duration.aligned(frames: 5) == 124)
    #expect(H3Duration.aligned(frames: 73) == 124, "73 was below the trained range")
    #expect(H3Duration.aligned(frames: 124) == 124)
    #expect(H3Duration.aligned(frames: 125) == 141)
    #expect(H3Duration.aligned(frames: 9999) == H3Duration.maximumFrames)
    for frames in stride(from: 124, through: 362, by: 17) {
      #expect(H3Duration.aligned(frames: frames) == frames)
      #expect((frames - 5) % 17 == 0)
    }
  }

  /// The snapshot writes its own coding keys, so a new knob has to be added
  /// in four places and reaches the wire only if every one of them agrees.
  /// A round trip is the cheapest way to catch the one that was missed.
  @Test("The image canvas survives a round trip and defaults when absent")
  func imageCanvasCoding() throws {
    let knobs = GenerationKnobSnapshot(
      canvas: .p768,
      imageCanvas: .s1280,
      denoisingSteps: 8,
      activeDiTLayers: 30,
      coreReuse: 1
    )
    let encoded = try JSONEncoder().encode(knobs)
    #expect(try JSONDecoder().decode(GenerationKnobSnapshot.self, from: encoded) == knobs)

    // Settings written before the knob existed carry no key at all.
    let legacy = Data(
      #"{"canvas":"p768","denoisingSteps":8,"activeDiTLayers":30,"coreReuse":1}"#.utf8
    )
    let restored = try JSONDecoder().decode(GenerationKnobSnapshot.self, from: legacy)
    #expect(restored.imageCanvas == .s1024)
  }

  /// Every tier the picker offers has to be one the renderer accepts: a
  /// multiple of 16 whose token count is a multiple of 32. 1440 looks like it
  /// belongs in this list and does not qualify, which is why it is asserted
  /// rather than eyeballed.
  @Test("Every offered image canvas is one the renderer can actually draw")
  func imageCanvasesAreRenderable() {
    for canvas in ImageCanvas.allCases {
      #expect(canvas.side % 16 == 0, "\(canvas.side) is not a multiple of 16")
      let tokensSide = canvas.side / 8 / 2
      #expect(
        (tokensSide * tokensSide) % 32 == 0,
        "\(canvas.side) gives \(tokensSide * tokensSide) tokens, not a multiple of 32"
      )
      #expect(canvas.label == "\(canvas.side) × \(canvas.side)")
    }
  }
}
