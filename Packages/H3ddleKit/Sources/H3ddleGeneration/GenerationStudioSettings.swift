import Foundation
import H3ddleEngineProtocol

/// How much work a generation spends: passes, blocks, denoiser reuse.
///
/// It is not a size, despite the case names. It used to be a ladder of short
/// edges claiming to reproduce a published resolution table, and the table
/// was wrong in both directions — its floor rendered below what H3 was
/// trained on, and its ceiling, 1920x1088, asked for nearly twice the pixels
/// the model accepts. H3 has one canvas; `H3Canvas` shapes it, and
/// `H3NativeCanvas` states it.
///
/// The cases keep their old names so settings saved under them still load.
public enum GenerationCanvas: String, CaseIterable, Codable, Sendable, Identifiable {
  case p352
  case p480
  case p576
  case p768
  case p1088

  public var id: String { rawValue }

  public var engineQuality: EngineGenerationQuality {
    switch self {
    case .p352, .p480: .preview
    case .p576: .standard
    case .p768, .p1088: .high
    }
  }

  public init(from decoder: Decoder) throws {
    let raw = try decoder.singleValueContainer().decode(String.self)
    // Settings saved before the ladder was named by short edge.
    switch raw {
    case "square256", "square512": self = .p480
    case "native768": self = .p768
    case "native1344": self = .p1088
    default:
      guard let value = GenerationCanvas(rawValue: raw) else {
        throw DecodingError.dataCorruptedError(
          in: try decoder.singleValueContainer(),
          debugDescription: "unknown canvas \(raw)"
        )
      }
      self = value
    }
  }
}

/// What a model built for stills renders, as against the video ladder above.
///
/// These are square, and that is this build's limit rather than the model's:
/// the Z-Image path carries one side through the transformer and the decoder,
/// so a canvas is a single number instead of a pair. The aspect-ratio control
/// therefore has nothing to act on and is hidden while such a model is
/// picked, rather than being offered and quietly ignored.
///
/// The ladder stops at 1536 because the decoder holds 256 channels at the
/// full picture — about 12 GB there — and starts at 512 because below it the
/// model stops following the prompt reliably.
public enum ImageCanvas: String, CaseIterable, Codable, Sendable, Identifiable {
  case p512
  case p768
  case p1024
  case p1280
  case p1536

  public var id: String { rawValue }

  /// The short edge this tier names. Z-Image's rule is that both sides divide
  /// by 16 and their token count — (width/16) x (height/16) — divides by 32;
  /// every one of these lands on that at every aspect the app offers, which a
  /// test checks rather than this comment asserting.
  public var shortEdge: Int {
    switch self {
    case .p512: 512
    case .p768: 768
    case .p1024: 1024
    case .p1280: 1280
    case .p1536: 1536
    }
  }

  public var label: String { "\(shortEdge)p" }

  /// The frame for a project's aspect ratio: the short edge is the tier's and
  /// the long one follows, nudged up until the token count is legal.
  ///
  /// The model itself has no aspect ratio it insists on — the released
  /// pipeline takes height and width separately and pads the token count. This
  /// engine refuses what would need padding rather than rendering it wrongly,
  /// so the nudge is what keeps every offered frame inside that.
  public func frame(aspect: Double) -> (width: Int, height: Int) {
    func snap(_ value: Double) -> Int { max(16, Int((value / 16).rounded()) * 16) }
    func legal(_ width: Int, _ height: Int) -> Bool {
      ((width / 16) * (height / 16)) % 32 == 0
    }
    guard aspect > 0 else { return (shortEdge, shortEdge) }
    var width = aspect >= 1 ? snap(Double(shortEdge) * aspect) : shortEdge
    var height = aspect >= 1 ? shortEdge : snap(Double(shortEdge) / aspect)
    var tries = 0
    while !legal(width, height), tries < 8 {
      if aspect >= 1 { width += 16 } else { height += 16 }
      tries += 1
    }
    return (width, height)
  }

  /// Image tokens at this size and shape, which is what the transformer's cost
  /// tracks.
  public func tokens(aspect: Double) -> Int {
    let frame = self.frame(aspect: aspect)
    return (frame.width / 16) * (frame.height / 16)
  }

  /// Roughly what a whole render costs on an M1 Pro at eight passes, so the
  /// price of a tier is visible before it is paid rather than discovered by
  /// waiting ten minutes.
  ///
  /// End to end, including loading and encoding — the number a person waits,
  /// not the sampler's share of it. 512 and 1536 are measured (78s and 495s);
  /// the rest are read off the line through them, which is fair because cost
  /// is close to linear in tokens over this range. The first render after
  /// launch runs longer than any of these while 14 GB of weights are read
  /// from disk for the first time; a measured 1024 came in at 322s cold
  /// against 234s warm.
  public func approximateMinutes(aspect: Double) -> Double {
    // 25.9s of fixed cost plus 50.9ms a token, from the two measured ends.
    (25.9 + 0.0509 * Double(tokens(aspect: aspect))) / 60
  }
}

/// H3's own canvas and duration rules, transcribed from the released
/// pipeline's `adapt_canvas` and its latent-length grid.
///
/// Both were being ignored, and the model does not fail loudly when they are:
/// asked for a canvas or a length it was never trained on it returns a
/// confident, well-made video of something else entirely. Measured here at 73
/// frames, roughly half of seeds came back with no relation to the prompt —
/// the same period-drama interior whatever the prompt said, at four passes and
/// at twelve, on the base checkpoint and the turbo one, and with the prompt
/// written to the vendor's own guide.
public enum H3Canvas {
  /// The short edge is *always* this. A caller's width and height choose the
  /// ratio and nothing else — which is what the reference does, and why every
  /// tier below 768 was drawing off-canvas.
  public static let shortEdge = H3NativeCanvas.shortEdge
  /// Area ceiling; the reference scales the whole canvas down to meet it
  /// rather than clamping one axis.
  public static let maximumPixels = H3NativeCanvas.maximumPixels

  public static func dimensions(aspect: Double) -> (width: Int, height: Int) {
    guard aspect.isFinite, aspect > 0 else { return (shortEdge, shortEdge) }
    var width = aspect >= 1 ? Double(shortEdge) * aspect : Double(shortEdge)
    var height = aspect >= 1 ? Double(shortEdge) : Double(shortEdge) / aspect
    let area = width * height
    if area > Double(maximumPixels) {
      let scale = (Double(maximumPixels) / area).squareRoot()
      width *= scale
      height *= scale
    }
    var snappedWidth = Self.snapped(width)
    var snappedHeight = Self.snapped(height)
    // Snapping rounds, and rounding up can put a scaled-down canvas back over
    // the cap: 2.39:1 lands on 1568x672, 21k pixels above it. Step the long
    // edge down until it fits. Ordinary ratios never enter this loop — 16:9
    // snaps to 1344x768, which is the cap exactly.
    while snappedWidth * snappedHeight > maximumPixels {
      if snappedWidth >= snappedHeight {
        snappedWidth -= 32
      } else {
        snappedHeight -= 32
      }
    }
    return (max(32, snappedWidth), max(32, snappedHeight))
  }

  private static func snapped(_ value: Double) -> Int {
    max(32, Int((value / 32).rounded()) * 32)
  }
}

/// The frame grid and the range the model was trained on.
public enum H3Duration {
  public static let fps = 24.0
  public static let chunk = 17
  /// 17k+5. 124 frames is the released pipeline's default and the bottom of
  /// the range it documents as trained; 362 is the top. We offered 22.
  public static let minimumFrames = 124
  public static let maximumFrames = 362

  public static func aligned(frames: Int) -> Int {
    var value = max(minimumFrames, frames)
    while (value - 5) % chunk != 0 { value += 1 }
    return min(value, maximumFrames)
  }

  public static var minimumSeconds: Double { Double(minimumFrames) / fps }
  public static var maximumSeconds: Double { Double(maximumFrames) / fps }
}

public enum GenerationPreset: String, CaseIterable, Codable, Sendable, Identifiable {
  case preview
  case standard
  case high
  case custom

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .preview: "Preview"
    case .standard: "Standard"
    case .high: "High"
    case .custom: "Custom"
    }
  }

  public static let named: [GenerationPreset] = [.preview, .standard, .high]
}

public struct GenerationKnobSnapshot: Hashable, Codable, Sendable {
  public var canvas: GenerationCanvas
  /// Kept beside the video canvas rather than replacing it: the two ladders
  /// are different shapes, and switching model should not forget which
  /// resolution the other one was set to.
  public var imageCanvas: ImageCanvas
  /// How large LTX renders. Separate from `imageCanvas` because a lane keeps
  /// its own choice: picking 1536 for a still should not quietly make every
  /// clip an hour long.
  public var ltxResolution: LTXResolution
  public var denoisingSteps: Int
  public var activeDiTLayers: Int
  public var coreReuse: Int
  /// Replay cached tail-block residuals on schedule-gated steps. Faster at
  /// standard step counts; the result is a different sample of the same
  /// quality, so it stays a deliberate choice.
  public var blockCache: Bool
  /// Render stills from a 5-frame clip rather than 22. Roughly 3x faster and
  /// visibly softer, so the detailed path stays the default.
  public var fastStill: Bool

  public init(
    canvas: GenerationCanvas,
    imageCanvas: ImageCanvas = .p1024,
    ltxResolution: LTXResolution = .p480,
    denoisingSteps: Int,
    activeDiTLayers: Int,
    coreReuse: Int,
    blockCache: Bool = false,
    fastStill: Bool = false
  ) {
    self.canvas = canvas
    self.imageCanvas = imageCanvas
    self.ltxResolution = ltxResolution
    self.denoisingSteps = denoisingSteps
    self.activeDiTLayers = activeDiTLayers
    self.coreReuse = coreReuse
    self.blockCache = blockCache
    self.fastStill = fastStill
  }

  enum CodingKeys: String, CodingKey {
    case canvas
    case imageCanvas
    case ltxResolution
    case denoisingSteps
    case activeDiTLayers
    case coreReuse
    case blockCache
    case fastStill
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    canvas = try container.decode(GenerationCanvas.self, forKey: .canvas)
    // Settings persisted before a dedicated image model existed decode to
    // the tier its own card quotes a time for.
    imageCanvas =
      try container.decodeIfPresent(ImageCanvas.self, forKey: .imageCanvas) ?? .p1024
    ltxResolution =
      try container.decodeIfPresent(LTXResolution.self, forKey: .ltxResolution)
      ?? .p480
    denoisingSteps = try container.decode(Int.self, forKey: .denoisingSteps)
    activeDiTLayers = try container.decode(Int.self, forKey: .activeDiTLayers)
    coreReuse = try container.decode(Int.self, forKey: .coreReuse)
    // Settings persisted before the knob existed decode to off.
    blockCache = try container.decodeIfPresent(Bool.self, forKey: .blockCache) ?? false
    fastStill = try container.decodeIfPresent(Bool.self, forKey: .fastStill) ?? false
  }

  public static func preset(_ preset: GenerationPreset) -> GenerationKnobSnapshot {
    switch preset {
    case .preview:
      GenerationKnobSnapshot(
        canvas: .p352, denoisingSteps: 4, activeDiTLayers: 50, coreReuse: 1
      )
    case .standard:
      GenerationKnobSnapshot(
        canvas: .p480, denoisingSteps: 20, activeDiTLayers: 45, coreReuse: 1
      )
    case .high:
      GenerationKnobSnapshot(
        canvas: .p768, denoisingSteps: 20, activeDiTLayers: 50, coreReuse: 1
      )
    case .custom:
      .preset(.preview)
    }
  }
}

/// LTX-2.5's frame sizes, named the way video is named: by the short edge.
///
/// The shape comes from the project's aspect ratio rather than from this, so
/// one choice reads the same in a landscape project and a portrait one. The
/// model has no aspect ratio it insists on — its only rule is that both sides
/// divide by 32, and its own released example renders 960x544 rather than a
/// square.
///
/// That rule is why two of these tiers do not render the number they name:
/// 720 and 1080 are not multiples of 32, so they render 704 and 1088. The
/// second is a happy landing — 1088 at 16:9 gives 1920x1088, which is exactly
/// what the released example reaches when its second stage doubles 960x544.
public enum LTXResolution: String, CaseIterable, Codable, Sendable, Identifiable {
    case p320
    case p480
    case p720
    case p1080

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .p320: "320p"
        case .p480: "480p"
        case .p720: "720p"
        case .p1080: "1080p"
        }
    }

    /// The short edge actually rendered, snapped to the multiple of 32 the
    /// video VAE insists on.
    public var shortEdge: Int {
        switch self {
        case .p320: 320
        case .p480: 480
        case .p720: 704
        case .p1080: 1088
        }
    }

    /// The frame for a project's aspect ratio: the short edge is the tier's,
    /// and the long one follows it, rounded to the nearest legal multiple.
    public func frame(aspect: Double) -> (width: Int, height: Int) {
        func snap(_ value: Double) -> Int {
            max(32, Int((value / 32).rounded()) * 32)
        }
        guard aspect > 0 else { return (shortEdge, shortEdge) }
        if aspect >= 1 {
            return (snap(Double(shortEdge) * aspect), shortEdge)
        }
        return (shortEdge, snap(Double(shortEdge) / aspect))
    }

    /// Latent cells in one frame at this aspect. The DiT's cost is linear in
    /// the token count and the token count is this times the latent frames —
    /// measured, not assumed: 29 s a step at 1872 tokens against 36 s at 2304.
    public func cellsPerFrame(aspect: Double) -> Int {
        let frame = self.frame(aspect: aspect)
        return (frame.width / 32) * (frame.height / 32)
    }
}

public struct GenerationStudioSettings: Hashable, Codable, Sendable {
  public var preset: GenerationPreset
  public var knobs: GenerationKnobSnapshot
  public var custom: GenerationKnobSnapshot
  public var seed: UInt64
  public var duration: Double
  public var alignedDurationStep: Double

  public static func makeDefault(
    seed: UInt64 = UInt64.random(in: 1..<100_000_000)
  ) -> GenerationStudioSettings {
    let preview = GenerationKnobSnapshot.preset(.preview)
    return GenerationStudioSettings(
      preset: .preview,
      knobs: preview,
      custom: preview,
      seed: seed,
      duration: 3,
      alignedDurationStep: 0
    )
  }

  public init(
    preset: GenerationPreset,
    knobs: GenerationKnobSnapshot,
    custom: GenerationKnobSnapshot,
    seed: UInt64,
    duration: Double,
    alignedDurationStep: Double
  ) {
    self.preset = preset
    self.knobs = knobs
    self.custom = custom
    self.seed = seed
    self.duration = duration
    self.alignedDurationStep = alignedDurationStep
  }

  public mutating func apply(preset newPreset: GenerationPreset) {
    if newPreset == .custom {
      knobs = custom
      preset = .custom
      return
    }
    preset = newPreset
    knobs = .preset(newPreset)
  }

  public mutating func updateKnobs(_ mutate: (inout GenerationKnobSnapshot) -> Void) {
    mutate(&knobs)
    if let match = GenerationPreset.named.first(where: { GenerationKnobSnapshot.preset($0) == knobs })
    {
      preset = match
    } else {
      custom = knobs
      preset = .custom
    }
  }
}
