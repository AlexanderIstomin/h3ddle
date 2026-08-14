import Foundation

public enum ProjectAspect: String, CaseIterable, Codable, Sendable, Identifiable {
  case sixteenNine = "16:9"
  case nineSixteen = "9:16"
  case oneOne = "1:1"
  case fourFive = "4:5"

  public var id: String { rawValue }

  public var widthUnits: Int {
    switch self {
    case .sixteenNine: 16
    case .nineSixteen: 9
    case .oneOne: 1
    case .fourFive: 4
    }
  }

  public var heightUnits: Int {
    switch self {
    case .sixteenNine: 9
    case .nineSixteen: 16
    case .oneOne: 1
    case .fourFive: 5
    }
  }

  public var fraction: Double {
    Double(widthUnits) / Double(heightUnits)
  }
}

public enum ProjectResolution: String, CaseIterable, Sendable, Identifiable {
  case ultraHD
  case fullHD
  case hd
  case sd
  case low
  case veryLow
  case extreme

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .ultraHD: "4K (Ultra HD)"
    case .fullHD: "1080p (Full HD)"
    case .hd: "720p (HD)"
    case .sd: "480p (SD)"
    case .low: "320p (Low)"
    case .veryLow: "240p (Very Low)"
    case .extreme: "120p (Extremely Low)"
    }
  }

  public var shortLabel: String {
    switch self {
    case .ultraHD: "4K"
    case .fullHD: "1080p"
    case .hd: "720p"
    case .sd: "480p"
    case .low: "320p"
    case .veryLow: "240p"
    case .extreme: "120p"
    }
  }

  public var landscapeWidth: Int {
    switch self {
    case .ultraHD: 3840
    case .fullHD: 1920
    case .hd: 1280
    case .sd: 854
    case .low: 568
    case .veryLow: 426
    case .extreme: 214
    }
  }

  public var landscapeHeight: Int {
    switch self {
    case .ultraHD: 2160
    case .fullHD: 1080
    case .hd: 720
    case .sd: 480
    case .low: 320
    case .veryLow: 240
    case .extreme: 120
    }
  }

  /// The "p" scale: 1080p means the short edge is 1080, in any orientation.
  public var shortSide: Int { landscapeHeight }
}

public enum ProjectPlatform: String, CaseIterable, Codable, Sendable, Identifiable {
  case youtube
  case youtubeShorts
  case tiktok
  case instagramReel
  case instagramStory
  case instagramPost
  case instagramFeed
  case landscape
  case portrait
  case square
  case custom

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .youtube: "YouTube"
    case .youtubeShorts: "YouTube Shorts"
    case .tiktok: "TikTok"
    case .instagramReel: "Instagram Reel"
    case .instagramStory: "Instagram Story"
    case .instagramPost: "Instagram Post"
    case .instagramFeed: "Instagram Feed"
    case .landscape: "Landscape"
    case .portrait: "Portrait"
    case .square: "Square"
    case .custom: "Custom / None"
    }
  }

  public var detail: String {
    switch self {
    case .youtube: "Landscape video"
    case .landscape: "Standard video"
    case .youtubeShorts: "Vertical short"
    case .tiktok: "Vertical short"
    case .instagramReel: "Vertical reel"
    case .instagramStory: "Vertical story"
    case .instagramPost: "Square post"
    case .instagramFeed: "Portrait feed"
    case .portrait: "Vertical video"
    case .square: "Square canvas"
    case .custom: "Tune everything"
    }
  }

  public var width: Int {
    switch self {
    case .youtube, .landscape, .custom: 1920
    case .youtubeShorts, .tiktok, .instagramReel, .instagramStory, .portrait: 1080
    case .instagramPost, .square: 1080
    case .instagramFeed: 1080
    }
  }

  public var height: Int {
    switch self {
    case .youtube, .landscape, .custom: 1080
    case .youtubeShorts, .tiktok, .instagramReel, .instagramStory, .portrait: 1920
    case .instagramPost, .square: 1080
    case .instagramFeed: 1350
    }
  }

  public var framesPerSecond: Double { 30 }
}

public enum ProjectToneMapping: String, CaseIterable, Codable, Sendable, Identifiable {
  case none
  case agx
  case aces
  case neutral

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .none: "None"
    case .agx: "AgX"
    case .aces: "ACES"
    case .neutral: "Neutral"
    }
  }
}

public enum ProjectBackground: String, CaseIterable, Codable, Sendable, Identifiable {
  case black = "#000000"
  case white = "#FFFFFF"
  case navy = "#10202E"
  case forest = "#0E2A1E"
  case clear = "transparent"

  public var id: String { rawValue }

  public var isClear: Bool { self == .clear }
}

public struct ProjectSettings: Hashable, Codable, Sendable {
  public var width: Int
  public var height: Int
  public var framesPerSecond: Double
  public var background: ProjectBackground
  public var platform: ProjectPlatform
  public var masterGain: Double
  public var toneMapping: ProjectToneMapping
  public var exposure: Double

  public static let `default` = ProjectSettings()

  public static let frameRates: [Double] = [8, 12, 15, 16, 24, 30, 60]

  public init(
    width: Int = 1920,
    height: Int = 1080,
    framesPerSecond: Double = 24,
    background: ProjectBackground = .black,
    platform: ProjectPlatform = .custom,
    masterGain: Double = 1,
    toneMapping: ProjectToneMapping = .none,
    exposure: Double = 1
  ) {
    self.width = max(1, width)
    self.height = max(1, height)
    self.framesPerSecond = max(1, framesPerSecond)
    self.background = background
    self.platform = platform
    self.masterGain = min(max(masterGain, 0), 1)
    self.toneMapping = toneMapping
    self.exposure = max(exposure, 0)
  }

  public var aspect: ProjectAspect {
    ProjectSettings.aspect(width: width, height: height) ?? .sixteenNine
  }

  public var aspectFraction: Double {
    Double(width) / Double(height)
  }

  public var resolution: ProjectResolution? {
    ProjectResolution.allCases.first { $0.shortSide == min(width, height) }
  }

  public var resolutionLabel: String {
    resolution?.label ?? "\(width)×\(height)"
  }

  public var isCustomPlatform: Bool {
    platform == .custom
  }

  public var exposureStops: Double {
    guard exposure > 0 else { return -3 }
    return min(3, max(-3, log2(exposure)))
  }

  public mutating func apply(platform newPlatform: ProjectPlatform) {
    platform = newPlatform
    width = newPlatform.width
    height = newPlatform.height
    framesPerSecond = newPlatform.framesPerSecond
  }

  public mutating func apply(aspect: ProjectAspect) {
    platform = .custom
    let longEdge = max(width, height)
    let size = ProjectSettings.dimensions(aspect: aspect, longEdge: longEdge)
    width = size.width
    height = size.height
  }

  public mutating func apply(resolution: ProjectResolution) {
    platform = .custom
    let shortSide = resolution.shortSide
    let ratio = aspectFraction
    if ratio >= 1 {
      height = shortSide
      width = max(1, Int((Double(shortSide) * ratio).rounded()))
    } else {
      width = shortSide
      height = max(1, Int((Double(shortSide) / ratio).rounded()))
    }
  }

  public mutating func apply(frameRate: Double) {
    platform = .custom
    framesPerSecond = frameRate
  }

  public static func dimensions(aspect: ProjectAspect, longEdge: Int) -> (
    width: Int, height: Int
  ) {
    if aspect.widthUnits >= aspect.heightUnits {
      return (
        longEdge,
        max(1, Int((Double(longEdge * aspect.heightUnits) / Double(aspect.widthUnits)).rounded()))
      )
    }
    return (
      max(1, Int((Double(longEdge * aspect.widthUnits) / Double(aspect.heightUnits)).rounded())),
      longEdge
    )
  }

  public static func aspect(width: Int, height: Int) -> ProjectAspect? {
    ProjectAspect.allCases.first { width * $0.heightUnits == height * $0.widthUnits }
  }
}
