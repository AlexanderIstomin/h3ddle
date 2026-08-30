import CoreGraphics
import Foundation
import H3ddleCore
import H3ddleGeneration
import H3ddleMedia

struct StudioImageAttachment: Identifiable, Hashable, Sendable {
  var id: UUID
  var url: URL

  init(id: UUID = UUID(), url: URL) {
    self.id = id
    self.url = url
  }
}

enum TimelinePresentationMode: String, CaseIterable, Identifiable {
  case expanded
  case collapsed

  var id: String { rawValue }
}

enum TimelineItemID: Hashable {
  case visual(UUID)
  case audio(UUID)
  case text(UUID)
}

enum EditorPanel: String, CaseIterable, Identifiable {
  case project
  case video
  case images
  case audio
  case effects
  case transitions
  case text
  case adjust
  case upscale
  case queue
  case models

  var id: String { rawValue }

  static let railTabs: [EditorPanel] = [
    .video, .images, .audio, .effects, .transitions, .text, .adjust, .queue, .models,
  ]

  var railLabel: String {
    switch self {
    case .project: "Project"
    case .video: "Video"
    case .images: "Images"
    case .audio: "Audio"
    case .effects: "Effects"
    case .transitions: "Transitions"
    case .text: "Text"
    case .adjust: "Adjust"
    case .upscale: "Upscale"
    case .queue: "Queue"
    case .models: "Models"
    }
  }

  var railSymbol: String {
    switch self {
    case .project: "slider.vertical.3"
    case .video: "film"
    case .images: "photo"
    case .audio: "waveform"
    case .effects: "sparkles"
    case .transitions: "rectangle.split.2x1"
    case .text: "textformat"
    case .adjust: "dial.low"
    case .upscale: "arrow.up.left.and.arrow.down.right"
    case .queue: "list.bullet.rectangle.portrait"
    case .models: "cpu"
    }
  }

  var panelTitle: String {
    switch self {
    case .project: "Project"
    case .video: "Video"
    case .images: "Images"
    case .audio: "Audio"
    case .effects: "Effects"
    case .transitions: "Transitions"
    case .text: "Text"
    case .adjust: "Adjust"
    case .upscale: "Upscale"
    case .queue: "Queue"
    case .models: "Models"
    }
  }

  var libraryKind: MediaKind? {
    switch self {
    case .video: .video
    case .images: .image
    case .audio: .audio
    default: nil
    }
  }

  var parentPanel: EditorPanel? {
    switch self {
    case .upscale: .adjust
    default: nil
    }
  }
}

struct CanvasGestureSession: Equatable {
  enum Kind: Equatable {
    case move
    case scale(CanvasCorner)
    case rotate
  }

  var target: TimelineItemID
  var kind: Kind
  var origin: CanvasObjectTransform
  var current: CanvasObjectTransform
  var startProgram: (x: Double, y: Double)
  var shiftDown: Bool
  var commandDown: Bool

  static func == (lhs: CanvasGestureSession, rhs: CanvasGestureSession) -> Bool {
    lhs.target == rhs.target
      && lhs.kind == rhs.kind
      && lhs.origin == rhs.origin
      && lhs.current == rhs.current
      && lhs.startProgram.x == rhs.startProgram.x
      && lhs.startProgram.y == rhs.startProgram.y
      && lhs.shiftDown == rhs.shiftDown
      && lhs.commandDown == rhs.commandDown
  }
}

enum ProgramAspectRatio: String, CaseIterable, Identifiable {
  case sixteenNine = "16:9"
  case nineSixteen = "9:16"
  case oneOne = "1:1"
  case fourFive = "4:5"
  case threeTwo = "3:2"

  var id: String { rawValue }

  var fraction: CGFloat {
    switch self {
    case .sixteenNine: 16 / 9
    case .nineSixteen: 9 / 16
    case .oneOne: 1
    case .fourFive: 4 / 5
    case .threeTwo: 3 / 2
    }
  }
}

struct GenerationResult: Identifiable, Hashable {
  var id: UUID
  var asset: AssetReference
  var kind: GenerationKind
  var prompt: String
  var createdAt: Date
}

struct TimelineMetrics: Equatable {
  var zoom: Double

  var pointsPerSecond: CGFloat {
    CGFloat(TimelineRuler.pointsPerSecond(zoom: zoom))
  }

  func x(for time: TimeInterval) -> CGFloat {
    CGFloat(time) * pointsPerSecond
  }

  func time(for x: CGFloat) -> TimeInterval {
    TimeInterval(x / max(pointsPerSecond, 1))
  }
}
