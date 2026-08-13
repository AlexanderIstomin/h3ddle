import CoreGraphics
import Foundation
import H3ddleCore
import H3ddleGeneration
import H3ddleMedia

enum TimelinePresentationMode: String, CaseIterable, Identifiable {
  case expanded
  case collapsed

  var id: String { rawValue }
}

enum TimelineItemID: Hashable {
  case visual(UUID)
  case audio(UUID)
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
