import Foundation
import H3ddleCore
import H3ddleUpscaling

/// App-owned state for an upscale that must outlive the inspector presenting it.
struct AssetUpscalingJob: Identifiable {
  enum State: Equatable {
    case running
    case completed
    case failed
    case cancelled
  }

  let id: UUID
  let sourceAssetID: AssetID
  let sourcePixelSize: UpscalingPixelSize
  let sourceDuration: TimeInterval
  let targetPixelSize: UpscalingPixelSize
  let mode: UpscalingMode
  let scaleFactor: Int
  var state: State
  var phase: String
  var progress: Double
  var completedAsset: AssetReference?
  var completedPixelSize: UpscalingPixelSize?
  var errorMessage: String?

  init(
    id: UUID = UUID(),
    sourceAssetID: AssetID,
    sourcePixelSize: UpscalingPixelSize,
    sourceDuration: TimeInterval,
    targetPixelSize: UpscalingPixelSize,
    mode: UpscalingMode,
    scaleFactor: Int
  ) {
    self.id = id
    self.sourceAssetID = sourceAssetID
    self.sourcePixelSize = sourcePixelSize
    self.sourceDuration = sourceDuration
    self.targetPixelSize = targetPixelSize
    self.mode = mode
    self.scaleFactor = scaleFactor
    self.state = .running
    self.phase = "Preparing"
    self.progress = 0
  }
}
