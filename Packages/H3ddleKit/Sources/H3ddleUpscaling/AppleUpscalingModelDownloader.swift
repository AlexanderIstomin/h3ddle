import Foundation
import H3ddleCore
import VideoToolbox

public struct UpscalingModelDownloadRequest: Hashable, Sendable {
  public var sourceKind: MediaKind
  public var sourcePixelSize: UpscalingPixelSize
  public var minimumScaleFactor: Int

  public init(
    sourceKind: MediaKind,
    sourcePixelSize: UpscalingPixelSize,
    minimumScaleFactor: Int
  ) {
    self.sourceKind = sourceKind
    self.sourcePixelSize = sourcePixelSize
    self.minimumScaleFactor = minimumScaleFactor
  }
}

public struct UpscalingModelDownloadSnapshot: Equatable, Sendable {
  public var status: UpscalingModelStatus
  public var fractionComplete: Double

  public init(status: UpscalingModelStatus, fractionComplete: Double) {
    self.status = status
    self.fractionComplete = min(max(fractionComplete, 0), 1)
  }
}

public enum UpscalingModelDownloadEvent: Equatable, Sendable {
  case preparing
  case progress(Double)
  case completed
}

public protocol UpscalingModelDownloadProviding: Sendable {
  func snapshot(
    for request: UpscalingModelDownloadRequest
  ) -> UpscalingModelDownloadSnapshot?

  func events(
    for request: UpscalingModelDownloadRequest
  ) -> AsyncThrowingStream<UpscalingModelDownloadEvent, any Error>
}

/// Explicit, user-initiated access to Apple's optional Super Resolution model.
/// Capability inspection remains read-only; only `events(for:)` can begin a download.
public struct AppleUpscalingModelDownloader: UpscalingModelDownloadProviding {
  public init() {}

  public func snapshot(
    for request: UpscalingModelDownloadRequest
  ) -> UpscalingModelDownloadSnapshot? {
    guard #available(macOS 26.0, *),
      let configuration = AppleSuperResolutionConfigurationFactory.make(for: request)
    else {
      return nil
    }
    return UpscalingModelDownloadSnapshot(
      status: AppleUpscalingCapabilityProbe.modelStatus(configuration.configurationModelStatus),
      fractionComplete: Double(configuration.configurationModelPercentageAvailable)
    )
  }

  public func events(
    for request: UpscalingModelDownloadRequest
  ) -> AsyncThrowingStream<UpscalingModelDownloadEvent, any Error> {
    AsyncThrowingStream { continuation in
      let task = Task.detached(priority: .userInitiated) {
        do {
          guard #available(macOS 26.0, *),
            let configuration = AppleSuperResolutionConfigurationFactory.make(for: request)
          else {
            throw UpscalingError.unavailable(
              "Apple Super Resolution is unavailable for this asset and scale."
            )
          }
          try await observeOrDownload(configuration, continuation: continuation)
        } catch is CancellationError {
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in task.cancel() }
    }
  }

  @available(macOS 26.0, *)
  private func observeOrDownload(
    _ configuration: VTSuperResolutionScalerConfiguration,
    continuation: AsyncThrowingStream<UpscalingModelDownloadEvent, any Error>.Continuation
  ) async throws {
    if configuration.configurationModelStatus == .ready {
      continuation.yield(.progress(1))
      continuation.yield(.completed)
      continuation.finish()
      return
    }

    let initialStatus = configuration.configurationModelStatus
    let completion = AppleModelDownloadCompletion()
    continuation.yield(.preparing)
    if initialStatus == .downloadRequired {
      Task.detached(priority: .utility) {
        do {
          try await configuration.downloadConfigurationModel()
          await completion.finish(errorDescription: nil)
        } catch {
          await completion.finish(errorDescription: error.localizedDescription)
        }
      }
    }

    while true {
      try Task.checkCancellation()
      let status = configuration.configurationModelStatus
      let fraction = Double(configuration.configurationModelPercentageAvailable)
      continuation.yield(.progress(min(max(fraction, 0), 1)))

      if status == .ready {
        continuation.yield(.progress(1))
        continuation.yield(.completed)
        continuation.finish()
        return
      }
      switch await completion.outcome {
      case .failed(let description):
        throw UpscalingError.failed(description)
      case .succeeded where status == .downloadRequired:
        throw UpscalingError.failed(
          "Apple completed the model download, but the model is not ready."
        )
      case .pending, .succeeded:
        break
      }
      if initialStatus == .downloading, status == .downloadRequired {
        throw UpscalingError.failed("The Apple Super Resolution model download did not complete.")
      }
      try await Task.sleep(for: .milliseconds(150))
    }
  }
}

enum AppleSuperResolutionScalePlanner {
  static func scaleFactor(
    minimumScaleFactor: Int,
    supportedScaleFactors: [Int]
  ) -> Int? {
    supportedScaleFactors
      .filter { $0 >= max(1, minimumScaleFactor) }
      .min()
  }
}

@available(macOS 26.0, *)
private enum AppleSuperResolutionConfigurationFactory {
  static func make(
    for request: UpscalingModelDownloadRequest
  ) -> VTSuperResolutionScalerConfiguration? {
    guard request.sourcePixelSize.isValid,
      request.sourceKind == .image || request.sourceKind == .video,
      VTSuperResolutionScalerConfiguration.isSupported,
      let scaleFactor = AppleSuperResolutionScalePlanner.scaleFactor(
        minimumScaleFactor: request.minimumScaleFactor,
        supportedScaleFactors: VTSuperResolutionScalerConfiguration.supportedScaleFactors
      )
    else {
      return nil
    }
    let inputType: VTSuperResolutionScalerConfiguration.InputType =
      request.sourceKind == .video ? .video : .image
    let dimensionCandidates = request.sourceKind == .video
      && request.sourcePixelSize.width != request.sourcePixelSize.height
      ? [
        request.sourcePixelSize,
        UpscalingPixelSize(
          width: request.sourcePixelSize.height,
          height: request.sourcePixelSize.width
        ),
      ]
      : [request.sourcePixelSize]
    return dimensionCandidates.lazy.compactMap { size in
      VTSuperResolutionScalerConfiguration(
        frameWidth: size.width,
        frameHeight: size.height,
        scaleFactor: scaleFactor,
        inputType: inputType,
        usePrecomputedFlow: false,
        qualityPrioritization: .normal,
        revision: VTSuperResolutionScalerConfiguration.defaultRevision
      )
    }.first
  }
}

private actor AppleModelDownloadCompletion {
  enum Outcome: Sendable {
    case pending
    case succeeded
    case failed(String)
  }

  private(set) var outcome = Outcome.pending

  func finish(errorDescription: String?) {
    if let errorDescription {
      outcome = .failed(errorDescription)
    } else {
      outcome = .succeeded
    }
  }
}
