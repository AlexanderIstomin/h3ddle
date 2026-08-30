import H3ddleCore
import H3ddleDesignSystem
import H3ddleUpscaling
import SwiftUI

struct AssetUpscaleControls: View {
  @Bindable var model: AppModel
  var asset: AssetReference

  @State private var mode = UpscalingMode.best
  @State private var scaleFactor = 2
  @State private var sourcePixelSize: UpscalingPixelSize?
  @State private var sourceDuration: TimeInterval?
  @State private var dimensionProbeFailed = false
  @State private var dimensionErrorMessage: String?
  @State private var detailedModelStatus: UpscalingModelStatus?
  @State private var modelDownloadProgress = 0.0
  @State private var modelDownloadErrorMessage: String?
  @State private var modelDownloadTask: Task<Void, Never>?

  private var upscaleJob: AssetUpscalingJob? {
    model.assetUpscalingJob(for: asset.id)
  }

  private var isUpscaling: Bool {
    upscaleJob?.state == .running
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      resolutionComparison

      VStack(alignment: .leading, spacing: 16) {
        optionGroup("Scale") {
          Picker("Scale", selection: $scaleFactor) {
            Text("2×").tag(2)
            Text("4×").tag(4)
          }
          .labelsHidden()
          .pickerStyle(.segmented)
        }

        if asset.kind == .image || asset.kind == .video {
          optionGroup("Quality") {
            Picker("Quality", selection: $mode) {
              Text("Best").tag(UpscalingMode.best)
              Text("Detailed").tag(UpscalingMode.detailed)
              Text("Fast").tag(UpscalingMode.fast)
            }
            .labelsHidden()
            .pickerStyle(.segmented)
          }
        }
      }
      .disabled(isUpscaling || isModelDownloading)

      Divider().overlay(H3Color.line)

      VStack(alignment: .leading, spacing: 10) {
        if isUpscaling {
          ProgressView(value: upscaleJob?.progress ?? 0)
            .progressViewStyle(.linear)
          HStack {
            Text(upscaleJob?.phase ?? "Preparing")
              .font(.system(size: 10))
              .foregroundStyle(H3Color.textSecondary)
            Spacer()
            Button("Cancel") { model.cancelAssetUpscale(asset.id) }
              .buttonStyle(.plain)
              .font(.system(size: 10, weight: .semibold))
          }
        } else if isModelDownloading {
          ProgressView(value: modelDownloadProgress)
            .progressViewStyle(.linear)
          HStack {
            Text("Downloading Apple Super Resolution model")
              .font(.system(size: 10))
              .foregroundStyle(H3Color.textSecondary)
            Spacer()
            Text("\(Int((modelDownloadProgress * 100).rounded()))%")
              .font(.system(size: 9, weight: .medium, design: .monospaced))
              .foregroundStyle(H3Color.accent)
          }
          .accessibilityIdentifier("upscale-model-download-progress")
        } else if needsDetailedModelDownload {
          Button {
            startModelDownload()
          } label: {
            Label("Download Detailed Model", systemImage: "arrow.down.circle")
              .font(.system(size: 11, weight: .semibold))
              .frame(maxWidth: .infinity)
              .frame(height: 32)
          }
          .buttonStyle(.borderedProminent)
          .accessibilityIdentifier("download-upscale-model")
        } else {
          Button {
            startUpscale()
          } label: {
            Label("Create \(scaleFactor)× copy", systemImage: "arrow.up.left.and.arrow.down.right")
              .font(.system(size: 11, weight: .semibold))
              .frame(maxWidth: .infinity)
              .frame(height: 32)
          }
          .buttonStyle(.borderedProminent)
          .disabled(!canStartUpscale)
          .accessibilityIdentifier("asset-upscale")
        }

        if let errorMessage = upscaleJob?.errorMessage {
          Text(errorMessage)
            .font(.system(size: 10))
            .foregroundStyle(.red)
            .fixedSize(horizontal: false, vertical: true)
        } else if let dimensionErrorMessage {
          Text(dimensionErrorMessage)
            .font(.system(size: 10))
            .foregroundStyle(.red)
            .fixedSize(horizontal: false, vertical: true)
        } else if let modelDownloadErrorMessage {
          Text(modelDownloadErrorMessage)
            .font(.system(size: 10))
            .foregroundStyle(.red)
            .fixedSize(horizontal: false, vertical: true)
        } else if upscaleJob?.state == .completed,
          let completedAsset = upscaleJob?.completedAsset
        {
          completionStatus(asset: completedAsset, pixelSize: upscaleJob?.completedPixelSize)
        } else {
          Text(modeDescription)
            .font(.system(size: 9))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
    }
    .animation(.easeOut(duration: 0.16), value: upscaleJob?.completedAsset?.id)
    .task(id: asset.id) { await loadSourceMetadata() }
    .onChange(of: scaleFactor) { _, _ in
      modelDownloadErrorMessage = nil
      refreshDetailedModelState()
    }
    .onChange(of: mode) { _, _ in modelDownloadErrorMessage = nil }
    .onChange(of: asset.id) { _, _ in
      modelDownloadTask?.cancel()
    }
    .onDisappear {
      modelDownloadTask?.cancel()
    }
  }

  private var isModelDownloading: Bool {
    detailedModelStatus == .downloading || modelDownloadTask != nil
  }

  private var needsDetailedModelDownload: Bool {
    (asset.kind == .image || asset.kind == .video)
      && mode == .detailed
      && detailedModelStatus == .downloadRequired
  }

  private var canStartUpscale: Bool {
    guard sourcePixelSize != nil, sourceDuration != nil, !dimensionProbeFailed,
      !isModelDownloading
    else {
      return false
    }
    guard (asset.kind == .image || asset.kind == .video), mode == .detailed else {
      return true
    }
    return detailedModelStatus == .ready || detailedModelStatus == .notRequired
  }

  private func completionStatus(
    asset: AssetReference,
    pixelSize: UpscalingPixelSize?
  ) -> some View {
    HStack(spacing: 8) {
      Image(systemName: "checkmark.circle.fill")
        .font(.system(size: 12, weight: .semibold))
        .foregroundStyle(H3Color.accent)
      VStack(alignment: .leading, spacing: 2) {
        Text("Upscaled copy created")
          .font(.system(size: 10, weight: .semibold))
          .foregroundStyle(H3Color.textPrimary)
        Text(completionDetail(pixelSize))
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
      }
      Spacer()
      Button("View") {
        model.revealLibraryAsset(asset.id)
      }
      .buttonStyle(.plain)
      .font(.system(size: 10, weight: .semibold))
      .foregroundStyle(H3Color.accent)
      .accessibilityIdentifier("view-upscaled-asset")
    }
    .transition(.opacity.combined(with: .move(edge: .top)))
    .accessibilityIdentifier("upscale-completed")
  }

  private func completionDetail(_ pixelSize: UpscalingPixelSize?) -> String {
    guard let pixelSize else { return "Added to Library" }
    return "\(pixelSize.width) × \(pixelSize.height) · Added to Library"
  }

  private var resolutionComparison: some View {
    HStack(spacing: 10) {
      resolutionValue(
        label: "Original",
        value: resolutionText(sourcePixelSize),
        emphasized: false,
        accessibilityIdentifier: "upscale-original-resolution"
      )
      Image(systemName: "arrow.right")
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(H3Color.textSecondary)
      resolutionValue(
        label: "Upscaled",
        value: resolutionText(targetPixelSize),
        emphasized: true,
        accessibilityIdentifier: "upscale-target-resolution"
      )
    }
    .padding(.vertical, 2)
    .animation(.easeOut(duration: 0.16), value: scaleFactor)
  }

  private func resolutionValue(
    label: String,
    value: String,
    emphasized: Bool,
    accessibilityIdentifier: String
  ) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(label)
        .font(.system(size: 9, weight: .medium))
        .foregroundStyle(H3Color.textSecondary)
      Text(value)
        .font(.system(size: 11, weight: .semibold, design: .monospaced))
        .foregroundStyle(emphasized ? H3Color.accent : H3Color.textPrimary)
        .lineLimit(1)
        .minimumScaleFactor(0.75)
        .contentTransition(.numericText())
        .accessibilityIdentifier(accessibilityIdentifier)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private var targetPixelSize: UpscalingPixelSize? {
    guard let sourcePixelSize else { return nil }
    return try? scaled(sourcePixelSize, by: scaleFactor)
  }

  private func resolutionText(_ size: UpscalingPixelSize?) -> String {
    if let size {
      return "\(size.width) × \(size.height)"
    }
    return dimensionProbeFailed ? "Unavailable" : "Reading…"
  }

  private func loadSourceMetadata() async {
    if let upscaleJob {
      mode = upscaleJob.mode
      scaleFactor = upscaleJob.scaleFactor
      sourcePixelSize = upscaleJob.sourcePixelSize
      sourceDuration = upscaleJob.sourceDuration
    } else {
      sourcePixelSize = nil
      sourceDuration = nil
    }
    dimensionProbeFailed = false
    dimensionErrorMessage = nil
    detailedModelStatus = nil
    modelDownloadProgress = 0
    modelDownloadErrorMessage = nil
    do {
      switch asset.kind {
      case .image:
        sourcePixelSize = try UpscalingImageProbe.pixelSize(at: asset.url)
        sourceDuration = asset.duration
      case .video:
        let properties = try await UpscalingVideoProbe.inspect(asset.url)
        sourcePixelSize = properties.pixelSize
        sourceDuration = properties.duration
      case .audio:
        dimensionProbeFailed = true
      }
      refreshDetailedModelState()
    } catch is CancellationError {
      return
    } catch {
      dimensionProbeFailed = true
      dimensionErrorMessage = error.localizedDescription
    }
  }

  private func optionGroup<Content: View>(
    _ title: String,
    @ViewBuilder content: () -> Content
  ) -> some View {
    VStack(alignment: .leading, spacing: 7) {
      Text(title)
        .font(.system(size: 10, weight: .semibold))
        .foregroundStyle(H3Color.textSecondary)
      content()
    }
  }

  private var modeDescription: String {
    if asset.kind == .video {
      return switch mode {
      case .best:
        "Uses temporal Super Resolution when installed, with a local fallback."
      case .detailed:
        detailedModeDescription
      case .fast:
        "Uses AVFoundation high-quality scaling and preserves the source audio."
      }
    }
    return switch mode {
    case .best:
      "Uses Apple Super Resolution when installed, with a built-in local fallback."
    case .detailed:
      detailedModeDescription
    case .fast:
      "Uses the built-in Accelerate high-quality scaler without a model."
    }
  }

  private var detailedModeDescription: String {
    switch detailedModelStatus {
    case .ready, .notRequired:
      "Apple Super Resolution is ready for this \(asset.kind == .video ? "video" : "image") and scale."
    case .downloadRequired:
      "Downloads Apple's high-quality model once on this Mac."
    case .downloading:
      "Apple is downloading the high-quality model in the background."
    case nil:
      "Detailed mode is unavailable for this asset and scale on this Mac."
    }
  }

  private var modelDownloadRequest: UpscalingModelDownloadRequest? {
    guard (asset.kind == .image || asset.kind == .video), let sourcePixelSize else {
      return nil
    }
    return UpscalingModelDownloadRequest(
      sourceKind: asset.kind,
      sourcePixelSize: sourcePixelSize,
      minimumScaleFactor: scaleFactor
    )
  }

  private func refreshDetailedModelState() {
    guard let request = modelDownloadRequest,
      let snapshot = AppleUpscalingModelDownloader().snapshot(for: request)
    else {
      detailedModelStatus = nil
      modelDownloadProgress = 0
      return
    }
    detailedModelStatus = snapshot.status
    modelDownloadProgress = snapshot.fractionComplete
    if snapshot.status == .downloading, modelDownloadTask == nil {
      observeModelDownload(request)
    }
  }

  private func startModelDownload() {
    guard let request = modelDownloadRequest else {
      modelDownloadErrorMessage =
        "Apple Super Resolution is unavailable for this asset and scale."
      return
    }
    modelDownloadErrorMessage = nil
    observeModelDownload(request)
  }

  private func observeModelDownload(_ request: UpscalingModelDownloadRequest) {
    modelDownloadTask?.cancel()
    detailedModelStatus = .downloading
    modelDownloadTask = Task { @MainActor in
      defer { modelDownloadTask = nil }
      do {
        for try await event in AppleUpscalingModelDownloader().events(for: request) {
          try Task.checkCancellation()
          switch event {
          case .preparing:
            detailedModelStatus = .downloading
          case .progress(let fractionComplete):
            detailedModelStatus = .downloading
            modelDownloadProgress = fractionComplete
          case .completed:
            detailedModelStatus = .ready
            modelDownloadProgress = 1
          }
        }
      } catch is CancellationError {
        refreshDetailedModelState()
      } catch {
        modelDownloadErrorMessage = error.localizedDescription
        refreshDetailedModelState()
      }
    }
  }

  private func startUpscale() {
    guard canStartUpscale, let sourceSize = sourcePixelSize, let sourceDuration,
      let targetSize = try? scaled(sourceSize, by: scaleFactor)
    else { return }
    model.startAssetUpscale(
      asset: asset,
      sourcePixelSize: sourceSize,
      sourceDuration: sourceDuration,
      targetPixelSize: targetSize,
      mode: mode,
      scaleFactor: scaleFactor
    )
  }

  private func scaled(
    _ size: UpscalingPixelSize,
    by factor: Int
  ) throws -> UpscalingPixelSize {
    let (width, widthOverflow) = size.width.multipliedReportingOverflow(by: factor)
    let (height, heightOverflow) = size.height.multipliedReportingOverflow(by: factor)
    guard !widthOverflow, !heightOverflow else { throw UpscalingError.invalidDimensions }
    return UpscalingPixelSize(width: width, height: height)
  }

}
