import CoreGraphics
import Foundation
import H3ddleCore
import H3ddleEngineClient
import H3ddleEngineProtocol
import H3ddleGeneration
import H3ddleModels
import ImageIO
import Observation
import os

enum ModelValidationState: Equatable {
  case notSelected
  case validating
  case ready
  case failed
}

enum ManagedModelState: Equatable {
  case checking
  case available
  case downloading
  case verifying
  case installing
  case installed
  case cancelled
  case failed
}

@MainActor
@Observable
final class AppModel {
  var project = H3ddleProject(name: "First H3ddle")
  var activeGenerationKind: GenerationKind?
  var isGenerating = false
  var generationPhase = ""
  var generationProgress = 0.0
  var generationElapsed: TimeInterval = 0
  var generationPreviewImage: CGImage?
  var generationPrompt = ""
  var errorMessage: String?
  var showsExportNotice = false
  var showsModelSettings = false
  var modelDirectory: URL?
  var modelValidationState: ModelValidationState = .notSelected
  var modelValidationMessage = "Choose a local MiniMax H3 model folder."
  var engineCapabilities: EngineCapabilities?
  var modelReport: EngineModelReport?
  let managedModel = ModelCatalog.minimaxH3Int8
  var managedModelState: ManagedModelState = .checking
  var managedModelProgress = 0.0
  var managedModelCompletedBytes: Int64 = 0
  var managedModelStatusMessage = "Checking managed model storage…"
  var managedModelInstalledURL: URL?
  var generationDurations: [AssetID: TimeInterval] = [:]
  var previewDenoise = false {
    didSet {
      userDefaults.set(previewDenoise, forKey: Self.previewDenoiseKey)
    }
  }

  private let generationProvider: any GenerationProvider
  private let engineSession: EngineSession
  private let engineInspector: any EngineInspecting
  private let modelDownloader: ModelPackageDownloader
  private let userDefaults: UserDefaults
  private var isAccessingModelDirectory = false
  private var generationTask: Task<Void, Never>?
  private var generationTimerTask: Task<Void, Never>?
  private var activeGenerationID: UUID?
  private var generationStartedAt: ContinuousClock.Instant?
  private var modelValidationTask: Task<Void, Never>?
  private var modelDownloadTask: Task<Void, Never>?
  private var phaseTimeline = GenerationPhaseTimeline()
  private var activeGenerationSettings = ""
  private static let generationLog = Logger(
    subsystem: "com.h3ddle.app",
    category: "generation"
  )

  private static let modelBookmarkKey = "H3ddle.modelDirectoryBookmark"
  private static let previewDenoiseKey = "H3ddle.previewDenoise"

  init(
    generationProvider: any GenerationProvider = FakeGenerationProvider(),
    engineExecutableURL: URL = EngineExecutableLocator.bundled(),
    engineSession: EngineSession? = nil,
    engineInspector: (any EngineInspecting)? = nil,
    modelDownloader: ModelPackageDownloader = ModelPackageDownloader(),
    userDefaults: UserDefaults = .standard
  ) {
    let session = engineSession ?? EngineSession(executableURL: engineExecutableURL)
    self.generationProvider = generationProvider
    self.engineSession = session
    self.engineInspector = engineInspector ?? session
    self.modelDownloader = modelDownloader
    self.userDefaults = userDefaults
    if userDefaults.object(forKey: Self.previewDenoiseKey) != nil {
      previewDenoise = userDefaults.bool(forKey: Self.previewDenoiseKey)
    }
    restoreModelDirectory()
    refreshManagedModelStatus()
  }

  var modelStatusTitle: String {
    switch modelValidationState {
    case .notSelected: "Model not set"
    case .validating: "Checking model"
    case .ready: "H3 ready"
    case .failed: "Model needs attention"
    }
  }

  var managedModelStatusTitle: String {
    switch managedModelState {
    case .checking: "Checking download"
    case .available: "Ready to download"
    case .downloading: "Downloading model"
    case .verifying: "Verifying model"
    case .installing: "Installing model"
    case .installed: "Model downloaded"
    case .cancelled: "Download paused"
    case .failed: "Download needs attention"
    }
  }

  var managedModelDownloadIsActive: Bool {
    switch managedModelState {
    case .downloading, .verifying, .installing:
      true
    default:
      false
    }
  }

  func generationBackendDescription(
    for kind: GenerationKind,
    quality: EngineGenerationQuality = .preview,
    denoisingSteps: Int? = nil,
    activeDiTLayers: Int? = nil,
    coreReuse: Int? = nil,
    previewDenoise: Bool = false
  ) -> String {
    if usesNativeEngine(for: kind) {
      let canvas = kind == .audio ? "32×32" : "\(quality.canvasSize)×\(quality.canvasSize)"
      let steps = denoisingSteps ?? quality.denoisingSteps
      var parts = ["Local h3.c"]
      if kind == .image {
        parts.append("22-frame still")
      }
      if kind == .audio {
        parts.append("soundtrack only")
      }
      parts.append(contentsOf: [canvas, "\(steps) denoising passes"])
      let layers = activeDiTLayers ?? quality.activeDiTLayers
      if layers < 50 {
        parts.append("\(layers) blocks")
      }
      if let coreReuse, coreReuse > 1 {
        parts.append("core reuse \(coreReuse)")
      }
      if previewDenoise {
        parts.append("preview on")
      }
      return parts.joined(separator: " · ")
    }
    switch kind {
    case .video, .image:
      return "Prototype provider · appends to the visual program"
    case .audio:
      return "Prototype audio provider · appends at the current audio end"
    }
  }

  func presentGeneration(_ kind: GenerationKind) {
    errorMessage = nil
    generationPhase = ""
    generationProgress = 0
    generationElapsed = 0
    generationPreviewImage = nil
    activeGenerationKind = kind
  }

  func generate(
    prompt: String,
    duration: TimeInterval,
    quality: EngineGenerationQuality = .preview,
    denoisingSteps: Int? = nil,
    activeDiTLayers: Int? = nil,
    coreReuse: Int? = nil,
    previewDenoise: Bool = false
  ) {
    guard let kind = activeGenerationKind else { return }
    generationTask?.cancel()
    isGenerating = true
    errorMessage = nil
    generationElapsed = 0
    generationPreviewImage = nil
    phaseTimeline = GenerationPhaseTimeline()
    activeGenerationSettings = Self.settingsDescription(
      kind: kind,
      duration: duration,
      quality: quality,
      denoisingSteps: denoisingSteps,
      activeDiTLayers: activeDiTLayers,
      coreReuse: coreReuse,
      previewDenoise: previewDenoise
    )

    let generationID = UUID()
    let clock = ContinuousClock()
    let startedAt = clock.now
    activeGenerationID = generationID
    generationStartedAt = startedAt
    generationTimerTask?.cancel()
    generationTimerTask = Task { [weak self] in
      while !Task.isCancelled {
        guard let self, activeGenerationID == generationID else { return }
        generationElapsed = Self.seconds(in: startedAt.duration(to: clock.now))
        do {
          try await Task.sleep(for: .seconds(1))
        } catch {
          return
        }
      }
    }

    let request = GenerationRequest(
      kind: kind,
      prompt: prompt,
      duration: duration,
      quality: quality,
      denoisingSteps: denoisingSteps,
      activeDiTLayers: activeDiTLayers,
      coreReuse: coreReuse,
      previewDenoise: previewDenoise
    )
    let nativeModelDirectory = usesNativeEngine(for: kind) ? modelDirectory : nil
    let provider: any GenerationProvider =
      if let nativeModelDirectory {
        EngineGenerationProvider(
          session: engineSession,
          modelDirectory: nativeModelDirectory
        )
      } else {
        generationProvider
      }
    generationTask = Task { [weak self] in
      guard let self else { return }
      var completedAssetID: AssetID?
      defer {
        finishGenerationTiming(
          generationID: generationID,
          startedAt: startedAt,
          completedAssetID: completedAssetID
        )
      }
      do {
        for try await event in provider.events(for: request) {
          switch event {
          case .progress(let phase, let fractionComplete):
            generationPhase = phase
            generationProgress = fractionComplete
            phaseTimeline.record(
              phase: phase,
              elapsed: Self.seconds(in: startedAt.duration(to: clock.now))
            )
          case .preview(let url):
            if let image = Self.loadPreviewImage(from: url) {
              generationPreviewImage = image
            }
          case .completed(let asset):
            try append(asset)
            completedAssetID = asset.id
            activeGenerationKind = nil
          }
        }
      } catch {
        errorMessage = error.localizedDescription
      }
    }
  }

  var generationElapsedDescription: String {
    Self.formatElapsed(generationElapsed)
  }

  var previewAsset: AssetReference? {
    for item in project.timeline.visualItems.reversed() where item.isEnabled {
      if let asset = project.asset(id: item.assetID) {
        return asset
      }
    }
    return nil
  }

  func generationDurationDescription(for asset: AssetReference) -> String? {
    generationDurations[asset.id].map(Self.formatElapsed)
  }

  var nativeVideoGenerationIsReady: Bool {
    modelValidationState == .ready
      && engineCapabilities?.supports(.videoGeneration) == true
      && modelReport?.supportsGeneration == true
  }

  var nativeImageGenerationIsReady: Bool {
    modelValidationState == .ready
      && engineCapabilities?.supports(.imageGeneration) == true
      && modelReport?.supportsGeneration == true
  }

  var nativeAudioGenerationIsReady: Bool {
    modelValidationState == .ready
      && engineCapabilities?.supports(.standaloneAudioGeneration) == true
      && modelReport?.supportsGeneration == true
  }

  func usesNativeEngine(for kind: GenerationKind) -> Bool {
    switch kind {
    case .video: nativeVideoGenerationIsReady
    case .image: nativeImageGenerationIsReady
    case .audio: nativeAudioGenerationIsReady
    }
  }

  func cancelGeneration() {
    let generationID = activeGenerationID
    let startedAt = generationStartedAt
    generationTask?.cancel()
    generationTask = nil
    if let generationID, let startedAt {
      finishGenerationTiming(
        generationID: generationID,
        startedAt: startedAt,
        completedAssetID: nil
      )
    } else {
      isGenerating = false
    }
  }

  func selectModelDirectory(_ url: URL) {
    if modelDirectory != url {
      engineSession.shutdown()
      endModelAccess()
    }
    modelDirectory = url
    persistModelDirectory(url)
    validateSelectedModel()
  }

  func validateSelectedModel() {
    guard let modelDirectory else {
      modelValidationState = .notSelected
      modelValidationMessage = "Choose a local MiniMax H3 model folder."
      modelReport = nil
      return
    }

    modelValidationTask?.cancel()
    modelValidationState = .validating
    modelValidationMessage = "Loading the H3 model and compiling Metal…"
    modelReport = nil
    beginModelAccess()

    modelValidationTask = Task { [weak self] in
      guard let self else { return }

      do {
        let capabilities = try await engineInspector.capabilities()
        try Task.checkCancellation()
        guard capabilities.supports(.modelInspection) else {
          throw EngineClientError.rejected("This engine cannot inspect local models.")
        }
        let report = try await engineInspector.inspectModel(at: modelDirectory)
        try Task.checkCancellation()
        engineCapabilities = capabilities
        modelReport = report
        if report.supportsGeneration {
          modelValidationState = .ready
          modelValidationMessage = "H3 model is loaded and Metal is ready."
        } else {
          modelValidationState = .failed
          modelValidationMessage =
            "This model is valid, but this h3.c build does not support its generation layout."
        }
      } catch is CancellationError {
        return
      } catch {
        engineCapabilities = nil
        modelReport = nil
        modelValidationState = .failed
        modelValidationMessage = error.localizedDescription
      }
    }
  }

  func clearModelDirectory() {
    modelValidationTask?.cancel()
    engineSession.shutdown()
    endModelAccess()
    userDefaults.removeObject(forKey: Self.modelBookmarkKey)
    modelDirectory = nil
    modelReport = nil
    engineCapabilities = nil
    modelValidationState = .notSelected
    modelValidationMessage = "Choose a local MiniMax H3 model folder."
  }

  func shutdownEngine() {
    generationTask?.cancel()
    modelValidationTask?.cancel()
    engineSession.shutdown()
    endModelAccess()
  }

  func downloadManagedModel() {
    guard !managedModelDownloadIsActive else { return }
    modelDownloadTask?.cancel()
    managedModelState = .downloading
    managedModelStatusMessage = "Preparing the pinned Hugging Face package…"

    let manifest = managedModel
    let downloader = modelDownloader
    modelDownloadTask = Task { [weak self] in
      guard let self else { return }
      do {
        let installedURL = try await downloader.download(manifest) { [weak self] progress in
          Task { @MainActor [weak self] in
            self?.applyManagedModelProgress(progress)
          }
        }
        try Task.checkCancellation()
        managedModelInstalledURL = installedURL
        managedModelCompletedBytes = manifest.totalByteCount
        managedModelProgress = 1
        managedModelState = .installed
        managedModelStatusMessage =
          manifest.compatibility == .ready
          ? "The package is installed and ready for prompt-only local video generation."
          : "The package is installed, but this engine build still requires a checkpoint adapter."
      } catch is CancellationError {
        managedModelState = .cancelled
        managedModelStatusMessage = "Paused. Downloaded data is kept for a later resume."
      } catch {
        managedModelState = .failed
        managedModelStatusMessage = error.localizedDescription
      }
    }
  }

  func cancelManagedModelDownload() {
    modelDownloadTask?.cancel()
    modelDownloadTask = nil
    managedModelState = .cancelled
    managedModelStatusMessage = "Pausing… downloaded data will be kept."
  }

  func refreshManagedModelStatus() {
    guard !managedModelDownloadIsActive else { return }
    modelDownloadTask?.cancel()
    managedModelState = .checking
    let manifest = managedModel
    let downloader = modelDownloader
    modelDownloadTask = Task { [weak self] in
      guard let self else { return }
      if let installedURL = await downloader.installedPackageURL(for: manifest) {
        managedModelInstalledURL = installedURL
        managedModelCompletedBytes = manifest.totalByteCount
        managedModelProgress = 1
        managedModelState = .installed
        managedModelStatusMessage =
          manifest.compatibility == .ready
          ? "The package is installed and ready for prompt-only local video generation."
          : "The package is installed, but this engine build still requires a checkpoint adapter."
        return
      }

      do {
        let stagedBytes = try await downloader.stagedByteCount(for: manifest)
        try Task.checkCancellation()
        managedModelCompletedBytes = stagedBytes
        managedModelProgress =
          manifest.totalByteCount > 0
          ? Double(stagedBytes) / Double(manifest.totalByteCount)
          : 0
        managedModelState = stagedBytes > 0 ? .cancelled : .available
        managedModelStatusMessage =
          stagedBytes > 0
          ? "A partial download is available and can be resumed."
          : "Pinned INT8 package recommended for Macs with 32 GB or more."
      } catch {
        managedModelState = .failed
        managedModelStatusMessage = error.localizedDescription
      }
    }
  }

  private func applyManagedModelProgress(_ progress: ModelDownloadProgress) {
    managedModelCompletedBytes = progress.completedBytes
    managedModelProgress = progress.fractionCompleted
    switch progress.phase {
    case .preparing:
      managedModelState = .downloading
      managedModelStatusMessage = "Checking free space and existing files…"
    case .downloading:
      managedModelState = .downloading
      managedModelStatusMessage =
        progress.currentFileName.map { "Downloading \($0)…" } ?? "Downloading model…"
    case .verifying:
      managedModelState = .verifying
      managedModelStatusMessage =
        progress.currentFileName.map { "Verifying \($0)…" } ?? "Verifying model…"
    case .installing:
      managedModelState = .installing
      managedModelStatusMessage = "Installing the verified package…"
    case .completed:
      managedModelState = .installed
      managedModelStatusMessage = "The optimized model package is installed."
    case .cancelled:
      managedModelState = .cancelled
      managedModelStatusMessage = "Paused. Downloaded data is kept for a later resume."
    }
  }

  func toggleVisual(_ id: UUID) {
    guard let item = project.timeline.visualItems.first(where: { $0.id == id }) else {
      return
    }
    project.timeline.setVisualEnabled(id, isEnabled: !item.isEnabled)
  }

  func toggleVisualNativeAudio(_ id: UUID) {
    guard let index = project.timeline.visualItems.firstIndex(where: { $0.id == id }) else {
      return
    }
    var items = project.timeline.visualItems
    items[index].includesNativeAudio.toggle()
    project.timeline = ProjectTimeline(
      visualItems: items,
      audioItems: project.timeline.audioItems
    )
  }

  func toggleAudio(_ id: UUID) {
    guard let item = project.timeline.audioItems.first(where: { $0.id == id }) else {
      return
    }
    project.timeline.setAudioEnabled(id, isEnabled: !item.isEnabled)
  }

  func removeVisual(_ id: UUID) {
    project.timeline.removeVisual(id)
  }

  func removeAudio(_ id: UUID) {
    project.timeline.removeAudio(id)
  }

  private func append(_ asset: AssetReference) throws {
    project.addAsset(asset)
    if asset.kind == .audio {
      try project.timeline.appendAudio(asset)
    } else {
      try project.timeline.appendVisual(asset)
    }
  }

  private func finishGenerationTiming(
    generationID: UUID,
    startedAt: ContinuousClock.Instant,
    completedAssetID: AssetID?
  ) {
    guard activeGenerationID == generationID else { return }
    let elapsed = Self.seconds(in: startedAt.duration(to: ContinuousClock().now))
    generationTimerTask?.cancel()
    generationTimerTask = nil
    generationElapsed = elapsed
    phaseTimeline.finish(elapsed: elapsed)
    if let summary = phaseTimeline.summary {
      Self.generationLog.info(
        "Generation [\(self.activeGenerationSettings, privacy: .public)] phases: \(summary, privacy: .public) · total \(elapsed, format: .fixed(precision: 1))s"
      )
    }
    if let completedAssetID {
      generationDurations[completedAssetID] = elapsed
    }
    activeGenerationID = nil
    generationStartedAt = nil
    generationTask = nil
    isGenerating = false
  }

  /// Every knob is stated explicitly, defaults included, so pasted logs from
  /// different runs are directly comparable.
  private static func settingsDescription(
    kind: GenerationKind,
    duration: TimeInterval,
    quality: EngineGenerationQuality,
    denoisingSteps: Int?,
    activeDiTLayers: Int?,
    coreReuse: Int?,
    previewDenoise: Bool
  ) -> String {
    guard kind == .video || kind == .image || kind == .audio else {
      return String(format: "%@ · %.0fs", kind.rawValue, duration)
    }
    let canvas =
      kind == .audio ? "32×32" : "\(quality.canvasSize)×\(quality.canvasSize)"
    let steps = denoisingSteps ?? quality.denoisingSteps
    let layers = activeDiTLayers ?? quality.activeDiTLayers
    let core = coreReuse ?? 1
    let reuse = core > 1 || steps < 10 ? 1 : quality.denoiseReuse
    return String(
      format:
        "%@ · %@ · %@ · passes %d · blocks %d · core-reuse %d · reuse %d · preview %@ · %.2fs",
      kind.rawValue, quality.rawValue, canvas, steps, layers, core, reuse,
      previewDenoise ? "on" : "off", duration
    )
  }

  private func beginModelAccess() {
    guard let modelDirectory, !isAccessingModelDirectory else { return }
    isAccessingModelDirectory = modelDirectory.startAccessingSecurityScopedResource()
  }

  private func endModelAccess() {
    guard isAccessingModelDirectory else { return }
    modelDirectory?.stopAccessingSecurityScopedResource()
    isAccessingModelDirectory = false
  }

  private static func loadPreviewImage(from url: URL) -> CGImage? {
    // The engine overwrites one preview file per job; caching would pin the
    // first frame forever.
    let options = [kCGImageSourceShouldCache: false] as CFDictionary
    guard let source = CGImageSourceCreateWithURL(url as CFURL, options) else {
      return nil
    }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
  }

  private static func seconds(in duration: Duration) -> TimeInterval {
    let components = duration.components
    return Double(components.seconds) + Double(components.attoseconds) / 1e18
  }

  private static func formatElapsed(_ elapsed: TimeInterval) -> String {
    let elapsed = max(0, elapsed)
    if elapsed < 60 {
      return String(format: "%.1f s", elapsed)
    }
    let minutes = Int(elapsed) / 60
    let seconds = elapsed - Double(minutes * 60)
    return String(format: "%d:%04.1f", minutes, seconds)
  }

  private func persistModelDirectory(_ url: URL) {
    guard
      let bookmark = try? url.bookmarkData(
        options: .withSecurityScope,
        includingResourceValuesForKeys: nil,
        relativeTo: nil
      )
    else {
      return
    }
    userDefaults.set(bookmark, forKey: Self.modelBookmarkKey)
  }

  private func restoreModelDirectory() {
    guard let bookmark = userDefaults.data(forKey: Self.modelBookmarkKey) else { return }
    var isStale = false
    guard
      let url = try? URL(
        resolvingBookmarkData: bookmark,
        options: .withSecurityScope,
        relativeTo: nil,
        bookmarkDataIsStale: &isStale
      )
    else {
      userDefaults.removeObject(forKey: Self.modelBookmarkKey)
      return
    }
    modelDirectory = url
    if isStale {
      persistModelDirectory(url)
    }
    validateSelectedModel()
  }
}
