import CoreGraphics
import Foundation
import H3ddleCore
import H3ddleEngineClient
import H3ddleEngineProtocol
import H3ddleGeneration
import H3ddleMedia
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

struct ManagedPackageStatus: Equatable {
  var state: ManagedModelState = .checking
  var progress = 0.0
  var completedBytes: Int64 = 0
  var message = "Checking managed model storage…"
  var installedURL: URL?

  var downloadIsActive: Bool {
    switch state {
    case .downloading, .verifying, .installing: true
    default: false
    }
  }
}

/// One selectable model: a managed catalog package or a user-added folder.
struct ModelChoice: Identifiable, Equatable {
  enum Source: Equatable {
    case managed(ModelPackageManifest)
    case localFolder(bookmark: Data)
  }

  let id: String
  var displayName: String
  var subtitle: String
  var source: Source
  var directory: URL?
  var generationProfile: ModelGenerationProfile

  var isInstalled: Bool { directory != nil }

  var isLocalFolder: Bool {
    if case .localFolder = source { return true }
    return false
  }
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
  var importErrorMessage: String?
  var showsExport = false
  var showsModelSettings = false
  var showsProjectSettings = false
  var modelDirectory: URL?
  var modelValidationState: ModelValidationState = .notSelected
  var modelValidationMessage = "Choose a local MiniMax H3 model folder."
  var engineCapabilities: EngineCapabilities?
  var modelReport: EngineModelReport?
  let managedModel = ModelCatalog.minimaxH3Int8
  let managedManifests = [
    ModelCatalog.minimaxH3Int8,
    ModelCatalog.minimaxH3TurboInt8,
    ModelCatalog.minimaxH3Ref2VAInt8,
  ]
  var managedStatuses: [String: ManagedPackageStatus] = [:]
  var modelChoices: [ModelChoice] = []
  var selectedModelID: String? {
    didSet {
      userDefaults.set(selectedModelID, forKey: Self.selectedModelKey)
    }
  }

  var managedModelState: ManagedModelState { status(for: managedModel).state }
  var managedModelProgress: Double { status(for: managedModel).progress }
  var managedModelCompletedBytes: Int64 { status(for: managedModel).completedBytes }
  var managedModelStatusMessage: String { status(for: managedModel).message }
  var managedModelInstalledURL: URL? { status(for: managedModel).installedURL }
  var generationDurations: [AssetID: TimeInterval] = [:]
  var previewDenoise = false {
    didSet {
      userDefaults.set(previewDenoise, forKey: Self.previewDenoiseKey)
    }
  }
  var playback = ProgramPlaybackController()
  var timelineZoom = 1.0
  var timelineMode = TimelinePresentationMode.expanded
  var showsEffectLanes = false
  var selectedTimelineItem: TimelineItemID?
  var visualTrackMuted = false
  var audioTrackMuted = false
  var studioResults: [GenerationResult] = []
  var studioAspect = ProgramAspectRatio.sixteenNine
  var studioSettings = GenerationStudioSettings.makeDefault() {
    didSet {
      persistStudioSettings()
    }
  }
  var studioStartFrame: StudioImageAttachment?
  var studioEndFrame: StudioImageAttachment?
  var studioReferenceImages: [StudioImageAttachment] = []

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

  private var localModelBookmarks: [Data] = [] {
    didSet {
      userDefaults.set(localModelBookmarks, forKey: Self.localModelsKey)
    }
  }

  private static let modelBookmarkKey = "H3ddle.modelDirectoryBookmark"
  private static let previewDenoiseKey = "H3ddle.previewDenoise"
  private static let studioSettingsKey = "H3ddle.studioGenerationSettings"
  private static let selectedModelKey = "H3ddle.selectedModelID"
  private static let localModelsKey = "H3ddle.localModelBookmarks"

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
    if let data = userDefaults.data(forKey: Self.studioSettingsKey),
      let decoded = try? JSONDecoder().decode(GenerationStudioSettings.self, from: data)
    {
      studioSettings = decoded
    } else {
      persistStudioSettings()
    }
    restoreModelLibrary()
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

  /// Audio renders the joint model at H3's native canvas; a smaller one loses
  /// the prompt and returns speech. Shown wherever settings are reported.
  static var audioCanvasLabel: String {
    let size = EngineGenerationRequest.audioCanvasSize
    return "\(size)×\(size)"
  }

  var managedModelDownloadIsActive: Bool {
    switch managedModelState {
    case .downloading, .verifying, .installing:
      true
    default:
      false
    }
  }

  /// Bytes an install of this package still has to fetch. Packages share most
  /// of their weights, so a manifest's total overstates the cost whenever
  /// another package already supplies those files.
  func pendingDownloadBytes(for manifest: ModelPackageManifest) -> Int64 {
    manifest.totalByteCount
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
      let canvas = kind == .audio ? Self.audioCanvasLabel : "\(quality.canvasSize)×\(quality.canvasSize)"
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
    if let ratio = ProgramAspectRatio(rawValue: project.settings.aspect.rawValue) {
      studioAspect = ratio
    }
  }

  func updateProjectSettings(_ mutate: (inout ProjectSettings) -> Void) {
    mutate(&project.settings)
    playback.clock.framesPerSecond = project.settings.framesPerSecond
    syncPlayback()
  }

  func applyStudioPreset(_ preset: GenerationPreset) {
    studioSettings.apply(preset: preset)
  }

  func updateStudioKnobs(_ mutate: (inout GenerationKnobSnapshot) -> Void) {
    studioSettings.updateKnobs(mutate)
  }

  var studioHasFrameAnchors: Bool {
    studioStartFrame != nil || studioEndFrame != nil
  }

  var studioHasReferences: Bool {
    !studioReferenceImages.isEmpty
  }

  func setStudioStartFrame(_ url: URL) {
    guard !studioHasReferences else { return }
    studioStartFrame = StudioImageAttachment(url: url)
  }

  func setStudioEndFrame(_ url: URL) {
    guard !studioHasReferences else { return }
    studioEndFrame = StudioImageAttachment(url: url)
  }

  func clearStudioStartFrame() {
    studioStartFrame = nil
  }

  func clearStudioEndFrame() {
    studioEndFrame = nil
  }

  /// Ref2VA accepts 12 references overall but only 9 of them images, and every
  /// reference the studio attaches is an image, so 9 is the limit that applies.
  static let studioReferenceLimit = 9

  func addStudioReference(_ url: URL) {
    guard !studioHasFrameAnchors, studioReferenceImages.count < Self.studioReferenceLimit
    else { return }
    studioReferenceImages.append(StudioImageAttachment(url: url))
  }

  func removeStudioReference(_ id: UUID) {
    studioReferenceImages.removeAll { $0.id == id }
  }

  func resolveStudioPromptMentions(_ prompt: String) -> String {
    var resolved = prompt
    for (index, _) in studioReferenceImages.enumerated() {
      let number = index + 1
      for token in ["@Picture \(number)", "@Picture\(number)", "@\(number)"] {
        resolved = resolved.replacingOccurrences(of: token, with: "<Picture \(number)>")
      }
    }
    return resolved
  }

  func persistStudioSettings() {
    if let data = try? JSONEncoder().encode(studioSettings) {
      userDefaults.set(data, forKey: Self.studioSettingsKey)
    }
  }

  func generate(
    prompt: String,
    duration: TimeInterval,
    quality: EngineGenerationQuality = .preview,
    denoisingSteps: Int? = nil,
    activeDiTLayers: Int? = nil,
    coreReuse: Int? = nil,
    previewDenoise: Bool = false,
    seed: UInt64? = nil,
    canvasWidth: Int? = nil,
    canvasHeight: Int? = nil
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
    ) + " · model \(selectedModelChoice?.displayName ?? "folder")"
      + (selectedGenerationProfile.usesBetaSchedule ? " · beta-schedule" : "")
      + (seed.map { " · seed \($0)" } ?? "")


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
      prompt: resolveStudioPromptMentions(prompt),
      duration: duration,
      quality: quality,
      denoisingSteps: denoisingSteps,
      activeDiTLayers: activeDiTLayers,
      coreReuse: coreReuse,
      previewDenoise: previewDenoise,
      useBetaSchedule: selectedGenerationProfile.usesBetaSchedule,
      seed: seed,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight,
      firstFrameURL: kind == .audio ? nil : studioStartFrame?.url,
      lastFrameURL: kind == .audio ? nil : studioEndFrame?.url,
      referenceImageURLs: kind == .audio ? [] : studioReferenceImages.map(\.url)
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
            studioResults.insert(
              GenerationResult(
                id: UUID(),
                asset: asset,
                kind: kind,
                prompt: prompt,
                createdAt: Date()
              ),
              at: 0
            )
            completedAssetID = asset.id
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

  var programDuration: TimeInterval {
    max(project.timeline.visualDuration, project.timeline.audioTrackEnd)
  }

  var visualLaneAudible: Bool {
    !visualTrackMuted
  }

  var audioLaneAudible: Bool {
    !audioTrackMuted
  }

  var latestStudioResult: GenerationResult? {
    studioResults.first
  }

  func togglePlayback() {
    playback.toggle(duration: programDuration)
    syncPlayback()
  }

  func stepPlayback(frames: Int) {
    playback.step(frames: frames, duration: programDuration)
    syncPlayback()
  }

  func skipToStart() {
    playback.skipToStart()
    syncPlayback()
  }

  func skipToEnd() {
    playback.skipToEnd(duration: programDuration)
    syncPlayback()
  }

  func seekPlayback(_ time: TimeInterval) {
    playback.seek(time, duration: programDuration)
    syncPlayback()
  }

  func syncPlayback() {
    playback.sync(
      project: project,
      visualMuted: !visualLaneAudible,
      audioMuted: !audioLaneAudible
    )
  }

  func setTimelineZoom(_ zoom: Double) {
    timelineZoom = TimelineRuler.clampZoom(zoom)
  }

  func adjustTimelineZoom(_ delta: Double) {
    setTimelineZoom(timelineZoom + delta)
  }

  func insertToTimeline(_ result: GenerationResult) {
    do {
      var asset = result.asset
      if result.kind == .image {
        asset.duration = 3
      }
      try append(asset)
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  @discardableResult
  func receiveDroppedFiles(_ urls: [URL], onto lane: MediaImportLane) -> Bool {
    let split = MediaImport.partition(urls, onto: lane)
    guard !split.accepted.isEmpty else { return false }
    Task {
      await importFiles(split.accepted, onto: lane)
      if !split.rejected.isEmpty {
        importErrorMessage = MediaImportError.wrongLane.errorDescription
      }
    }
    return true
  }

  func importFiles(_ urls: [URL], onto lane: MediaImportLane) async {
    importErrorMessage = nil
    let directory = Self.importedMediaDirectory
    var lastError: String?
    for url in urls {
      do {
        let imported = try await MediaImport.makeAsset(
          from: url,
          onto: lane,
          copyingInto: directory
        )
        try appendImported(imported)
      } catch {
        lastError = error.localizedDescription
      }
    }
    playback.clock.setTime(playback.clock.currentTime, duration: programDuration)
    syncPlayback()
    if let lastError {
      importErrorMessage = lastError
    }
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
    playback.shutdown()
    engineSession.shutdown()
    endModelAccess()
  }

  func status(for manifest: ModelPackageManifest) -> ManagedPackageStatus {
    managedStatuses[manifest.id] ?? ManagedPackageStatus()
  }

  var anyManagedDownloadIsActive: Bool {
    managedStatuses.values.contains { $0.downloadIsActive }
  }

  func downloadManagedModel() {
    downloadManagedModel(managedModel)
  }

  func downloadManagedModel(_ manifest: ModelPackageManifest) {
    guard !anyManagedDownloadIsActive else { return }
    modelDownloadTask?.cancel()
    managedStatuses[manifest.id, default: ManagedPackageStatus()].state = .downloading
    managedStatuses[manifest.id]?.message = "Preparing the package…"

    let downloader = modelDownloader
    modelDownloadTask = Task { [weak self] in
      guard let self else { return }
      do {
        let installedURL = try await downloader.download(manifest) { [weak self] progress in
          Task { @MainActor [weak self] in
            self?.applyManagedModelProgress(progress, manifest: manifest)
          }
        }
        try Task.checkCancellation()
        managedStatuses[manifest.id] = ManagedPackageStatus(
          state: .installed,
          progress: 1,
          completedBytes: manifest.totalByteCount,
          message: installedMessage(for: manifest),
          installedURL: installedURL
        )
        refreshModelChoices()
      } catch is CancellationError {
        managedStatuses[manifest.id]?.state = .cancelled
        managedStatuses[manifest.id]?.message =
          "Paused. Downloaded data is kept for a later resume."
      } catch {
        managedStatuses[manifest.id]?.state = .failed
        managedStatuses[manifest.id]?.message = error.localizedDescription
      }
    }
  }

  func cancelManagedModelDownload() {
    modelDownloadTask?.cancel()
    modelDownloadTask = nil
    for id in managedStatuses.keys where managedStatuses[id]?.downloadIsActive == true {
      managedStatuses[id]?.state = .cancelled
      managedStatuses[id]?.message = "Pausing… downloaded data will be kept."
    }
  }

  func refreshManagedModelStatus() {
    guard !anyManagedDownloadIsActive else { return }
    modelDownloadTask?.cancel()
    let manifests = managedManifests
    let downloader = modelDownloader
    for manifest in manifests where managedStatuses[manifest.id] == nil {
      managedStatuses[manifest.id] = ManagedPackageStatus()
    }
    modelDownloadTask = Task { [weak self] in
      for manifest in manifests {
        guard let self else { return }
        if let installedURL = await downloader.installedPackageURL(for: manifest) {
          managedStatuses[manifest.id] = ManagedPackageStatus(
            state: .installed,
            progress: 1,
            completedBytes: manifest.totalByteCount,
            message: installedMessage(for: manifest),
            installedURL: installedURL
          )
          continue
        }
        do {
          let stagedBytes = try await downloader.stagedByteCount(for: manifest)
          try Task.checkCancellation()
          managedStatuses[manifest.id] = ManagedPackageStatus(
            state: stagedBytes > 0 ? .cancelled : .available,
            progress: manifest.totalByteCount > 0
              ? Double(stagedBytes) / Double(manifest.totalByteCount) : 0,
            completedBytes: stagedBytes,
            message: stagedBytes > 0
              ? "A partial download is available and can be resumed."
              : availableMessage(for: manifest)
          )
        } catch {
          managedStatuses[manifest.id]?.state = .failed
          managedStatuses[manifest.id]?.message = error.localizedDescription
        }
      }
      self?.refreshModelChoices()
      self?.restoreSelectedModelIfNeeded()
    }
  }

  private func installedMessage(for manifest: ModelPackageManifest) -> String {
    manifest.compatibility == .ready
      ? "Installed and ready for local generation."
      : "Installed, but this engine build still requires a checkpoint adapter."
  }

  private func availableMessage(for manifest: ModelPackageManifest) -> String {
    if manifest.generationProfile == .turbo {
      return "Installs instantly from the local conversion; shared files are reused."
    }
    return "Pinned INT8 package recommended for Macs with 32 GB or more."
  }

  private func applyManagedModelProgress(
    _ progress: ModelDownloadProgress,
    manifest: ModelPackageManifest
  ) {
    managedStatuses[manifest.id]?.completedBytes = progress.completedBytes
    managedStatuses[manifest.id]?.progress = progress.fractionCompleted
    let (state, message): (ManagedModelState, String) =
      switch progress.phase {
      case .preparing:
        (.downloading, "Checking free space and existing files…")
      case .downloading:
        (.downloading, progress.currentFileName.map { "Downloading \($0)…" } ?? "Downloading…")
      case .verifying:
        (.verifying, progress.currentFileName.map { "Verifying \($0)…" } ?? "Verifying…")
      case .installing:
        (.installing, "Installing the verified package…")
      case .completed:
        (.installed, "The model package is installed.")
      case .cancelled:
        (.cancelled, "Paused. Downloaded data is kept for a later resume.")
      }
    managedStatuses[manifest.id]?.state = state
    managedStatuses[manifest.id]?.message = message
  }

  // MARK: - Model library

  var selectedModelChoice: ModelChoice? {
    modelChoices.first { $0.id == selectedModelID }
  }

  var installedModelChoices: [ModelChoice] {
    modelChoices.filter(\.isInstalled)
  }

  var selectedGenerationProfile: ModelGenerationProfile {
    selectedModelChoice?.generationProfile ?? .standard
  }

  func selectModel(_ id: String) {
    guard
      let choice = modelChoices.first(where: { $0.id == id }),
      let directory = choice.directory
    else { return }
    selectedModelID = id
    selectModelDirectory(directory)
  }

  func addLocalModelFolder(_ url: URL) {
    guard
      let bookmark = try? url.bookmarkData(
        options: .withSecurityScope,
        includingResourceValuesForKeys: nil,
        relativeTo: nil
      )
    else {
      errorMessage = "Cannot keep access to \(url.lastPathComponent); choose it again."
      return
    }
    localModelBookmarks.append(bookmark)
    refreshModelChoices()
    if let added = modelChoices.last(where: { $0.isLocalFolder && $0.directory == url }) {
      selectModel(added.id)
    }
  }

  func removeLocalModel(_ id: String) {
    guard let choice = modelChoices.first(where: { $0.id == id }), choice.isLocalFolder,
      case .localFolder(let bookmark) = choice.source
    else { return }
    localModelBookmarks.removeAll { $0 == bookmark }
    if selectedModelID == id {
      selectedModelID = nil
      clearModelDirectory()
    }
    refreshModelChoices()
  }

  private func refreshModelChoices() {
    var choices: [ModelChoice] = managedManifests.map { manifest in
      ModelChoice(
        id: manifest.id,
        displayName: manifest.displayName,
        subtitle: manifest.generationProfile == .turbo
          ? "Fastest · loose prompt control"
          : "Balanced · follows prompts",
        source: .managed(manifest),
        directory: managedStatuses[manifest.id]?.installedURL,
        generationProfile: manifest.generationProfile
      )
    }
    for bookmark in localModelBookmarks {
      var isStale = false
      guard
        let url = try? URL(
          resolvingBookmarkData: bookmark,
          options: .withSecurityScope,
          relativeTo: nil,
          bookmarkDataIsStale: &isStale
        )
      else { continue }
      choices.append(
        ModelChoice(
          id: "local:" + url.path,
          displayName: url.lastPathComponent,
          subtitle: "Local folder",
          source: .localFolder(bookmark: bookmark),
          directory: url,
          generationProfile: .standard
        )
      )
    }
    modelChoices = choices
  }

  private func restoreModelLibrary() {
    localModelBookmarks =
      userDefaults.array(forKey: Self.localModelsKey) as? [Data] ?? []
    selectedModelID = userDefaults.string(forKey: Self.selectedModelKey)
    // A pre-picker install selected one folder through a single bookmark;
    // carry it into the library as a local folder.
    if localModelBookmarks.isEmpty, selectedModelID == nil,
      let legacy = userDefaults.data(forKey: Self.modelBookmarkKey)
    {
      localModelBookmarks = [legacy]
      userDefaults.removeObject(forKey: Self.modelBookmarkKey)
    }
    refreshModelChoices()
    if let selected = selectedModelChoice, selected.isLocalFolder,
      let directory = selected.directory
    {
      selectModelDirectory(directory)
    } else if selectedModelID == nil,
      let onlyLocal = modelChoices.first(where: \.isLocalFolder),
      let directory = onlyLocal.directory
    {
      selectedModelID = onlyLocal.id
      selectModelDirectory(directory)
    }
  }

  /// Managed installs resolve asynchronously; re-apply a persisted managed
  /// selection once its installed directory is known.
  private func restoreSelectedModelIfNeeded() {
    guard modelDirectory == nil, let selected = selectedModelChoice,
      let directory = selected.directory
    else { return }
    selectModelDirectory(directory)
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

  func setVisualCanvasFit(_ id: UUID, _ fit: CanvasFit) {
    project.timeline.setVisualCanvasFit(id, fit)
  }

  func rotateVisual(_ id: UUID) {
    project.timeline.rotateVisual(id)
  }

  func toggleAudio(_ id: UUID) {
    guard let item = project.timeline.audioItems.first(where: { $0.id == id }) else {
      return
    }
    project.timeline.setAudioEnabled(id, isEnabled: !item.isEnabled)
  }

  func applyAudioTrim(
    _ id: UUID,
    edge: TimelineTrimEdge,
    delta: TimeInterval,
    origin: AudioTrim,
    sourceLimit: TimeInterval,
    earliestStart: TimeInterval,
    latestEnd: TimeInterval
  ) {
    let trim = AudioTrimMath.apply(
      edge: edge,
      delta: delta,
      startTime: origin.startTime,
      duration: origin.duration,
      sourceOffset: origin.sourceOffset,
      sourceLimit: sourceLimit,
      earliestStart: earliestStart,
      latestEnd: latestEnd,
      minimumDuration: VisualTrimMath.minimumDuration(
        framesPerSecond: project.settings.framesPerSecond
      )
    )
    project.timeline.setAudioTrim(id, trim)
    playback.clock.setTime(playback.clock.currentTime, duration: programDuration)
    syncPlayback()
  }

  func applyVisualTrim(
    _ id: UUID,
    edge: VisualTrimEdge,
    delta: TimeInterval,
    origin: VisualTrim,
    startTime: TimeInterval,
    sourceLimit: TimeInterval?
  ) {
    let trim = VisualTrimMath.apply(
      edge: edge,
      delta: delta,
      startTime: startTime,
      duration: origin.duration,
      sourceOffset: origin.sourceOffset,
      gapBefore: origin.gapBefore,
      sourceLimit: sourceLimit,
      minimumDuration: VisualTrimMath.minimumDuration(
        framesPerSecond: project.settings.framesPerSecond
      )
    )
    project.timeline.setVisualTrim(id, trim)
    playback.clock.setTime(playback.clock.currentTime, duration: programDuration)
    syncPlayback()
  }

  func removeVisual(_ id: UUID) {
    project.timeline.removeVisual(id)
    if selectedTimelineItem == .visual(id) {
      selectedTimelineItem = nil
    }
    playback.clock.setTime(playback.clock.currentTime, duration: programDuration)
  }

  func removeAudio(_ id: UUID) {
    project.timeline.removeAudio(id)
    if selectedTimelineItem == .audio(id) {
      selectedTimelineItem = nil
    }
  }

  var canSplitSelectedAtPlayhead: Bool {
    canSplit(selectedTimelineItem)
  }

  func canSplit(_ item: TimelineItemID?) -> Bool {
    guard let item else { return false }
    let time = playback.clock.currentTime
    let fps = project.settings.framesPerSecond
    switch item {
    case .visual(let id):
      return project.timeline.canSplitVisual(id, at: time, framesPerSecond: fps)
    case .audio(let id):
      return project.timeline.canSplitAudio(id, at: time, framesPerSecond: fps)
    }
  }

  func splitSelectedAtPlayhead() {
    split(selectedTimelineItem)
  }

  func split(_ item: TimelineItemID?) {
    guard let item else { return }
    let time = playback.clock.currentTime
    let fps = project.settings.framesPerSecond
    switch item {
    case .visual(let id):
      guard let visual = project.timeline.visualItems.first(where: { $0.id == id }) else {
        return
      }
      let kind = project.asset(id: visual.assetID)?.kind ?? .video
      guard
        let split = project.timeline.splitVisual(
          id,
          at: time,
          sourceKind: kind,
          framesPerSecond: fps
        )
      else { return }
      selectedTimelineItem = .visual(split.left)
    case .audio(let id):
      guard let split = project.timeline.splitAudio(id, at: time, framesPerSecond: fps) else {
        return
      }
      selectedTimelineItem = .audio(split.left)
    }
    playback.clock.setTime(playback.clock.currentTime, duration: programDuration)
    syncPlayback()
  }

  func deleteSelectedTimelineItem() {
    guard let selectedTimelineItem else { return }
    switch selectedTimelineItem {
    case .visual(let id):
      removeVisual(id)
    case .audio(let id):
      removeAudio(id)
    }
  }

  private func append(_ asset: AssetReference) throws {
    project.addAsset(asset)
    if asset.kind == .audio {
      try project.timeline.appendAudio(asset)
    } else {
      try project.timeline.appendVisual(asset)
    }
  }

  private func appendImported(_ imported: ImportedMedia) throws {
    project.addAsset(imported.asset)
    if imported.asset.kind == .audio {
      try project.timeline.appendAudio(imported.asset)
    } else {
      try project.timeline.appendVisual(
        imported.asset,
        includesNativeAudio: imported.includesNativeAudio
      )
    }
  }

  private static var importedMediaDirectory: URL {
    URL.applicationSupportDirectory
      .appendingPathComponent("H3ddle", isDirectory: true)
      .appendingPathComponent("Media", isDirectory: true)
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
      DockAttention.markGenerationFinished()
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
      kind == .audio ? Self.audioCanvasLabel : "\(quality.canvasSize)×\(quality.canvasSize)"
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

}
