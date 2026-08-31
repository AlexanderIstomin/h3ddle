import Foundation
import H3ddleCore

public enum GenerationJobState: String, Codable, CaseIterable, Sendable {
  case queued
  case preparing
  case running
  case paused
  case blocked
  case failed
  case completed
  case cancelled

  public var isActive: Bool { self == .preparing || self == .running }

  public var isFinished: Bool {
    switch self {
    case .blocked, .failed, .completed, .cancelled: true
    case .queued, .preparing, .running, .paused: false
    }
  }
}

/// One durable execution snapshot. The request is never mutated while the
/// job is active; editing replaces it only after the scheduler has stopped
/// that job. The extra studio fields are presentation state, retained so an
/// edit returns to the same controls rather than exposing the composed engine
/// prompt or reconstructing lossy defaults.
public struct GenerationQueueJob: Identifiable, Hashable, Codable, Sendable {
  public var id: UUID
  public var request: GenerationRequest
  public var modelID: String?
  public var modelDirectory: URL?
  public var displayPrompt: String
  public var settingsDescription: String
  public var studioSettings: GenerationStudioSettings
  public var aspectRatio: String
  public var soundscape: String
  public var music: String
  public var createdAt: Date
  public var startedAt: Date?
  public var finishedAt: Date?
  public var state: GenerationJobState
  /// A queued job may merely be saved for later. Scheduled jobs are the ones
  /// the single worker is allowed to pick without another user action.
  public var isScheduled: Bool
  public var phase: String
  public var overallProgress: Double
  public var elapsed: TimeInterval
  public var projectedDuration: TimeInterval?
  public var errorMessage: String?
  public var result: AssetReference?
  public var statistics: GenerationStatistics?
  /// Optional clip-local destination. The generated asset is always retained
  /// in the library; this target only controls an additional safe swap.
  public var replacementTarget: GenerationReplacementTarget?
  public var replacementMessage: String?

  public init(
    id: UUID = UUID(),
    request: GenerationRequest,
    modelID: String? = nil,
    modelDirectory: URL? = nil,
    displayPrompt: String,
    settingsDescription: String,
    studioSettings: GenerationStudioSettings,
    aspectRatio: String,
    soundscape: String = "",
    music: String = "",
    createdAt: Date = Date(),
    startedAt: Date? = nil,
    finishedAt: Date? = nil,
    state: GenerationJobState = .queued,
    isScheduled: Bool = false,
    phase: String = "Waiting",
    overallProgress: Double = 0,
    elapsed: TimeInterval = 0,
    projectedDuration: TimeInterval? = nil,
    errorMessage: String? = nil,
    result: AssetReference? = nil,
    statistics: GenerationStatistics? = nil,
    replacementTarget: GenerationReplacementTarget? = nil,
    replacementMessage: String? = nil
  ) {
    self.id = id
    self.request = request
    self.modelID = modelID
    self.modelDirectory = modelDirectory
    self.displayPrompt = displayPrompt
    self.settingsDescription = settingsDescription
    self.studioSettings = studioSettings
    self.aspectRatio = aspectRatio
    self.soundscape = soundscape
    self.music = music
    self.createdAt = createdAt
    self.startedAt = startedAt
    self.finishedAt = finishedAt
    self.state = state
    self.isScheduled = isScheduled
    self.phase = phase
    self.overallProgress = min(1, max(0, overallProgress))
    self.elapsed = max(0, elapsed)
    self.projectedDuration = projectedDuration
    self.errorMessage = errorMessage
    self.result = result
    self.statistics = statistics
    self.replacementTarget = replacementTarget
    self.replacementMessage = replacementMessage
  }

  /// Checkpoint support is decided by the exact request. An H3 model with an
  /// incompatible optimization or inpainting mode deliberately has no
  /// recovery context and therefore presents Cancel rather than Pause.
  public var supportsPause: Bool { request.recovery != nil }
}

public struct GenerationQueueRun: Hashable, Codable, Sendable {
  public var id: UUID
  public var jobIDs: [UUID]
  public var startedAt: Date

  public init(id: UUID = UUID(), jobIDs: [UUID], startedAt: Date = Date()) {
    self.id = id
    self.jobIDs = jobIDs
    self.startedAt = startedAt
  }
}

/// Persistent queue state and its deterministic transitions. Execution stays
/// outside this value type; keeping ordering and run membership here makes
/// them directly testable without a model, helper process, or SwiftUI.
public struct GenerationJobQueue: Hashable, Codable, Sendable {
  public static let currentVersion = 1

  public var version: Int
  public var jobs: [GenerationQueueJob]
  public var activeRun: GenerationQueueRun?

  public init(
    version: Int = currentVersion,
    jobs: [GenerationQueueJob] = [],
    activeRun: GenerationQueueRun? = nil
  ) {
    self.version = version
    self.jobs = jobs
    self.activeRun = activeRun
  }

  public var activeJob: GenerationQueueJob? {
    jobs.first { $0.state.isActive }
  }

  public var nextScheduledJobID: UUID? {
    jobs.first { $0.state == .queued && $0.isScheduled }?.id
  }

  public var pendingCount: Int {
    jobs.count {
      switch $0.state {
      case .queued, .preparing, .running, .paused, .blocked, .failed: true
      case .completed, .cancelled: false
      }
    }
  }

  public var hasCancellableJobs: Bool {
    jobs.contains { job in
      switch job.state {
      case .queued, .preparing, .running, .paused: true
      case .blocked, .failed, .completed, .cancelled: false
      }
    }
  }

  public mutating func append(_ job: GenerationQueueJob, scheduled: Bool) {
    var job = job
    job.isScheduled = scheduled
    jobs.append(job)
    if scheduled { includeInActiveRun(job.id) }
  }

  public mutating func replace(_ job: GenerationQueueJob) {
    guard let index = jobs.firstIndex(where: { $0.id == job.id }) else { return }
    jobs[index] = job
  }

  public mutating func scheduleAll() {
    let ids = jobs.indices.compactMap { index -> UUID? in
      guard jobs[index].state == .queued else { return nil }
      jobs[index].isScheduled = true
      return jobs[index].id
    }
    includeInActiveRun(ids)
  }

  public mutating func schedule(_ id: UUID, next: Bool) {
    guard let index = jobs.firstIndex(where: { $0.id == id }),
      jobs[index].state == .queued
    else { return }
    jobs[index].isScheduled = true
    includeInActiveRun(id)
    if next { moveQueuedJobToFront(id) }
  }

  /// Cancels every job that could still run or resume. Terminal entries stay
  /// untouched so completed output and failure diagnostics remain available.
  /// The returned snapshots let the execution layer discard any managed
  /// checkpoints after this atomic queue transition.
  @discardableResult
  public mutating func cancelAll(at date: Date = Date()) -> [GenerationQueueJob] {
    let cancelled = jobs.filter { job in
      switch job.state {
      case .queued, .preparing, .running, .paused: true
      case .blocked, .failed, .completed, .cancelled: false
      }
    }
    let cancelledIDs = Set(cancelled.map(\.id))
    for index in jobs.indices where cancelledIDs.contains(jobs[index].id) {
      jobs[index].state = .cancelled
      jobs[index].isScheduled = false
      jobs[index].phase = "Cancelled"
      jobs[index].finishedAt = date
    }
    finishRunIfNeeded()
    return cancelled
  }

  public mutating func moveQueuedJobToFront(_ id: UUID) {
    guard let source = jobs.firstIndex(where: { $0.id == id && $0.state == .queued })
    else { return }
    let job = jobs.remove(at: source)
    let target = jobs.firstIndex { $0.state == .queued } ?? jobs.endIndex
    jobs.insert(job, at: target)
  }

  public mutating func moveQueuedJob(_ id: UUID, by offset: Int) {
    guard offset != 0,
      let source = jobs.firstIndex(where: { $0.id == id && $0.state == .queued })
    else { return }
    let queued = jobs.indices.filter { jobs[$0].state == .queued }
    guard let position = queued.firstIndex(of: source) else { return }
    let destinationPosition = min(max(0, position + offset), queued.count - 1)
    guard destinationPosition != position else { return }
    let destination = queued[destinationPosition]
    let job = jobs.remove(at: source)
    jobs.insert(job, at: source < destination ? destination : destination)
  }

  @discardableResult
  public mutating func remove(_ id: UUID) -> GenerationQueueJob? {
    guard let index = jobs.firstIndex(where: { $0.id == id }),
      !jobs[index].state.isActive
    else { return nil }
    let removed = jobs.remove(at: index)
    if let runIndex = activeRun?.jobIDs.firstIndex(of: id) {
      activeRun?.jobIDs.remove(at: runIndex)
    }
    finishRunIfNeeded()
    return removed
  }

  /// A persisted active job means the app or helper disappeared between
  /// terminal events. Requeue it with its previous scheduling intent: H3 can
  /// use its checkpoint; other engines safely restart from the beginning.
  public mutating func prepareForRelaunch() {
    var restoredIDs: [UUID] = []
    for index in jobs.indices where jobs[index].state.isActive {
      jobs[index].state = .queued
      jobs[index].isScheduled = true
      jobs[index].phase = jobs[index].supportsPause
        ? "Restoring interrupted generation" : "Restarting interrupted generation"
      jobs[index].overallProgress = 0
      jobs[index].startedAt = nil
      jobs[index].finishedAt = nil
      jobs[index].errorMessage = nil
      restoredIDs.append(jobs[index].id)
    }
    includeInActiveRun(restoredIDs)
  }

  public mutating func finishRunIfNeeded() {
    guard let activeRun else { return }
    let stillWorking = activeRun.jobIDs.contains { id in
      guard let job = jobs.first(where: { $0.id == id }) else { return false }
      return job.state.isActive || (job.state == .queued && job.isScheduled)
    }
    if !stillWorking { self.activeRun = nil }
  }

  /// Whole-queue progress for the current scheduled run. A nil result means
  /// at least one job has no honest duration estimate; callers should show
  /// the job count and indeterminate activity instead of a false percentage.
  public var activeRunProgress: Double? {
    guard let activeRun, !activeRun.jobIDs.isEmpty else { return nil }
    let runJobs = activeRun.jobIDs.compactMap { id in jobs.first { $0.id == id } }
    guard runJobs.count == activeRun.jobIDs.count,
      runJobs.allSatisfy({ ($0.projectedDuration ?? 0) > 0 })
    else { return nil }
    let total = runJobs.reduce(0) { $0 + ($1.projectedDuration ?? 0) }
    guard total > 0 else { return nil }
    let completed = runJobs.reduce(0.0) { partial, job in
      let fraction: Double = switch job.state {
      case .completed, .cancelled, .failed, .blocked, .paused: 1
      case .preparing, .running: job.overallProgress
      case .queued: 0
      }
      return partial + (job.projectedDuration ?? 0) * fraction
    }
    return min(1, max(0, completed / total))
  }

  public var activeRunPosition: (completed: Int, total: Int)? {
    guard let activeRun, !activeRun.jobIDs.isEmpty else { return nil }
    let completed = activeRun.jobIDs.count { id in
      guard let job = jobs.first(where: { $0.id == id }) else { return true }
      return !job.state.isActive && !(job.state == .queued && job.isScheduled)
    }
    return (completed, activeRun.jobIDs.count)
  }

  private mutating func includeInActiveRun(_ id: UUID) {
    includeInActiveRun([id])
  }

  private mutating func includeInActiveRun(_ ids: [UUID]) {
    guard !ids.isEmpty else { return }
    if activeRun == nil { activeRun = GenerationQueueRun(jobIDs: []) }
    for id in ids where activeRun?.jobIDs.contains(id) != true {
      activeRun?.jobIDs.append(id)
    }
  }
}

/// Atomic JSON persistence and per-job managed files. Queue metadata is small;
/// generated assets and H3 sampler checkpoints remain in their job folders.
public struct GenerationQueueStore: Sendable {
  public static let documentName = "queue.json"

  public var rootURL: URL

  public init(
    rootURL: URL = URL.applicationSupportDirectory
      .appendingPathComponent("H3ddle", isDirectory: true)
      .appendingPathComponent("GenerationQueue", isDirectory: true)
  ) {
    self.rootURL = rootURL
  }

  public var documentURL: URL {
    rootURL.appendingPathComponent(Self.documentName)
  }

  public func jobDirectory(for id: UUID) -> URL {
    rootURL.appendingPathComponent("Jobs", isDirectory: true)
      .appendingPathComponent(id.uuidString, isDirectory: true)
  }

  public func load() -> GenerationJobQueue {
    let decoder = JSONDecoder()
    guard let data = try? Data(contentsOf: documentURL),
      let queue = try? decoder.decode(GenerationJobQueue.self, from: data),
      queue.version <= GenerationJobQueue.currentVersion
    else { return GenerationJobQueue() }
    return queue
  }

  public func save(_ queue: GenerationJobQueue) throws {
    try FileManager.default.createDirectory(
      at: rootURL,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(queue)
    try data.write(to: documentURL, options: .atomic)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o600], ofItemAtPath: documentURL.path)
  }

  public func stableOutputURL(for id: UUID, sourceURL: URL) throws -> URL {
    let directory = jobDirectory(for: id)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    let pathExtension = sourceURL.pathExtension.isEmpty ? "output" : sourceURL.pathExtension
    return directory.appendingPathComponent("output").appendingPathExtension(pathExtension)
  }

  public func discardFiles(for id: UUID) {
    try? FileManager.default.removeItem(at: jobDirectory(for: id))
  }
}
