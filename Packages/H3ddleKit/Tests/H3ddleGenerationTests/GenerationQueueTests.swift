import Foundation
import H3ddleCore
import Testing

@testable import H3ddleGeneration

@Suite("Generation job queue")
struct GenerationQueueTests {
  private func job(
    prompt: String,
    state: GenerationJobState = .queued,
    estimate: TimeInterval? = 100,
    progress: Double = 0,
    recovery: Bool = false
  ) -> GenerationQueueJob {
    let id = UUID()
    let request = GenerationRequest(
      kind: .image,
      prompt: prompt,
      duration: 3,
      recovery: recovery
        ? GenerationRecoveryContext(
          jobID: id,
          directoryURL: URL(fileURLWithPath: "/tmp/\(id.uuidString)"),
          outputURL: URL(fileURLWithPath: "/tmp/\(id.uuidString).png")
        ) : nil
    )
    return GenerationQueueJob(
      id: id,
      request: request,
      displayPrompt: prompt,
      settingsDescription: "image",
      studioSettings: .makeDefault(seed: 7),
      aspectRatio: "1:1",
      state: state,
      overallProgress: progress,
      projectedDuration: estimate
    )
  }

  @Test("Saved jobs do not run until scheduled")
  func savedVersusScheduled() {
    var queue = GenerationJobQueue()
    let saved = job(prompt: "saved")
    let runningNext = job(prompt: "scheduled")
    queue.append(saved, scheduled: false)
    queue.append(runningNext, scheduled: true)

    #expect(queue.nextScheduledJobID == runningNext.id)
    #expect(queue.jobs.first?.isScheduled == false)
    #expect(queue.activeRun?.jobIDs == [runningNext.id])
  }

  @Test("Run next changes priority without disturbing active work")
  func runNext() {
    var queue = GenerationJobQueue()
    let active = job(prompt: "active", state: .running)
    let first = job(prompt: "first")
    let next = job(prompt: "next")
    queue.append(active, scheduled: true)
    queue.append(first, scheduled: true)
    queue.append(next, scheduled: false)

    queue.schedule(next.id, next: true)

    #expect(queue.jobs.first?.id == active.id)
    #expect(queue.jobs.dropFirst().first?.id == next.id)
    #expect(queue.nextScheduledJobID == next.id)
  }

  @Test("Run all schedules only waiting jobs")
  func runAll() {
    var queue = GenerationJobQueue()
    let first = job(prompt: "first")
    let paused = job(prompt: "paused", state: .paused, recovery: true)
    let failed = job(prompt: "failed", state: .failed)
    queue.append(first, scheduled: false)
    queue.append(paused, scheduled: false)
    queue.append(failed, scheduled: false)

    queue.scheduleAll()

    #expect(queue.jobs[0].isScheduled)
    #expect(!queue.jobs[1].isScheduled)
    #expect(!queue.jobs[2].isScheduled)
    #expect(queue.activeRun?.jobIDs == [first.id])
  }

  @Test("Cancel all stops runnable work and preserves terminal history")
  func cancelAll() {
    var queue = GenerationJobQueue()
    let queued = job(prompt: "queued")
    let running = job(prompt: "running", state: .running)
    let paused = job(prompt: "paused", state: .paused, recovery: true)
    let failed = job(prompt: "failed", state: .failed)
    let completed = job(prompt: "completed", state: .completed)
    queue.append(queued, scheduled: true)
    queue.append(running, scheduled: true)
    queue.append(paused, scheduled: false)
    queue.append(failed, scheduled: false)
    queue.append(completed, scheduled: false)
    let finishedAt = Date(timeIntervalSince1970: 123)

    let cancelled = queue.cancelAll(at: finishedAt)

    #expect(Set(cancelled.map(\.id)) == Set([queued.id, running.id, paused.id]))
    #expect(queue.jobs[0].state == .cancelled)
    #expect(queue.jobs[1].state == .cancelled)
    #expect(queue.jobs[2].state == .cancelled)
    #expect(queue.jobs.prefix(3).allSatisfy { !$0.isScheduled })
    #expect(queue.jobs.prefix(3).allSatisfy { $0.finishedAt == finishedAt })
    #expect(queue.jobs[3].state == .failed)
    #expect(queue.jobs[4].state == .completed)
    #expect(queue.activeRun == nil)
    #expect(!queue.hasCancellableJobs)
  }

  @Test("Relaunch requeues active work and retains pause capability")
  func relaunch() {
    var queue = GenerationJobQueue()
    let active = job(prompt: "recover", state: .running, progress: 0.6, recovery: true)
    queue.append(active, scheduled: true)

    queue.prepareForRelaunch()

    let restored = queue.jobs[0]
    #expect(restored.state == .queued)
    #expect(restored.isScheduled)
    #expect(restored.supportsPause)
    #expect(restored.overallProgress == 0)
    #expect(restored.phase == "Restoring interrupted generation")
  }

  @Test("Aggregate progress is weighted by expected duration")
  func weightedProgress() {
    var queue = GenerationJobQueue()
    let short = job(prompt: "short", state: .completed, estimate: 100, progress: 1)
    let long = job(prompt: "long", state: .running, estimate: 300, progress: 0.5)
    queue.append(short, scheduled: true)
    queue.append(long, scheduled: true)

    #expect(queue.activeRunProgress == 0.625)
    #expect(queue.activeRunPosition?.completed == 1)
    #expect(queue.activeRunPosition?.total == 2)
  }

  @Test("Unknown estimates make aggregate progress indeterminate")
  func indeterminateProgress() {
    var queue = GenerationJobQueue()
    queue.append(job(prompt: "known", estimate: 100), scheduled: true)
    queue.append(job(prompt: "unknown", estimate: nil), scheduled: true)
    #expect(queue.activeRunProgress == nil)
  }

  @Test("Queue store round trips atomically")
  func persistence() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleQueueTests-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = GenerationQueueStore(rootURL: root)
    var queue = GenerationJobQueue()
    var original = job(prompt: "persist me", recovery: true)
    original.modelID = "saved-model-v1"
    original.modelDirectory = URL(fileURLWithPath: "/tmp/saved-model-v1", isDirectory: true)
    queue.append(original, scheduled: true)

    try store.save(queue)
    let restored = store.load()

    #expect(restored == queue)
    #expect(FileManager.default.fileExists(atPath: store.documentURL.path))
  }

  @Test("Clip replacement targets survive queue persistence")
  func replacementTargetPersists() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleQueueReplacement-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = GenerationQueueStore(rootURL: root)
    var queued = job(prompt: "final pass")
    let target = GenerationReplacementTarget(
      projectID: UUID(),
      clipID: UUID(),
      expectedAssetID: AssetID()
    )
    queued.replacementTarget = target
    var queue = GenerationJobQueue()
    queue.append(queued, scheduled: true)

    try store.save(queue)

    #expect(store.load().jobs[0].replacementTarget == target)
  }
}
