import Darwin
import Foundation
import H3ddleEngineProtocol

/// A long-lived helper process. Handshake compiles Metal once, inspect loads
/// the H3 context, and later generates reuse that process until `shutdown()`,
/// a hard cancel, or the helper exiting.
public final class EngineSession: EngineInspecting, @unchecked Sendable {
  public let executableURL: URL
  public let arguments: [String]
  /// Cooperative cancel is escalated to process kill if the job does not
  /// acknowledge within this window. Tests shorten it.
  public var cancelEscalation: Duration = .seconds(3)

  private let lock = NSLock()
  private let commandGate = NSLock()
  private let workQueue = DispatchQueue(label: "h3ddle.engine.session")
  private var process: Process?
  private var standardInput: FileHandle?
  private var standardOutput: FileHandle?
  private var standardError: FileHandle?
  private var leftover = Data()
  private var stderrLog = Data()
  private var cancelledJobs = Set<UUID>()
  private var launchCount = 0
  private var activeJobID: UUID?
  private var escalationWork: [UUID: DispatchWorkItem] = [:]

  private static let stderrLimit = 16 * 1024

  public init(executableURL: URL, arguments: [String] = []) {
    self.executableURL = executableURL
    self.arguments = arguments
  }

  deinit {
    abandon(wait: true, escalateToKill: true)
  }

  public func capabilities() async throws -> EngineCapabilities {
    let event = try await exchange(EngineCommand(kind: .handshake), failIfBusy: true)
    guard event.kind == .ready, let capabilities = event.capabilities else {
      throw EngineClientError.invalidResponse
    }
    return capabilities
  }

  public func inspectModel(at directory: URL) async throws -> EngineModelReport {
    let event = try await exchange(
      EngineCommand(
        kind: .inspectModel,
        modelInspection: EngineModelInspectionRequest(modelDirectory: directory)
      ),
      failIfBusy: true
    )
    guard event.kind == .modelInspected, let model = event.model else {
      throw EngineClientError.invalidResponse
    }
    return model
  }

  func run(command: EngineCommand, receive: (EngineEvent) -> Bool) throws {
    try run(command: command, failIfBusy: false, receive: receive)
  }

  func cancel(jobID: UUID) {
    let payload = try? EngineLineCodec.encode(
      EngineCommand(jobID: jobID, kind: .cancel)
    )
    let shouldEscalate = lock.withLock { () -> Bool in
      cancelledJobs.insert(jobID)
      if let payload {
        try? standardInput?.write(contentsOf: payload)
      }
      guard escalationWork[jobID] == nil else { return false }
      return activeJobID == jobID || activeJobID == nil
    }
    guard shouldEscalate else { return }

    let nanoseconds = UInt64(cancelEscalation.components.seconds) * 1_000_000_000
      + UInt64(cancelEscalation.components.attoseconds / 1_000_000_000)
    let work = DispatchWorkItem { [weak self] in
      self?.escalateIfNeeded(jobID: jobID)
    }
    lock.withLock { escalationWork[jobID] = work }
    workQueue.asyncAfter(
      deadline: .now() + .nanoseconds(Int(min(nanoseconds, UInt64(Int.max)))),
      execute: work
    )
  }

  public func shutdown() {
    abandon(wait: true, escalateToKill: false)
  }

  var processIdentifier: Int32? {
    lock.withLock {
      guard let process, process.isRunning else { return nil }
      return process.processIdentifier
    }
  }

  /// Helper processes launched over this session's lifetime. A relaunch is
  /// asserted through this counter rather than by comparing process
  /// identifiers, which the OS may legally hand back to the replacement.
  var helperLaunchCount: Int {
    lock.withLock { launchCount }
  }

  private func exchange(
    _ command: EngineCommand,
    failIfBusy: Bool
  ) async throws -> EngineEvent {
    try await Task.detached(priority: .userInitiated) {
      var captured: EngineEvent?
      try self.run(command: command, failIfBusy: failIfBusy) { event in
        if event.kind == .failed {
          captured = event
          return true
        }
        captured = event
        return event.kind == .ready || event.kind == .modelInspected
      }
      guard let captured else { throw EngineClientError.noResponse }
      if captured.kind == .failed {
        throw EngineClientError.rejected(
          captured.message ?? "The H3 engine rejected the request."
        )
      }
      return captured
    }.value
  }

  private func run(
    command: EngineCommand,
    failIfBusy: Bool,
    receive: (EngineEvent) -> Bool
  ) throws {
    if failIfBusy {
      guard commandGate.lock(before: Date()) else {
        throw EngineClientError.busy
      }
    } else {
      preemptCancelledJob()
      commandGate.lock()
    }
    defer { commandGate.unlock() }

    if let jobID = command.jobID {
      let cancelled = lock.withLock { cancelledJobs.contains(jobID) }
      if cancelled {
        throw CancellationError()
      }
    }

    try ensureRunning()
    lock.withLock { activeJobID = command.jobID }
    defer { finishJob(command.jobID) }

    try send(command)
    var receivedTerminalEvent = false

    while !receivedTerminalEvent {
      let output = lock.withLock { standardOutput }
      let data = output?.availableData ?? Data()
      if data.isEmpty {
        let cancelled = command.jobID.map { jobID in
          lock.withLock { cancelledJobs.contains(jobID) }
        } ?? false
        let message = stderrSnapshot()
        markProcessDead()
        if cancelled {
          throw CancellationError()
        }
        throw EngineGenerationProviderError.engineStopped(
          message
            ?? "The H3 engine exited before the request completed."
        )
      }

      leftover.append(data)
      while let newline = leftover.firstIndex(of: 0x0A) {
        let line = leftover[..<newline]
        leftover.removeSubrange(...newline)
        guard
          let event = try? EngineLineCodec.decode(EngineEvent.self, from: Data(line)),
          event.requestID == command.requestID
        else {
          continue
        }
        receivedTerminalEvent = receive(event)
        if receivedTerminalEvent { break }
      }
    }
  }

  private func ensureRunning() throws {
    if lock.withLock({ self.process?.isRunning == true }) { return }
    guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
      throw EngineClientError.executableMissing(executableURL)
    }

    let process = Process()
    let inputPipe = Pipe()
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    process.executableURL = executableURL
    if !arguments.isEmpty {
      process.arguments = arguments
    }
    process.standardInput = inputPipe
    process.standardOutput = outputPipe
    process.standardError = errorPipe
    EngineProcessConfiguration.apply(to: process, executableURL: executableURL)

    do {
      try process.run()
    } catch {
      throw EngineClientError.launchFailed(error.localizedDescription)
    }

    let errorHandle = errorPipe.fileHandleForReading
    errorHandle.readabilityHandler = { [weak self] handle in
      self?.appendStderr(handle.availableData)
    }

    lock.withLock {
      self.process = process
      launchCount += 1
      standardInput = inputPipe.fileHandleForWriting
      standardOutput = outputPipe.fileHandleForReading
      standardError = errorHandle
      leftover = Data()
      stderrLog.removeAll(keepingCapacity: true)
    }
  }

  private func send(_ command: EngineCommand) throws {
    let data = try EngineLineCodec.encode(command)
    try lock.withLock {
      guard let standardInput else {
        throw EngineClientError.launchFailed("The H3 engine helper is not running.")
      }
      try standardInput.write(contentsOf: data)
    }
  }

  /// A new generate must not wait out the escalation window behind a job
  /// the user already cancelled. Kill that helper now so `commandGate` frees.
  private func preemptCancelledJob() {
    let process = lock.withLock { () -> Process? in
      guard let jobID = activeJobID, cancelledJobs.contains(jobID) else { return nil }
      escalationWork.removeValue(forKey: jobID)?.cancel()
      try? standardInput?.close()
      standardInput = nil
      return self.process
    }
    guard let process, process.isRunning else { return }
    kill(process.processIdentifier, SIGKILL)
    process.waitUntilExit()
  }

  private func escalateIfNeeded(jobID: UUID) {
    let process = lock.withLock { () -> Process? in
      guard cancelledJobs.contains(jobID), activeJobID == jobID else { return nil }
      escalationWork[jobID] = nil
      try? standardInput?.close()
      standardInput = nil
      return self.process
    }
    guard let process, process.isRunning else { return }
    process.terminate()
  }

  private func finishJob(_ jobID: UUID?) {
    lock.withLock {
      if activeJobID == jobID {
        activeJobID = nil
      }
      if let jobID {
        cancelledJobs.remove(jobID)
        escalationWork.removeValue(forKey: jobID)?.cancel()
      }
    }
  }

  private func appendStderr(_ data: Data) {
    guard !data.isEmpty else { return }
    lock.lock()
    stderrLog.append(data)
    if stderrLog.count > Self.stderrLimit {
      stderrLog.removeSubrange(..<(stderrLog.count - Self.stderrLimit))
    }
    lock.unlock()
  }

  private func stderrSnapshot() -> String? {
    lock.withLock {
      let text = String(data: stderrLog, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
      return text.flatMap { $0.isEmpty ? nil : $0 }
    }
  }

  private func markProcessDead() {
    lock.withLock {
      standardError?.readabilityHandler = nil
      process = nil
      standardInput = nil
      standardOutput = nil
      standardError = nil
      leftover = Data()
      activeJobID = nil
    }
  }

  private func abandon(wait: Bool, escalateToKill: Bool) {
    let process = lock.withLock { () -> Process? in
      if let data = try? EngineLineCodec.encode(EngineCommand(kind: .shutdown)) {
        try? standardInput?.write(contentsOf: data)
      }
      try? standardInput?.close()
      standardError?.readabilityHandler = nil
      standardInput = nil
      leftover = Data()
      cancelledJobs.removeAll()
      escalationWork.values.forEach { $0.cancel() }
      escalationWork.removeAll()
      activeJobID = nil
      defer {
        self.process = nil
        standardOutput = nil
        standardError = nil
      }
      return self.process
    }
    guard let process else { return }
    if process.isRunning {
      if escalateToKill {
        kill(process.processIdentifier, SIGKILL)
      } else {
        process.terminate()
      }
    }
    guard wait else { return }
    if process.isRunning, !escalateToKill {
      let deadline = Date().addingTimeInterval(1)
      while process.isRunning, Date() < deadline {
        Thread.sleep(forTimeInterval: 0.02)
      }
      if process.isRunning {
        kill(process.processIdentifier, SIGKILL)
      }
    }
    process.waitUntilExit()
  }
}
