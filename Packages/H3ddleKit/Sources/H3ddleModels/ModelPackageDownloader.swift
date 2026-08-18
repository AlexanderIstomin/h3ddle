import CryptoKit
import Foundation

public enum ModelDownloadPhase: Equatable, Sendable {
  case preparing
  case downloading
  case verifying
  case installing
  case completed
  case cancelled
}

public struct ModelDownloadProgress: Equatable, Sendable {
  public let phase: ModelDownloadPhase
  public let currentFileName: String?
  public let completedBytes: Int64
  public let totalBytes: Int64

  public init(
    phase: ModelDownloadPhase,
    currentFileName: String? = nil,
    completedBytes: Int64,
    totalBytes: Int64
  ) {
    self.phase = phase
    self.currentFileName = currentFileName
    self.completedBytes = completedBytes
    self.totalBytes = totalBytes
  }

  public var fractionCompleted: Double {
    guard totalBytes > 0 else { return 0 }
    return min(1, max(0, Double(completedBytes) / Double(totalBytes)))
  }
}

public enum ModelDownloadError: LocalizedError, Equatable, Sendable {
  case alreadyDownloading
  case invalidResponse(Int)
  case insufficientDiskSpace(required: Int64, available: Int64)
  case sizeMismatch(file: String, expected: Int64, actual: Int64)
  case checksumMismatch(file: String)
  case installCollision(URL)
  case localSourceUnavailable(file: String)

  public var errorDescription: String? {
    switch self {
    case .alreadyDownloading:
      "Another model download is already active."
    case .invalidResponse(let statusCode):
      "The model host returned HTTP status \(statusCode)."
    case .insufficientDiskSpace(let required, let available):
      "The model needs \(Self.bytes(required)) free, but only \(Self.bytes(available)) is available."
    case .sizeMismatch(let file, let expected, let actual):
      "\(file) has the wrong size (expected \(Self.bytes(expected)), found \(Self.bytes(actual)))."
    case .checksumMismatch(let file):
      "\(file) did not match its published SHA-256 checksum."
    case .installCollision(let url):
      "A different model package already exists at \(url.path(percentEncoded: false))."
    case .localSourceUnavailable(let file):
      "\(file) has no download source and no verified local copy was found."
    }
  }

  private static func bytes(_ value: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
  }
}

public struct ModelHTTPResponse: Equatable, Sendable {
  public let statusCode: Int
  public let bytesWritten: Int64

  public init(statusCode: Int, bytesWritten: Int64) {
    self.statusCode = statusCode
    self.bytesWritten = bytesWritten
  }
}

public protocol ModelFileTransport: Sendable {
  func download(
    request: URLRequest,
    to destination: URL,
    existingBytes: Int64,
    progress: @escaping @Sendable (Int64) -> Void
  ) async throws -> ModelHTTPResponse
}

public protocol ModelStorageCapacityChecking: Sendable {
  func availableCapacity(at url: URL) throws -> Int64
}

public struct VolumeStorageCapacityChecker: ModelStorageCapacityChecking {
  public init() {}

  public func availableCapacity(at url: URL) throws -> Int64 {
    let values = try url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
    return values.volumeAvailableCapacityForImportantUsage ?? 0
  }
}

public struct ModelPackageStore: Equatable, Sendable {
  public static let installedManifestName = ".h3ddle-model.json"

  public let rootURL: URL

  public init(rootURL: URL) {
    self.rootURL = rootURL
  }

  public static var applicationSupport: ModelPackageStore {
    ModelPackageStore(
      rootURL: URL.applicationSupportDirectory
        .appendingPathComponent("H3ddle", isDirectory: true)
        .appendingPathComponent("Models", isDirectory: true)
    )
  }

  public func installedURL(for manifest: ModelPackageManifest) -> URL {
    rootURL.appendingPathComponent(manifest.id, isDirectory: true)
  }

  public func stagingURL(for manifest: ModelPackageManifest) -> URL {
    rootURL
      .appendingPathComponent(".staging", isDirectory: true)
      .appendingPathComponent(manifest.id, isDirectory: true)
  }
}

public actor ModelPackageDownloader {
  public typealias ProgressHandler = @Sendable (ModelDownloadProgress) -> Void

  private static let diskSafetyMargin: Int64 = 2 * 1_024 * 1_024 * 1_024

  private let store: ModelPackageStore
  private let transport: any ModelFileTransport
  private let capacityChecker: any ModelStorageCapacityChecking
  private var isDownloading = false

  public init(
    store: ModelPackageStore = .applicationSupport,
    transport: any ModelFileTransport = URLSessionModelFileTransport(),
    capacityChecker: any ModelStorageCapacityChecking = VolumeStorageCapacityChecker()
  ) {
    self.store = store
    self.transport = transport
    self.capacityChecker = capacityChecker
  }

  public func installedPackageURL(for manifest: ModelPackageManifest) -> URL? {
    let installedURL = store.installedURL(for: manifest)
    let manifestURL = installedURL.appendingPathComponent(
      ModelPackageStore.installedManifestName,
      isDirectory: false
    )
    guard
      let data = try? Data(contentsOf: manifestURL),
      let installedManifest = try? JSONDecoder().decode(ModelPackageManifest.self, from: data),
      installedManifest.describesSameFiles(as: manifest)
    else {
      return nil
    }
    return installedURL
  }

  public func download(
    _ manifest: ModelPackageManifest,
    progress: @escaping ProgressHandler = { _ in }
  ) async throws -> URL {
    guard !isDownloading else { throw ModelDownloadError.alreadyDownloading }
    isDownloading = true
    defer { isDownloading = false }

    if let installedURL = installedPackageURL(for: manifest) {
      progress(
        ModelDownloadProgress(
          phase: .completed,
          completedBytes: manifest.totalByteCount,
          totalBytes: manifest.totalByteCount
        )
      )
      return installedURL
    }

    do {
      return try await performDownload(manifest, progress: progress)
    } catch is CancellationError {
      progress(
        ModelDownloadProgress(
          phase: .cancelled,
          completedBytes: try stagedByteCount(for: manifest),
          totalBytes: manifest.totalByteCount
        )
      )
      throw CancellationError()
    }
  }

  private func performDownload(
    _ manifest: ModelPackageManifest,
    progress: @escaping ProgressHandler
  ) async throws -> URL {
    let fileManager = FileManager.default
    let stagingURL = store.stagingURL(for: manifest)
    try fileManager.createDirectory(at: stagingURL, withIntermediateDirectories: true)

    let alreadyPresent = try stagedByteCount(for: manifest)
    progress(
      ModelDownloadProgress(
        phase: .preparing,
        completedBytes: alreadyPresent,
        totalBytes: manifest.totalByteCount
      )
    )
    try preflightDiskSpace(manifest: manifest, alreadyPresent: alreadyPresent)

    var completedBeforeFile: Int64 = 0
    for file in manifest.files {
      try Task.checkCancellation()
      let destination = stagingURL.appendingPathComponent(file.path, isDirectory: false)
      try fileManager.createDirectory(
        at: destination.deletingLastPathComponent(),
        withIntermediateDirectories: true
      )

      if try fileSize(at: destination) == file.byteCount {
        progress(
          ModelDownloadProgress(
            phase: .verifying,
            currentFileName: file.displayName,
            completedBytes: completedBeforeFile + file.byteCount,
            totalBytes: manifest.totalByteCount
          )
        )
        if try await fileIsValid(file, at: destination) {
          completedBeforeFile += file.byteCount
          continue
        }
      }

      if try await acquireLocalCopy(
        of: file,
        to: destination,
        manifest: manifest,
        completedBeforeFile: completedBeforeFile,
        progress: progress
      ) {
        completedBeforeFile += file.byteCount
        continue
      }
      if file.requiresLocalSource {
        throw ModelDownloadError.localSourceUnavailable(file: file.displayName)
      }

      let partialURL = destination.appendingPathExtension("partial")
      var partialSize = try fileSize(at: partialURL)
      if partialSize > file.byteCount {
        try fileManager.removeItem(at: partialURL)
        partialSize = 0
      }

      var request = URLRequest(url: manifest.downloadURL(for: file))
      request.timeoutInterval = 7 * 24 * 60 * 60
      if partialSize > 0 {
        request.setValue("bytes=\(partialSize)-", forHTTPHeaderField: "Range")
      }

      let reporter = ThrottledProgressReporter(
        completedBeforeFile: completedBeforeFile,
        totalBytes: manifest.totalByteCount,
        fileName: file.displayName,
        handler: progress
      )
      let response = try await transport.download(
        request: request,
        to: partialURL,
        existingBytes: partialSize
      ) { fileBytes in
        reporter.report(fileBytes: fileBytes)
      }
      guard (200...299).contains(response.statusCode) else {
        throw ModelDownloadError.invalidResponse(response.statusCode)
      }

      progress(
        ModelDownloadProgress(
          phase: .verifying,
          currentFileName: file.displayName,
          completedBytes: completedBeforeFile + min(response.bytesWritten, file.byteCount),
          totalBytes: manifest.totalByteCount
        )
      )
      try await verify(file, at: partialURL)
      if fileManager.fileExists(atPath: destination.path) {
        try fileManager.removeItem(at: destination)
      }
      try fileManager.moveItem(at: partialURL, to: destination)
      completedBeforeFile += file.byteCount
    }

    progress(
      ModelDownloadProgress(
        phase: .installing,
        completedBytes: manifest.totalByteCount,
        totalBytes: manifest.totalByteCount
      )
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let manifestData = try encoder.encode(manifest)
    try manifestData.write(
      to: stagingURL.appendingPathComponent(ModelPackageStore.installedManifestName),
      options: .atomic
    )

    let installedURL = store.installedURL(for: manifest)
    if fileManager.fileExists(atPath: installedURL.path) {
      // Swap rather than refuse: the staged tree is complete and verified,
      // and its unchanged files are hardlinks to the ones being replaced.
      let previousURL = installedURL.appendingPathExtension("previous")
      if fileManager.fileExists(atPath: previousURL.path) {
        try fileManager.removeItem(at: previousURL)
      }
      try fileManager.moveItem(at: installedURL, to: previousURL)
      do {
        try fileManager.moveItem(at: stagingURL, to: installedURL)
      } catch {
        try? fileManager.moveItem(at: previousURL, to: installedURL)
        throw error
      }
      try? fileManager.removeItem(at: previousURL)
    } else {
      try fileManager.moveItem(at: stagingURL, to: installedURL)
    }
    progress(
      ModelDownloadProgress(
        phase: .completed,
        completedBytes: manifest.totalByteCount,
        totalBytes: manifest.totalByteCount
      )
    )
    return installedURL
  }

  private func preflightDiskSpace(
    manifest: ModelPackageManifest,
    alreadyPresent: Int64
  ) throws {
    let available = try capacityChecker.availableCapacity(at: store.rootURL)
    let remaining = max(0, pendingByteCount(for: manifest) - alreadyPresent)
    let required = remaining + Self.diskSafetyMargin
    guard available >= required else {
      throw ModelDownloadError.insufficientDiskSpace(required: required, available: available)
    }
  }

  /// Bytes this install actually has to fetch. A file already present in
  /// another installed package, or at the manifest's declared local candidate
  /// path, hardlinks into place instead of downloading — so a package that
  /// only adds one weight file to an existing install costs that file alone,
  /// not the sum of its parts.
  public func pendingByteCount(for manifest: ModelPackageManifest) -> Int64 {
    manifest.files.reduce(into: 0) { total, file in
      if !isLocallySatisfiable(file, in: manifest) { total += file.byteCount }
    }
  }

  private func isLocallySatisfiable(
    _ file: ModelPackageFile,
    in manifest: ModelPackageManifest
  ) -> Bool {
    let hasLocalCandidate =
      file.localCandidatePath.map {
        (try? fileSize(at: URL(fileURLWithPath: $0))) == file.byteCount
      } ?? false
    return hasLocalCandidate
      || !installedCandidates(sha256: file.sha256, excludingPackage: manifest.id).isEmpty
      // Counted here as well as used above, so the size the app quotes before
      // the download is the size the download actually is. Quoting 38.69 GB and
      // then fetching none of it is its own kind of wrong.
      || !huggingFaceCacheCandidates(sha256: file.sha256).isEmpty
  }

  /// Deletes an installed package's files. Weights shared with another
  /// install are hardlinks, so removing this copy leaves the other intact
  /// and reclaims only what nothing else references.
  public func removeInstalledPackage(
    for manifest: ModelPackageManifest
  ) throws {
    let fileManager = FileManager.default
    let installed = store.installedURL(for: manifest)
    if fileManager.fileExists(atPath: installed.path) {
      try fileManager.removeItem(at: installed)
    }
    try? discardStagedDownload(for: manifest)
  }

  /// Throws away a paused download's partial files. Pausing deliberately
  /// keeps them so a resume costs nothing; discarding is the separate,
  /// explicit act of reclaiming that disk.
  public func discardStagedDownload(for manifest: ModelPackageManifest) throws {
    let fileManager = FileManager.default
    let stagingURL = store.stagingURL(for: manifest)
    guard fileManager.fileExists(atPath: stagingURL.path) else { return }
    try fileManager.removeItem(at: stagingURL)
  }

  public func stagedByteCount(for manifest: ModelPackageManifest) throws -> Int64 {
    let stagingURL = store.stagingURL(for: manifest)
    var count: Int64 = 0
    for file in manifest.files {
      let destination = stagingURL.appendingPathComponent(file.path)
      let destinationSize = try fileSize(at: destination)
      if destinationSize == file.byteCount {
        count += destinationSize
      } else {
        let partialSize = try fileSize(at: destination.appendingPathExtension("partial"))
        count += min(partialSize, file.byteCount)
      }
    }
    return count
  }

  /// Tries every machine-local source for a file — the manifest's declared
  /// candidate, then identical files inside other installed packages — and
  /// installs the first one that verifies. Hardlinks when the volume allows
  /// it so shared files cost no additional disk.
  private func acquireLocalCopy(
    of file: ModelPackageFile,
    to destination: URL,
    manifest: ModelPackageManifest,
    completedBeforeFile: Int64,
    progress: @escaping ProgressHandler
  ) async throws -> Bool {
    let fileManager = FileManager.default
    var candidates: [URL] = []
    if let localPath = file.localCandidatePath {
      candidates.append(URL(fileURLWithPath: localPath))
    }
    // An update reuses the bytes it already has: unchanged files hardlink
    // out of the existing install instead of downloading again.
    let installed = store.installedURL(for: manifest)
      .appendingPathComponent(file.path, isDirectory: false)
    if FileManager.default.fileExists(atPath: installed.path) {
      candidates.append(installed)
    }
    candidates.append(
      contentsOf: installedCandidates(sha256: file.sha256, excludingPackage: manifest.id)
    )
    candidates.append(contentsOf: huggingFaceCacheCandidates(sha256: file.sha256))

    for candidate in candidates {
      try Task.checkCancellation()
      guard try fileSize(at: candidate) == file.byteCount else { continue }
      let scratch = destination.appendingPathExtension("local")
      if fileManager.fileExists(atPath: scratch.path) {
        try fileManager.removeItem(at: scratch)
      }
      do {
        try fileManager.linkItem(at: candidate, to: scratch)
      } catch {
        do {
          try fileManager.copyItem(at: candidate, to: scratch)
        } catch {
          continue
        }
      }
      progress(
        ModelDownloadProgress(
          phase: .verifying,
          currentFileName: file.displayName,
          completedBytes: completedBeforeFile + file.byteCount,
          totalBytes: manifest.totalByteCount
        )
      )
      do {
        try await verify(file, at: scratch)
      } catch {
        try? fileManager.removeItem(at: scratch)
        continue
      }
      if fileManager.fileExists(atPath: destination.path) {
        try fileManager.removeItem(at: destination)
      }
      try fileManager.moveItem(at: scratch, to: destination)
      return true
    }
    return false
  }

  /// Files with a matching checksum inside other installed packages.
  /// Files the Hugging Face cache already holds, found by content hash.
  ///
  /// The cache stores every LFS file under `blobs/<oid>`, and that oid *is*
  /// the SHA-256 this manifest declares — so this is a direct lookup costing
  /// one `stat` per cached repository, not a scan of forty gigabytes.
  ///
  /// This is worth more than it looks. The weights these packages install are
  /// very often already on the machine, pulled by `hf`, `transformers`, ComfyUI
  /// or a previous experiment, and re-fetching 38 GB that is sitting in
  /// `~/.cache/huggingface` is the kind of thing people rightly complain about.
  /// A hardlink means the reused copy costs no disk either, and survives the
  /// cache being pruned later.
  ///
  /// Nothing is trusted on the strength of a filename: every candidate is
  /// verified against the declared hash before it is installed, exactly as a
  /// downloaded file is.
  private func huggingFaceCacheCandidates(sha256: String) -> [URL] {
    let fileManager = FileManager.default
    let environment = ProcessInfo.processInfo.environment
    var hubs: [URL] = []
    if let explicit = environment["HF_HUB_CACHE"] {
      hubs.append(URL(fileURLWithPath: explicit))
    }
    if let home = environment["HF_HOME"] {
      hubs.append(URL(fileURLWithPath: home).appendingPathComponent("hub"))
    }
    hubs.append(
      fileManager.homeDirectoryForCurrentUser
        .appendingPathComponent(".cache/huggingface/hub")
    )
    var candidates: [URL] = []
    var seen = Set<String>()
    for hub in hubs {
      guard
        let repositories = try? fileManager.contentsOfDirectory(
          at: hub,
          includingPropertiesForKeys: nil,
          options: [.skipsHiddenFiles]
        )
      else { continue }
      for repository in repositories {
        let blob = repository
          .appendingPathComponent("blobs", isDirectory: true)
          .appendingPathComponent(sha256, isDirectory: false)
        guard fileManager.fileExists(atPath: blob.path),
          seen.insert(blob.path).inserted
        else { continue }
        candidates.append(blob)
      }
    }
    return candidates
  }

  private func installedCandidates(sha256: String, excludingPackage: String) -> [URL] {
    let fileManager = FileManager.default
    guard
      let entries = try? fileManager.contentsOfDirectory(
        at: store.rootURL,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
      )
    else {
      return []
    }
    var candidates: [URL] = []
    for entry in entries where entry.lastPathComponent != excludingPackage {
      let manifestURL = entry.appendingPathComponent(
        ModelPackageStore.installedManifestName,
        isDirectory: false
      )
      guard
        let data = try? Data(contentsOf: manifestURL),
        let installed = try? JSONDecoder().decode(ModelPackageManifest.self, from: data)
      else {
        continue
      }
      for installedFile in installed.files where installedFile.sha256 == sha256 {
        candidates.append(entry.appendingPathComponent(installedFile.path, isDirectory: false))
      }
    }
    return candidates
  }

  private func fileIsValid(_ file: ModelPackageFile, at url: URL) async throws -> Bool {
    guard try fileSize(at: url) == file.byteCount else { return false }
    do {
      try await verify(file, at: url)
      return true
    } catch ModelDownloadError.checksumMismatch {
      return false
    }
  }

  private func verify(_ file: ModelPackageFile, at url: URL) async throws {
    let size = try fileSize(at: url)
    guard size == file.byteCount else {
      throw ModelDownloadError.sizeMismatch(
        file: file.displayName,
        expected: file.byteCount,
        actual: size
      )
    }
    let verificationTask = Task.detached(priority: .utility) {
      try await Self.sha256(of: url)
    }
    let digest = try await withTaskCancellationHandler {
      try await verificationTask.value
    } onCancel: {
      verificationTask.cancel()
    }
    guard digest == file.sha256.lowercased() else {
      try? FileManager.default.removeItem(at: url)
      throw ModelDownloadError.checksumMismatch(file: file.displayName)
    }
  }

  private func fileSize(at url: URL) throws -> Int64 {
    guard FileManager.default.fileExists(atPath: url.path) else { return 0 }
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    return (attributes[.size] as? NSNumber)?.int64Value ?? 0
  }

  private nonisolated static func sha256(of url: URL) async throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let data = try handle.read(upToCount: 8 * 1_024 * 1_024), !data.isEmpty {
      try Task.checkCancellation()
      hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

private final class ThrottledProgressReporter: @unchecked Sendable {
  private let lock = NSLock()
  private let completedBeforeFile: Int64
  private let totalBytes: Int64
  private let fileName: String
  private let handler: ModelPackageDownloader.ProgressHandler
  private var lastReport = ContinuousClock.now - .seconds(1)

  init(
    completedBeforeFile: Int64,
    totalBytes: Int64,
    fileName: String,
    handler: @escaping ModelPackageDownloader.ProgressHandler
  ) {
    self.completedBeforeFile = completedBeforeFile
    self.totalBytes = totalBytes
    self.fileName = fileName
    self.handler = handler
  }

  func report(fileBytes: Int64) {
    lock.lock()
    let now = ContinuousClock.now
    guard now - lastReport >= .milliseconds(150) else {
      lock.unlock()
      return
    }
    lastReport = now
    lock.unlock()
    handler(
      ModelDownloadProgress(
        phase: .downloading,
        currentFileName: fileName,
        completedBytes: completedBeforeFile + fileBytes,
        totalBytes: totalBytes
      )
    )
  }
}
