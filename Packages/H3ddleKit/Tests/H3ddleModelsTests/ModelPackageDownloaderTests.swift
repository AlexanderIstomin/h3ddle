import CryptoKit
import Foundation
import Testing

@testable import H3ddleModels

@Suite("Managed model packages")
struct ModelPackageDownloaderTests {
  @Test("The curated INT8 package is immutable and selective")
  func curatedManifest() throws {
    let manifest = ModelCatalog.minimaxH3Int8

    #expect(manifest.repository == "Comfy-Org/MiniMax-H3")
    #expect(manifest.revision == "014cd40f7e177756c6b2473c0d93b1c89a790dd2")
    #expect(manifest.files.count == 9)
    #expect(manifest.totalByteCount == 53_931_823_333)
    #expect(manifest.files.filter { $0.role != .runtimeMetadata }.allSatisfy {
      $0.path.hasSuffix(".safetensors")
    })
    #expect(manifest.files.filter { $0.role == .runtimeMetadata }.allSatisfy {
      $0.sourceRepository == "MiniMaxAI/MiniMax-H3"
        && $0.sourceRevision == "939557dc319dd91227e30195a763f272ba7f8765"
    })
    let tokenizer = try #require(
      manifest.files.first { $0.path == "FL2VA/tokenizer/tokenizer.json" }
    )
    #expect(
      manifest.downloadURL(for: tokenizer).absoluteString
        == "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/939557dc319dd91227e30195a763f272ba7f8765/FL2VA/tokenizer/tokenizer.json"
    )
    #expect(manifest.compatibility == .ready)
    #expect(manifest.licenseURL.absoluteString.contains("939557dc319dd91227e30195a763f272ba7f8765"))
  }

  @Test("A completed package is verified and installed atomically")
  func installsPackage() async throws {
    let fixture = Data("small model fixture".utf8)
    let manifest = makeManifest(payload: fixture)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let transport = FixtureTransport(payload: fixture)
    let downloader = ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root),
      transport: transport,
      capacityChecker: FixedCapacityChecker(bytes: .max)
    )

    let installedURL = try await downloader.download(manifest)

    #expect(installedURL == root.appendingPathComponent(manifest.id))
    #expect(
      FileManager.default.fileExists(
        atPath: installedURL.appendingPathComponent("weights/model.safetensors").path
      )
    )
    #expect(
      FileManager.default.fileExists(
        atPath: installedURL.appendingPathComponent(ModelPackageStore.installedManifestName).path
      )
    )
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent(".staging").appendingPathComponent(manifest.id).path))
    #expect(await downloader.installedPackageURL(for: manifest) == installedURL)
  }

  @Test("A partial file resumes with a byte range")
  func resumesPartialFile() async throws {
    let fixture = Data("0123456789abcdef".utf8)
    let manifest = makeManifest(payload: fixture)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelPackageStore(rootURL: root)
    let partialURL = store.stagingURL(for: manifest)
      .appendingPathComponent("weights/model.safetensors.partial")
    try FileManager.default.createDirectory(
      at: partialURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    try fixture.prefix(7).write(to: partialURL)
    let transport = FixtureTransport(payload: fixture)
    let downloader = ModelPackageDownloader(
      store: store,
      transport: transport,
      capacityChecker: FixedCapacityChecker(bytes: .max)
    )

    _ = try await downloader.download(manifest)

    let ranges = await transport.requestedRanges
    #expect(ranges == ["bytes=7-"])
  }

  @Test("A corrupt response is rejected and cannot be resumed as valid")
  func rejectsCorruption() async throws {
    let expected = Data("expected".utf8)
    let received = Data("bad data".utf8)
    let manifest = makeManifest(payload: expected)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelPackageStore(rootURL: root)
    let downloader = ModelPackageDownloader(
      store: store,
      transport: FixtureTransport(payload: received),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    )

    do {
      _ = try await downloader.download(manifest)
      Issue.record("Expected checksum verification to fail")
    } catch let error as ModelDownloadError {
      #expect(error == .checksumMismatch(file: "model.safetensors"))
    }

    let partialURL = store.stagingURL(for: manifest)
      .appendingPathComponent("weights/model.safetensors.partial")
    #expect(!FileManager.default.fileExists(atPath: partialURL.path))
  }

  @Test("Cancellation preserves partial data for a later resume")
  func cancellationPreservesPartialData() async throws {
    let fixture = Data("a useful partial model download".utf8)
    let manifest = makeManifest(payload: fixture)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelPackageStore(rootURL: root)
    let transport = SuspendingTransport(payload: fixture)
    let downloader = ModelPackageDownloader(
      store: store,
      transport: transport,
      capacityChecker: FixedCapacityChecker(bytes: .max)
    )
    let task = Task {
      try await downloader.download(manifest)
    }

    let writtenBytes = await transport.waitUntilStarted()
    task.cancel()
    do {
      _ = try await task.value
      Issue.record("Expected the download task to be cancelled")
    } catch is CancellationError {
      // Expected.
    }

    let partialURL = store.stagingURL(for: manifest)
      .appendingPathComponent("weights/model.safetensors.partial")
    let attributes = try FileManager.default.attributesOfItem(atPath: partialURL.path)
    #expect((attributes[.size] as? NSNumber)?.int64Value == writtenBytes)

    let resumeTransport = FixtureTransport(payload: fixture)
    let resumeDownloader = ModelPackageDownloader(
      store: store,
      transport: resumeTransport,
      capacityChecker: FixedCapacityChecker(bytes: .max)
    )
    _ = try await resumeDownloader.download(manifest)
    #expect(await resumeTransport.requestedRanges == ["bytes=\(writtenBytes)-"])
  }

  @Test("Disk preflight fails before issuing a request")
  func checksDiskCapacity() async throws {
    let fixture = Data("fixture".utf8)
    let manifest = makeManifest(payload: fixture)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let transport = FixtureTransport(payload: fixture)
    let downloader = ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root),
      transport: transport,
      capacityChecker: FixedCapacityChecker(bytes: 1)
    )

    do {
      _ = try await downloader.download(manifest)
      Issue.record("Expected the disk preflight to fail")
    } catch let error as ModelDownloadError {
      guard case .insufficientDiskSpace(_, let available) = error else {
        Issue.record("Unexpected error: \(error)")
        return
      }
      #expect(available == 1)
    }
    #expect(await transport.requestedRanges.isEmpty)
  }

  @Test("The URL session transport streams a byte-range response to disk")
  func urlSessionTransportStreamsRange() async throws {
    let fixture = Data("streamed fixture".utf8)
    FixtureURLProtocol.install(payload: fixture)
    defer { FixtureURLProtocol.reset() }
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [FixtureURLProtocol.self]
    let transport = URLSessionModelFileTransport(configuration: configuration)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let destination = root.appendingPathComponent("fixture.partial")
    try fixture.prefix(5).write(to: destination)
    var request = URLRequest(url: URL(string: "https://fixture.invalid/model")!)
    request.setValue("bytes=5-", forHTTPHeaderField: "Range")

    let response = try await transport.download(
      request: request,
      to: destination,
      existingBytes: 5,
      progress: { _ in }
    )

    #expect(response == ModelHTTPResponse(statusCode: 206, bytesWritten: Int64(fixture.count)))
    #expect(try Data(contentsOf: destination) == fixture)
  }

  private func makeManifest(payload: Data) -> ModelPackageManifest {
    ModelPackageManifest(
      id: "fixture-model",
      displayName: "Fixture",
      detail: "Tiny test package",
      repository: "example/fixture",
      revision: String(repeating: "a", count: 40),
      licenseName: "Test",
      licenseURL: URL(string: "https://example.com/license")!,
      minimumUnifiedMemoryBytes: 1,
      compatibility: .ready,
      files: [
        ModelPackageFile(
          role: .transformer,
          path: "weights/model.safetensors",
          byteCount: Int64(payload.count),
          sha256: SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        )
      ]
    )
  }

  private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleModelsTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }
}

private struct FixedCapacityChecker: ModelStorageCapacityChecking {
  let bytes: Int64

  func availableCapacity(at url: URL) throws -> Int64 {
    bytes
  }
}

private actor FixtureTransport: ModelFileTransport {
  let payload: Data
  private(set) var requestedRanges: [String?] = []

  init(payload: Data) {
    self.payload = payload
  }

  func download(
    request: URLRequest,
    to destination: URL,
    existingBytes: Int64,
    progress: @escaping @Sendable (Int64) -> Void
  ) async throws -> ModelHTTPResponse {
    requestedRanges.append(request.value(forHTTPHeaderField: "Range"))
    let statusCode = existingBytes > 0 ? 206 : 200
    let bytes = payload.dropFirst(Int(existingBytes))
    if existingBytes > 0 {
      let handle = try FileHandle(forWritingTo: destination)
      defer { try? handle.close() }
      try handle.seekToEnd()
      try handle.write(contentsOf: bytes)
    } else {
      try Data(bytes).write(to: destination)
    }
    progress(Int64(payload.count))
    return ModelHTTPResponse(statusCode: statusCode, bytesWritten: Int64(payload.count))
  }
}

private actor SuspendingTransport: ModelFileTransport {
  let payload: Data
  private var writtenBytes: Int64?
  private var waiters: [CheckedContinuation<Int64, Never>] = []

  init(payload: Data) {
    self.payload = payload
  }

  func waitUntilStarted() async -> Int64 {
    if let writtenBytes { return writtenBytes }
    return await withCheckedContinuation { continuation in
      waiters.append(continuation)
    }
  }

  func download(
    request: URLRequest,
    to destination: URL,
    existingBytes: Int64,
    progress: @escaping @Sendable (Int64) -> Void
  ) async throws -> ModelHTTPResponse {
    let prefix = payload.prefix(max(1, payload.count / 2))
    try Data(prefix).write(to: destination)
    let count = Int64(prefix.count)
    writtenBytes = count
    let pendingWaiters = waiters
    waiters.removeAll()
    for waiter in pendingWaiters {
      waiter.resume(returning: count)
    }
    progress(count)
    try await Task.sleep(for: .seconds(3_600))
    throw CancellationError()
  }
}

private final class FixtureURLProtocol: URLProtocol, @unchecked Sendable {
  private static let lock = NSLock()
  nonisolated(unsafe) private static var payload = Data()

  static func install(payload: Data) {
    lock.withLock {
      self.payload = payload
    }
  }

  static func reset() {
    lock.withLock {
      payload = Data()
    }
  }

  override class func canInit(with request: URLRequest) -> Bool {
    true
  }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest {
    request
  }

  override func startLoading() {
    let payload = Self.lock.withLock { Self.payload }
    let range = request.value(forHTTPHeaderField: "Range")
    let offset = range.flatMap {
      Int($0.dropFirst("bytes=".count).dropLast())
    } ?? 0
    let statusCode = offset > 0 ? 206 : 200
    let response = HTTPURLResponse(
      url: request.url!,
      statusCode: statusCode,
      httpVersion: "HTTP/1.1",
      headerFields: [
        "Content-Length": String(payload.count - offset),
        "Accept-Ranges": "bytes",
      ]
    )!
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: Data(payload.dropFirst(offset)))
    client?.urlProtocolDidFinishLoading(self)
  }

  override func stopLoading() {}
}
