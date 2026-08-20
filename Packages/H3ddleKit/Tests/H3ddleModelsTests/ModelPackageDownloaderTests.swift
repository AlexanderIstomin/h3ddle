import CryptoKit
import Foundation
import Testing

@testable import H3ddleModels

/// Serialized because two of these set `HF_HUB_CACHE`, which is process-global,
/// and Swift Testing runs tests in parallel by default. One test's `unsetenv`
/// clearing another's cache mid-run is exactly how this suite first failed —
/// intermittently, and only once both cache tests existed.
@Suite("Managed model packages", .serialized)
struct ModelPackageDownloaderTests {
  @Test("The curated INT8 package is immutable and selective")
  func curatedManifest() throws {
    let manifest = ModelCatalog.minimaxH3Int8

    #expect(manifest.repository == "Comfy-Org/MiniMax-H3")
    #expect(manifest.revision == "014cd40f7e177756c6b2473c0d93b1c89a790dd2")
    #expect(manifest.files.count == 10)
    #expect(manifest.totalByteCount == 53_941_614_829)
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

  @Test("Video and image package descriptions name user-facing capabilities")
  func packageCapabilityDescriptions() {
    let ltx = ModelCatalog.ltx25.detail
    #expect(ltx.contains("2–20 second videos with synchronized sound"))
    #expect(ltx.contains("start/end frames"))
    #expect(ltx.contains("up to four reference images"))
    #expect(ltx.contains("320p to 1080p"))
    #expect(ltx.contains("eight steps"))
    #expect(!ltx.contains("no reference images"))

    let zImage = ModelCatalog.zImageTurbo.detail
    #expect(zImage.contains("high-quality still images"))
    #expect(zImage.contains("repaints a source picture"))
    #expect(zImage.contains("portrait, square, and landscape"))
    #expect(zImage.contains("512p to 1536p"))
    #expect(zImage.contains("eight passes"))

    let hybrid = ModelCatalog.minimaxH3Ref2VATurboInt8
    #expect(hybrid.displayName.contains("Hybrid References"))
    #expect(hybrid.detail.contains("compact hybrid references"))
    let standardHybrid = ModelCatalog.minimaxH3Ref2VAInt8
    #expect(standardHybrid.displayName.contains("Hybrid References"))
    #expect(standardHybrid.detail.contains("compact hybrid references"))
  }

  @Test("Z-Image carries an immutable TAEF1 live-preview decoder")
  func zImagePreviewDecoder() throws {
    let manifest = ModelCatalog.zImageTurbo
    let preview = try #require(
      manifest.files.first { $0.role == .previewDecoder }
    )

    #expect(preview.path == "vae_approx/taef1.safetensors")
    #expect(preview.byteCount == 9_848_636)
    #expect(
      preview.sha256
        == "47a6c2bff850da04b267cab70fe3553fef57255eb9a8e76852baa0a87850e54d"
    )
    #expect(preview.sourceRepository == "madebyollin/taef1")
    #expect(preview.sourceRevision == "b1b2d00e9e440cfbf3dedb34266864da86016ceb")
    #expect(preview.sourcePath == "diffusion_pytorch_model.safetensors")
    #expect(
      manifest.downloadURL(for: preview).absoluteString
        == "https://huggingface.co/madebyollin/taef1/resolve/"
          + "b1b2d00e9e440cfbf3dedb34266864da86016ceb/"
          + "diffusion_pytorch_model.safetensors"
    )
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

  @Test("The turbo package ships its exact-output full input-major transformer")
  func turboManifestSharing() throws {
    let standard = ModelCatalog.minimaxH3Int8
    let turbo = ModelCatalog.minimaxH3TurboInt8

    #expect(turbo.generationProfile == .turbo)
    #expect(turbo.generationProfile.defaultDenoisingSteps == 8)
    #expect(turbo.generationProfile.usesBetaSchedule)
    #expect(standard.generationProfile == .standard)

    let standardTransformer = try #require(standard.files.first { $0.role == .transformer })
    let turboTransformer = try #require(turbo.files.first { $0.role == .transformer })
    // h3.c resolves the transformer by this exact in-package path, while the
    // hosted file lives at the root of its own repository.
    #expect(turboTransformer.path == standardTransformer.path)
    #expect(turboTransformer.sha256 != standardTransformer.sha256)
    #expect(standardTransformer.byteCount == 20_970_379_724)
    #expect(
      standardTransformer.sha256
        == "5a0a3e1e73f099680896a98ab418870f27711800ac73b5f16af81724fc7e567a"
    )
    #expect(!standardTransformer.requiresLocalSource)
    #expect(standardTransformer.sourceRepository == "PulpCut/MiniMax-H3-INT8-ConvRot")
    #expect(
      standardTransformer.sourceRevision
        == "0324458ae2cfd885a2252ee752e7d60d2466c345"
    )
    #expect(
      standardTransformer.sourcePath
        == "minimax_h3_fl2va_pruned_int8_convrot_input_major.safetensors"
    )
    #expect(
      standardTransformer.localCandidatePath?.hasSuffix(
        "minimax_h3_fl2va_pruned_int8_convrot_input_major.safetensors"
      ) == true
    )
    #expect(turboTransformer.byteCount == 20_970_380_012)
    #expect(
      turboTransformer.sha256
        == "1dfe28c517a937fb9876f0975f224fd6e7ecb8744219f89bb8ba954403e10dc3"
    )
    #expect(!turboTransformer.requiresLocalSource)
    #expect(turboTransformer.localCandidatePath != nil)
    #expect(
      turbo.downloadURL(for: turboTransformer).absoluteString
        == "https://huggingface.co/PulpCut/MiniMax-H3-Turbo-INT8-ConvRot/resolve/"
          + "7a8e67cc51737428938fd9e39903a66e8ba58a18/"
          + "minimax_h3_fl2va_pruned_turbo_int8_convrot_input_major.safetensors"
    )
    // Every base-package file is identical across the two packages, so an
    // install of one can supply the other.
    let sharedStandard = standard.files.filter { $0.role != .transformer }
    let sharedTurbo = turbo.files.filter { $0.role != .transformer }
    #expect(sharedStandard == sharedTurbo)
    let videoVAE = try #require(turbo.files.first { $0.role == .videoVAE })
    #expect(
      turbo.downloadURL(for: videoVAE).absoluteString
        .hasPrefix("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/")
    )
  }

  @Test("Both reference packages share the compact overlay")
  func referenceManifestsUseHybridOverlay() throws {
    let manifest = ModelCatalog.minimaxH3Ref2VATurboInt8
    let transformers = manifest.files.filter {
      $0.role == .transformer || $0.role == .referenceTransformer
    }
    #expect(transformers.count == 2)
    let prompt = try #require(transformers.first { $0.role == .transformer })
    #expect(prompt.byteCount == 20_970_380_012)
    #expect(prompt.sourcePath?.hasSuffix("_input_major.safetensors") == true)
    #expect(
      prompt.sha256
        == "1dfe28c517a937fb9876f0975f224fd6e7ecb8744219f89bb8ba954403e10dc3"
    )
    let reference = try #require(
      transformers.first { $0.role == .referenceTransformer }
    )
    #expect(reference.byteCount == 43_551_180)
    #expect(
      reference.path
        == "diffusion_models/"
          + "minimax_h3_ref2va_pruned_int8_convrot_hybrid_adaln_25_49.safetensors"
    )
    #expect(
      reference.sha256
        == "c3d80a9a2d17a30caf83e933262473cbf0b1ba7de4d29556646e9a92ab5f17aa"
    )
    #expect(
      manifest.downloadURL(for: reference).absoluteString
        == "https://huggingface.co/PulpCut/"
          + "MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot/resolve/"
          + "2f01b8f268aeb6c3e73eac7fe4ab13c2ad0c1954/"
          + "minimax_h3_ref2va_pruned_int8_convrot_hybrid_adaln_25_49.safetensors"
    )
    let standard = ModelCatalog.minimaxH3Ref2VAInt8
    let standardPrompt = try #require(
      standard.files.first { $0.role == .transformer }
    )
    #expect(standardPrompt.byteCount == 20_970_379_724)
    #expect(
      standardPrompt.sha256
        == "5a0a3e1e73f099680896a98ab418870f27711800ac73b5f16af81724fc7e567a"
    )
    #expect(!standardPrompt.requiresLocalSource)
    #expect(
      standard.downloadURL(for: standardPrompt).absoluteString
        == "https://huggingface.co/PulpCut/MiniMax-H3-INT8-ConvRot/resolve/"
          + "0324458ae2cfd885a2252ee752e7d60d2466c345/"
          + "minimax_h3_fl2va_pruned_int8_convrot_input_major.safetensors"
    )
    let standardReference = try #require(
      standard.files.first { $0.role == .referenceTransformer }
    )
    #expect(standardReference == reference)
    #expect(
      standard.downloadURL(for: standardReference)
        == manifest.downloadURL(for: reference)
    )
  }

  @Test("Manifests written before generation profiles decode with defaults")
  func legacyManifestDecoding() throws {
    let encoder = JSONEncoder()
    var object = try #require(
      try JSONSerialization.jsonObject(
        with: encoder.encode(ModelCatalog.minimaxH3Int8)
      ) as? [String: Any]
    )
    object.removeValue(forKey: "generationProfile")
    var files = try #require(object["files"] as? [[String: Any]])
    for index in files.indices {
      files[index].removeValue(forKey: "requiresLocalSource")
      files[index].removeValue(forKey: "localCandidatePath")
      files[index].removeValue(forKey: "sourcePath")
    }
    object["files"] = files
    let legacy = try JSONSerialization.data(withJSONObject: object)

    let decoded = try JSONDecoder().decode(ModelPackageManifest.self, from: legacy)
    // The absent fields take their defaults; everything else round trips.
    #expect(decoded.generationProfile == .standard)
    #expect(decoded.files.allSatisfy { !$0.requiresLocalSource })
    #expect(decoded.files.allSatisfy { $0.localCandidatePath == nil })
    #expect(decoded.files.allSatisfy { $0.sourcePath == nil })
    #expect(decoded.id == ModelCatalog.minimaxH3Int8.id)
    #expect(decoded.files.map(\.sha256) == ModelCatalog.minimaxH3Int8.files.map(\.sha256))
  }

  /// What the user is told before they agree to anything. An update that adds
  /// one small file to an installed package must quote that file, not the
  /// package — being asked to download something already on disk is
  /// indistinguishable from the app having lost it.
  @Test("An update quotes only the bytes it will actually fetch")
  func updateQuotesOnlyTheNewBytes() async throws {
    let payload = Data(repeating: 7, count: 4096)
    let extra = Data(repeating: 9, count: 32)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let before = makeManifest(payload: payload)
    _ = try await ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root),
      transport: FixtureTransport(payload: payload),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(before)

    var files = before.files
    files.append(
      ModelPackageFile(
        role: .imageVAE,
        path: "weights/extra.safetensors",
        byteCount: Int64(extra.count),
        sha256: SHA256.hash(data: extra).map { String(format: "%02x", $0) }.joined()
      )
    )
    let after = ModelPackageManifest(
      id: before.id, displayName: before.displayName, detail: before.detail,
      repository: before.repository, revision: before.revision,
      licenseName: before.licenseName, licenseURL: before.licenseURL,
      minimumUnifiedMemoryBytes: before.minimumUnifiedMemoryBytes,
      compatibility: before.compatibility, files: files
    )

    let downloader = ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root),
      transport: RefusingTransport(),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    )
    let pending = await downloader.pendingByteCount(for: after)
    #expect(after.totalByteCount == Int64(payload.count + extra.count))
    #expect(pending == Int64(extra.count))
  }

  @Test("Replacing a full reference file with an overlay reclaims the old file")
  func replacesFullReferenceWithOverlay() async throws {
    let fullReference = Data("large full reference fixture".utf8)
    let overlay = Data("compact hybrid overlay".utf8)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let before = makeManifest(payload: fullReference)
    let store = ModelPackageStore(rootURL: root)
    let installed = try await ModelPackageDownloader(
      store: store,
      transport: FixtureTransport(payload: fullReference),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(before)
    let fullURL = installed.appendingPathComponent("weights/model.safetensors")
    #expect(FileManager.default.fileExists(atPath: fullURL.path))

    let overlayFile = ModelPackageFile(
      role: .referenceTransformer,
      path: "weights/hybrid-overlay.safetensors",
      byteCount: Int64(overlay.count),
      sha256: SHA256.hash(data: overlay).map { String(format: "%02x", $0) }.joined()
    )
    let after = ModelPackageManifest(
      id: before.id, displayName: before.displayName, detail: before.detail,
      repository: before.repository, revision: before.revision,
      licenseName: before.licenseName, licenseURL: before.licenseURL,
      minimumUnifiedMemoryBytes: before.minimumUnifiedMemoryBytes,
      compatibility: before.compatibility, files: [overlayFile]
    )
    let updater = ModelPackageDownloader(
      store: store,
      transport: FixtureTransport(payload: overlay),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    )
    #expect(await updater.pendingByteCount(for: after) == Int64(overlay.count))

    let updated = try await updater.download(after)
    #expect(!FileManager.default.fileExists(atPath: fullURL.path))
    #expect(
      try Data(
        contentsOf: updated.appendingPathComponent("weights/hybrid-overlay.safetensors")
      ) == overlay
    )
  }

  /// Adding one accessory file to a manifest must not cost a re-download, and
  /// must not cost a re-hash of everything already on disk either — on the
  /// real Z-Image package that was fourteen gigabytes of digesting to gain
  /// 168 MB, which reads to the user as being asked to download the model
  /// again.
  ///
  /// The corruption below is how the skip is made observable: the installed
  /// bytes are replaced with different bytes of the same length, and they
  /// survive the update untouched. That is the tradeoff stated out loud —
  /// bytes this app verified when it wrote them are taken on trust when they
  /// are hardlinked in place, and only files arriving from anywhere else are
  /// hashed again.
  @Test("Adding a file to a manifest reuses the rest without re-hashing it")
  func addingAFileDoesNotRehashTheRest() async throws {
    let payload = Data("original transformer bytes".utf8)
    let extra = Data("the accessory file".utf8)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let before = makeManifest(payload: payload)
    let installed = try await ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root),
      transport: FixtureTransport(payload: payload),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(before)

    let existing = installed.appendingPathComponent("weights/model.safetensors")
    let tampered = Data("TAMPERED transformer bytes".utf8)
    #expect(tampered.count == payload.count)
    try tampered.write(to: existing)

    var files = before.files
    files.append(
      ModelPackageFile(
        role: .imageVAE,
        path: "weights/extra.safetensors",
        byteCount: Int64(extra.count),
        sha256: SHA256.hash(data: extra).map { String(format: "%02x", $0) }.joined()
      )
    )
    let after = ModelPackageManifest(
      id: before.id,
      displayName: before.displayName,
      detail: before.detail,
      repository: before.repository,
      revision: before.revision,
      licenseName: before.licenseName,
      licenseURL: before.licenseURL,
      minimumUnifiedMemoryBytes: before.minimumUnifiedMemoryBytes,
      compatibility: before.compatibility,
      files: files
    )

    let updated = try await ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root),
      transport: FixtureTransport(payload: extra),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(after)

    /* The new file arrived. */
    #expect(
      try Data(contentsOf: updated.appendingPathComponent("weights/extra.safetensors")) == extra)
    /* The old one was carried across as-is rather than digested again. */
    #expect(
      try Data(contentsOf: updated.appendingPathComponent("weights/model.safetensors")) == tampered)
  }

  @Test("Shared files hardlink from installed packages instead of downloading")
  func reusesInstalledFiles() async throws {
    let sharedPayload = Data("shared encoder bytes".utf8)
    let localPayload = Data("locally converted transformer".utf8)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let first = makeManifest(payload: sharedPayload)
    _ = try await ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root),
      transport: FixtureTransport(payload: sharedPayload),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(first)

    let candidate = root.appendingPathComponent("converted.safetensors")
    try localPayload.write(to: candidate)
    let second = ModelPackageManifest(
      id: "fixture-turbo",
      displayName: "Fixture Turbo",
      detail: "Shares one file, imports one locally",
      repository: "example/fixture",
      revision: String(repeating: "a", count: 40),
      licenseName: "Test",
      licenseURL: URL(string: "https://example.com/license")!,
      minimumUnifiedMemoryBytes: 1,
      compatibility: .ready,
      generationProfile: .turbo,
      files: [
        ModelPackageFile(
          role: .transformer,
          path: "weights/turbo.safetensors",
          byteCount: Int64(localPayload.count),
          sha256: SHA256.hash(data: localPayload).map { String(format: "%02x", $0) }.joined(),
          localCandidatePath: candidate.path,
          requiresLocalSource: true
        ),
        ModelPackageFile(
          role: .textEncoder,
          path: "weights/model.safetensors",
          byteCount: Int64(sharedPayload.count),
          sha256: SHA256.hash(data: sharedPayload).map { String(format: "%02x", $0) }.joined()
        ),
      ]
    )

    let installedURL = try await ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root),
      transport: RefusingTransport(),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(second)

    let sharedData = try Data(
      contentsOf: installedURL.appendingPathComponent("weights/model.safetensors")
    )
    let turboData = try Data(
      contentsOf: installedURL.appendingPathComponent("weights/turbo.safetensors")
    )
    #expect(sharedData == sharedPayload)
    #expect(turboData == localPayload)
  }

  /// The weights these packages install are very often already on the machine,
  /// pulled by `hf` or `transformers`. The Hub names every cached LFS file
  /// `blobs/<oid>`, and that oid is the SHA-256 the manifest declares — so a
  /// cached copy is a direct lookup away.
  ///
  /// Proved by refusing every download: if the package installs at all, it
  /// came out of the cache.
  @Test("A file already in the Hugging Face cache installs without downloading")
  func reusesHuggingFaceCache() async throws {
    let payload = Data("weights already pulled by the hub client".utf8)
    let digest = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    // A cache laid out the way the Hub client lays one out.
    let hub = root.appendingPathComponent("hub", isDirectory: true)
    let blobs = hub
      .appendingPathComponent("models--vendor--fixture", isDirectory: true)
      .appendingPathComponent("blobs", isDirectory: true)
    try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
    try payload.write(to: blobs.appendingPathComponent(digest, isDirectory: false))
    setenv("HF_HUB_CACHE", hub.path, 1)
    defer { unsetenv("HF_HUB_CACHE") }

    let manifest = makeManifest(payload: payload)
    let downloader = ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root.appendingPathComponent("install")),
      transport: RefusingTransport(),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    )
    // What the app quotes before the download has to agree with what the
    // download turns out to be.
    #expect(await downloader.pendingByteCount(for: manifest) == 0)

    let installedURL = try await downloader.download(manifest)
    let installed = try Data(
      contentsOf: installedURL.appendingPathComponent("weights/model.safetensors")
    )
    #expect(installed == payload)
  }

  /// A cached file whose bytes do not match the declared hash is not trusted
  /// on the strength of its name — it is verified and rejected, and the
  /// download proceeds as if it had never been there.
  @Test("A cache entry that fails verification does not get installed")
  func rejectsMismatchedCacheEntry() async throws {
    let payload = Data("the bytes the manifest describes".utf8)
    let digest = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let hub = root.appendingPathComponent("hub", isDirectory: true)
    let blobs = hub
      .appendingPathComponent("models--vendor--fixture", isDirectory: true)
      .appendingPathComponent("blobs", isDirectory: true)
    try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
    // Right name and right length, wrong bytes.
    try Data("the bytes the manifest describe!".utf8)
      .write(to: blobs.appendingPathComponent(digest, isDirectory: false))
    setenv("HF_HUB_CACHE", hub.path, 1)
    defer { unsetenv("HF_HUB_CACHE") }

    let manifest = makeManifest(payload: payload)
    let installedURL = try await ModelPackageDownloader(
      store: ModelPackageStore(rootURL: root.appendingPathComponent("install")),
      transport: FixtureTransport(payload: payload),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(manifest)
    let installed = try Data(
      contentsOf: installedURL.appendingPathComponent("weights/model.safetensors")
    )
    #expect(installed == payload)
  }

  @Test("A changed manifest updates in place and keeps unchanged bytes")
  func updatesInstalledPackage() async throws {
    let shared = Data("weights that do not change".utf8)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelPackageStore(rootURL: root)

    func manifest(extra: Data?) -> ModelPackageManifest {
      var files = [
        ModelPackageFile(
          role: .transformer,
          path: "weights/model.safetensors",
          byteCount: Int64(shared.count),
          sha256: SHA256.hash(data: shared).map { String(format: "%02x", $0) }.joined()
        )
      ]
      if let extra {
        files.append(
          ModelPackageFile(
            role: .videoVAE,
            path: "vae/added.safetensors",
            byteCount: Int64(extra.count),
            sha256: SHA256.hash(data: extra).map { String(format: "%02x", $0) }.joined()
          )
        )
      }
      return ModelPackageManifest(
        id: "fixture-model", displayName: "Fixture", detail: "d",
        repository: "example/fixture", revision: String(repeating: "a", count: 40),
        licenseName: "Test", licenseURL: URL(string: "https://example.com/l")!,
        minimumUnifiedMemoryBytes: 1, compatibility: .ready, files: files
      )
    }

    _ = try await ModelPackageDownloader(
      store: store, transport: FixtureTransport(payload: shared),
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(manifest(extra: nil))

    // The updated manifest adds a file; the unchanged one must not be fetched.
    let added = Data("a newly added file".utf8)
    let updated = manifest(extra: added)
    let transport = SingleFileTransport(payload: added)
    let installedURL = try await ModelPackageDownloader(
      store: store, transport: transport,
      capacityChecker: FixedCapacityChecker(bytes: .max)
    ).download(updated)

    #expect(
      try Data(contentsOf: installedURL.appendingPathComponent("vae/added.safetensors"))
        == added
    )
    #expect(
      try Data(contentsOf: installedURL.appendingPathComponent("weights/model.safetensors"))
        == shared
    )
    #expect(await transport.requestCount == 1)
    #expect(
      !FileManager.default.fileExists(
        atPath: installedURL.appendingPathExtension("previous").path)
    )
  }

  @Test("A local-only file with no verified copy fails with a clear error")
  func missingLocalSource() async throws {
    let payload = Data("never hosted".utf8)
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let manifest = ModelPackageManifest(
      id: "fixture-local-only",
      displayName: "Fixture",
      detail: "Requires a local file that is absent",
      repository: "example/fixture",
      revision: String(repeating: "a", count: 40),
      licenseName: "Test",
      licenseURL: URL(string: "https://example.com/license")!,
      minimumUnifiedMemoryBytes: 1,
      compatibility: .ready,
      files: [
        ModelPackageFile(
          role: .transformer,
          path: "weights/turbo.safetensors",
          byteCount: Int64(payload.count),
          sha256: SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined(),
          localCandidatePath: root.appendingPathComponent("absent.safetensors").path,
          requiresLocalSource: true
        )
      ]
    )

    await #expect(throws: ModelDownloadError.localSourceUnavailable(file: "turbo.safetensors")) {
      _ = try await ModelPackageDownloader(
        store: ModelPackageStore(rootURL: root),
        transport: RefusingTransport(),
        capacityChecker: FixedCapacityChecker(bytes: .max)
      ).download(manifest)
    }
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

private actor SingleFileTransport: ModelFileTransport {
  let payload: Data
  private(set) var requestCount = 0

  init(payload: Data) { self.payload = payload }

  func download(
    request: URLRequest,
    to destination: URL,
    existingBytes: Int64,
    progress: @escaping @Sendable (Int64) -> Void
  ) async throws -> ModelHTTPResponse {
    requestCount += 1
    try payload.write(to: destination)
    progress(Int64(payload.count))
    return ModelHTTPResponse(statusCode: 200, bytesWritten: Int64(payload.count))
  }
}

private struct RefusingTransport: ModelFileTransport {
  struct UnexpectedRequest: Error {}

  func download(
    request: URLRequest,
    to destination: URL,
    existingBytes: Int64,
    progress: @escaping @Sendable (Int64) -> Void
  ) async throws -> ModelHTTPResponse {
    throw UnexpectedRequest()
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
