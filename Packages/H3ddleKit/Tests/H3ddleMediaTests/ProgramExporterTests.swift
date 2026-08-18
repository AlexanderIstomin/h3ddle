import AVFoundation
import CoreGraphics
import Foundation
import H3ddleCore
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import H3ddleMedia

@Suite("Program exporter")
struct ProgramExporterTests {
  @Test("Empty visual programs are rejected")
  func emptyProgram() async {
    let destination = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-empty-\(UUID().uuidString).mp4")
    do {
      for try await _ in ProgramExporter().export(
        project: H3ddleProject(),
        settings: ProgramExportSettings(),
        destination: destination
      ) {
      }
      Issue.record("Expected an empty-program error")
    } catch MediaExportError.emptyProgram {
    } catch {
      Issue.record("Unexpected error \(error)")
    }
  }

  @Test("Image plus trailing audio exports the visual duration")
  func exportsVisualDuration() async throws {
    let folder = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-export-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }

    let imageURL = folder.appendingPathComponent("still.png")
    let audioURL = folder.appendingPathComponent("tone.wav")
    try ExportTestMedia.writePNG(to: imageURL, width: 32, height: 18, red: 220, green: 80, blue: 30)
    try ExportTestMedia.writeTone(to: audioURL, duration: 2)

    var project = H3ddleProject()
    project.settings.apply(resolution: .extreme)
    project.settings.apply(frameRate: 8)
    let image = AssetReference(kind: .image, displayName: "Still", url: imageURL, duration: 2)
    let audio = AssetReference(kind: .audio, displayName: "Tone", url: audioURL, duration: 2)
    project.addAsset(image)
    project.addAsset(audio)
    try project.timeline.appendVisual(image)
    try project.timeline.appendAudio(audio)

    var settings = ProgramExportSettings.makeDefault(project: project)
    settings.updateCustom {
      $0.resolution = .extreme
      $0.framesPerSecond = 24
      $0.videoBitrateKbps = 400
      $0.format = .h264
    }

    let destination = folder.appendingPathComponent("out.mp4")
    var completed: URL?
    var sawProgress = false
    for try await event in ProgramExporter().export(
      project: project,
      settings: settings,
      destination: destination
    ) {
      switch event {
      case .progress:
        sawProgress = true
      case .completed(let url):
        completed = url
      case .preparing, .preview:
        break
      }
    }

    #expect(sawProgress)
    #expect(completed == destination)
    #expect(FileManager.default.fileExists(atPath: destination.path))

    let asset = AVURLAsset(url: destination)
    let duration = try await asset.load(.duration)
    #expect(abs(duration.seconds - 2) < 0.35)
    let videoTracks = try await asset.loadTracks(withMediaType: .video)
    #expect(videoTracks.count == 1)
    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
    #expect(audioTracks.count == 1)
  }

  @Test("A disabled audio lane is omitted from the file")
  func omittedAudioLane() async throws {
    let folder = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-mute-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }

    let imageURL = folder.appendingPathComponent("still.png")
    let audioURL = folder.appendingPathComponent("tone.wav")
    try ExportTestMedia.writePNG(to: imageURL, width: 32, height: 18, red: 10, green: 80, blue: 40)
    try ExportTestMedia.writeTone(to: audioURL, duration: 1)

    var project = H3ddleProject()
    let image = AssetReference(kind: .image, displayName: "Still", url: imageURL, duration: 0.5)
    let audio = AssetReference(kind: .audio, displayName: "Tone", url: audioURL, duration: 1)
    project.addAsset(image)
    project.addAsset(audio)
    try project.timeline.appendVisual(image)
    try project.timeline.appendAudio(audio)

    var settings = ProgramExportSettings.makeDefault(project: project)
    settings.updateCustom {
      $0.resolution = .extreme
      $0.framesPerSecond = 8
      $0.videoBitrateKbps = 400
    }
    settings.includeAudioLane = false

    let destination = folder.appendingPathComponent("silent.mp4")
    for try await _ in ProgramExporter().export(
      project: project,
      settings: settings,
      destination: destination
    ) {
    }

    let asset = AVURLAsset(url: destination)
    #expect(!(try await asset.loadTracks(withMediaType: .video)).isEmpty)
    #expect((try await asset.loadTracks(withMediaType: .audio)).isEmpty)
  }

  @Test("A custom range shortens the file")
  func customRange() async throws {
    let folder = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-range-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }

    let imageURL = folder.appendingPathComponent("still.png")
    try ExportTestMedia.writePNG(to: imageURL, width: 32, height: 32, red: 20, green: 40, blue: 80)
    var project = H3ddleProject()
    let image = AssetReference(kind: .image, displayName: "Still", url: imageURL, duration: 1)
    project.addAsset(image)
    try project.timeline.appendVisual(image)

    var settings = ProgramExportSettings.makeDefault(project: project)
    settings.updateCustom {
      $0.resolution = .extreme
      $0.framesPerSecond = 8
      $0.videoBitrateKbps = 400
      $0.range = ProgramExportRange(mode: .custom, inSec: 0, outSec: 0.4)
    }

    let destination = folder.appendingPathComponent("clip.mp4")
    for try await _ in ProgramExporter().export(
      project: project,
      settings: settings,
      destination: destination
    ) {
    }

    let duration = try await AVURLAsset(url: destination).load(.duration)
    #expect(duration.seconds < 0.7)
    #expect(duration.seconds > 0.2)
  }

  @Test("Software encode with audio does not hang or crash")
  func softwareEncodeWithAudio() async throws {
    let folder = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-sw-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }

    let imageURL = folder.appendingPathComponent("still.png")
    let audioURL = folder.appendingPathComponent("tone.wav")
    try ExportTestMedia.writePNG(to: imageURL, width: 32, height: 18, red: 30, green: 120, blue: 200)
    try ExportTestMedia.writeTone(to: audioURL, duration: 3)

    var project = H3ddleProject()
    let image = AssetReference(kind: .image, displayName: "Still", url: imageURL, duration: 2)
    let audio = AssetReference(kind: .audio, displayName: "Tone", url: audioURL, duration: 3)
    project.addAsset(image)
    project.addAsset(audio)
    try project.timeline.appendVisual(image)
    try project.timeline.appendAudio(audio)

    var settings = ProgramExportSettings.makeDefault(project: project)
    settings.updateCustom {
      $0.resolution = .extreme
      $0.framesPerSecond = 24
      $0.videoBitrateKbps = 400
      $0.format = .h264
    }
    settings.setAdditiveHardwareAcceleration(false)

    let destination = folder.appendingPathComponent("soft.mp4")
    for try await _ in ProgramExporter().export(
      project: project,
      settings: settings,
      destination: destination
    ) {
    }

    let asset = AVURLAsset(url: destination)
    let duration = try await asset.load(.duration)
    #expect(abs(duration.seconds - 2) < 0.35)
    #expect(!(try await asset.loadTracks(withMediaType: .video)).isEmpty)
    #expect(!(try await asset.loadTracks(withMediaType: .audio)).isEmpty)
  }

  @Test("Cancelling the stream stops the writer")
  func cancelStopsWriter() async throws {
    let folder = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-cancel-\(UUID().uuidString)", isDirectory: true)
    try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }

    let imageURL = folder.appendingPathComponent("still.png")
    try? ExportTestMedia.writePNG(to: imageURL, width: 64, height: 36, red: 10, green: 10, blue: 10)
    var project = H3ddleProject()
    let image = AssetReference(kind: .image, displayName: "Still", url: imageURL, duration: 3)
    project.addAsset(image)
    try project.timeline.appendVisual(image)

    var settings = ProgramExportSettings.makeDefault(project: project)
    settings.updateCustom {
      $0.resolution = .sd
      $0.framesPerSecond = 24
    }

    let destination = folder.appendingPathComponent("cancel.mp4")
    let stream = ProgramExporter().export(
      project: project,
      settings: settings,
      destination: destination
    )
    let task = Task {
      for try await _ in stream {}
    }
    task.cancel()
    do {
      try await task.value
    } catch MediaExportError.cancelled {
    } catch is CancellationError {
    } catch {
      Issue.record("Unexpected cancel error \(error)")
    }
  }
}

@Suite("Loudness")
struct LoudnessTests {
  @Test("A quiet sine needs boost to reach the export target")
  func quietSineNeedsBoost() {
    let sampleRate = 48_000.0
    let count = Int(sampleRate)
    var samples = [Float](repeating: 0, count: count)
    let twoPi = 2 * Float.pi
    for index in 0..<count {
      samples[index] = sin(twoPi * 1_000 * Float(index) / Float(sampleRate)) * 0.05
    }
    let measured = Loudness.integratedLUFS(samples: samples, sampleRate: sampleRate, channels: 1)
    let gain = Loudness.gain(from: measured)
    #expect(measured < -14)
    #expect(gain > 1)
    #expect(gain < 20)
  }

  @Test("Streaming chunks preserve the integrated measurement")
  func streamingChunks() {
    let sampleRate = 48_000.0
    let channels = 2
    let frameCount = Int(sampleRate * 1.25)
    var samples = [Float](repeating: 0, count: frameCount * channels)
    for frame in 0..<frameCount {
      let amplitude: Float = frame < frameCount / 2 ? 0.03 : 0.12
      let sample = sin(2 * Float.pi * 440 * Float(frame) / Float(sampleRate)) * amplitude
      samples[frame * channels] = sample
      samples[frame * channels + 1] = sample * 0.8
    }

    let expected = Loudness.integratedLUFS(
      samples: samples,
      sampleRate: sampleRate,
      channels: channels
    )
    var meter = Loudness.Meter(sampleRate: sampleRate, channels: channels)
    samples.withUnsafeBufferPointer { pointer in
      var offset = 0
      var chunkIndex = 0
      let chunkFrames = [73, 521, 1_024, 257]
      while offset < pointer.count {
        let count = min(chunkFrames[chunkIndex % chunkFrames.count] * channels, pointer.count - offset)
        meter.append(UnsafeBufferPointer(rebasing: pointer[offset..<(offset + count)]))
        offset += count
        chunkIndex += 1
      }
    }

    #expect(abs(meter.integratedLUFS - expected) < 0.000_1)
  }
}

enum ExportTestMedia {
  static func writePNG(
    to url: URL,
    width: Int,
    height: Int,
    red: UInt8,
    green: UInt8,
    blue: UInt8
  ) throws {
    var pixels = [UInt8](repeating: 255, count: width * height * 4)
    for index in stride(from: 0, to: pixels.count, by: 4) {
      pixels[index] = red
      pixels[index + 1] = green
      pixels[index + 2] = blue
    }
    let context = CGContext(
      data: &pixels,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )
    guard let image = context?.makeImage(),
      let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
      )
    else {
      throw MediaExportError.failed("Could not write a test still.")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw MediaExportError.failed("Could not finalize a test still.")
    }
  }

  static func writeTone(to url: URL, duration: TimeInterval) throws {
    let sampleRate = 22_050
    let sampleCount = max(1, Int((duration * Double(sampleRate)).rounded()))
    var data = Data()
    data.reserveCapacity(44 + sampleCount * 2)

    func appendUInt32(_ value: UInt32) {
      var value = value.littleEndian
      withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    func appendUInt16(_ value: UInt16) {
      var value = value.littleEndian
      withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }

    data.append(contentsOf: Array("RIFF".utf8))
    appendUInt32(UInt32(36 + sampleCount * 2))
    data.append(contentsOf: Array("WAVE".utf8))
    data.append(contentsOf: Array("fmt ".utf8))
    appendUInt32(16)
    appendUInt16(1)
    appendUInt16(1)
    appendUInt32(UInt32(sampleRate))
    appendUInt32(UInt32(sampleRate * 2))
    appendUInt16(2)
    appendUInt16(16)
    data.append(contentsOf: Array("data".utf8))
    appendUInt32(UInt32(sampleCount * 2))

    let twoPi = 2.0 * Double.pi
    for index in 0..<sampleCount {
      let sample = sin(twoPi * 220 * Double(index) / Double(sampleRate)) * 0.22
      let quantized = Int16((sample * Double(Int16.max)).rounded())
      appendUInt16(UInt16(bitPattern: quantized))
    }
    try data.write(to: url, options: .atomic)
  }
}
