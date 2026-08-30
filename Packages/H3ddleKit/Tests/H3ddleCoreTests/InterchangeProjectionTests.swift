import Foundation
import Testing

@testable import H3ddleCore

@Suite("Interchange projection")
struct InterchangeProjectionTests {
  @Test("Empty project round-trips identity")
  func emptyProjectRoundTrips() throws {
    let project = H3ddleProject(name: "Untitled Project")
    let restored = try roundTrip(project)
    #expect(restored.name == project.name)
    #expect(restored.settings == project.settings)
    #expect(restored.timeline.visualItems.isEmpty)
    #expect(restored.timeline.audioItems.isEmpty)
    #expect(restored.timeline.textItems.isEmpty)
    #expect(restored.id == project.id)
  }

  @Test("A rich program round-trips assets, timing, text, effects, and transitions")
  func richProjectRoundTrips() throws {
    var project = H3ddleProject(
      name: "Street Night",
      settings: ProjectSettings(
        width: 1080,
        height: 1920,
        framesPerSecond: 24,
        background: .navy,
        platform: .tiktok,
        masterGain: 0.8,
        toneMapping: .agx,
        exposure: 1.25
      )
    )
    let video = AssetReference(
      kind: .video,
      displayName: "Opening",
      url: URL(fileURLWithPath: "/tmp/opening.mp4"),
      duration: 6
    )
    let still = AssetReference(
      kind: .image,
      displayName: "Poster",
      url: URL(fileURLWithPath: "/tmp/poster.png"),
      duration: 3
    )
    let music = AssetReference(
      kind: .audio,
      displayName: "Bed",
      url: URL(fileURLWithPath: "/tmp/bed.wav"),
      duration: 8
    )
    project.addAsset(video)
    project.addAsset(still)
    project.addAsset(music)
    try project.timeline.appendVisual(video)
    try project.timeline.appendVisual(still)
    try project.timeline.appendAudio(music)

    let first = project.timeline.visualItems[0]
    project.timeline.setVisualTrim(
      first.id,
      VisualTrim(duration: 5, sourceOffset: 0.5, gapBefore: 1)
    )
    project.timeline.setVisualCanvasTransform(
      first.id,
      CanvasObjectTransform(
        fit: .cover,
        translationX: 0.1,
        translationY: -0.2,
        scale: 1.4,
        rotationRadians: 0.3
      )
    )
    project.timeline.setVisualIncludesNativeAudio(first.id, includes: false)
    project.timeline.setVisualEnabled(first.id, isEnabled: false)
    _ = project.timeline.addVisualEffect(first.id, kind: .filmGrain)
    _ = project.timeline.addVisualEffect(first.id, kind: .colorGrade)

    let second = project.timeline.visualItems[1]
    project.timeline.setVisualTransition(
      second.id,
      VisualTransition(kind: .dissolve, duration: 0.5)
    )

    let audio = project.timeline.audioItems[0]
    project.timeline.setAudioTrim(
      audio.id,
      AudioTrim(startTime: 1.25, duration: 6, sourceOffset: 0.25)
    )
    project.timeline.setAudioEnabled(audio.id, isEnabled: false)

    let title = project.timeline.insertText(
      TextItem(
        startTime: 2,
        duration: 4,
        text: "Night market",
        style: TextStyle(
          fontFamily: "Helvetica",
          fontPostScriptName: "Helvetica-Bold",
          fontWeight: 700,
          fontSize: 64,
          alignment: .leading,
          fill: TextColor(r: 1, g: 0.9, b: 0.2, a: 1),
          wrap: .wrap,
          boxWidth: 800,
          lineHeight: 1.1,
          letterSpacing: 1,
          strokeWidth: 2,
          strokeColor: .black,
          shadowOffsetX: 2,
          shadowOffsetY: 3,
          shadowBlur: 4,
          backgroundColor: TextColor(r: 0, g: 0, b: 0, a: 0.4),
          backgroundPadding: 8,
          backgroundCornerRadius: 6
        ),
        canvasTransform: CanvasObjectTransform(
          translationX: 0.05,
          translationY: 0.15,
          scale: 1.1,
          rotationRadians: -0.1
        )
      )
    )
    project.timeline.setTextEnabled(title.id, isEnabled: false)

    let restored = try roundTrip(project)
    #expect(restored.name == project.name)
    #expect(restored.settings.width == 1080)
    #expect(restored.settings.height == 1920)
    #expect(restored.settings.framesPerSecond == 24)
    #expect(restored.settings.background == .navy)
    #expect(restored.settings.platform == .tiktok)
    #expect(abs(restored.settings.masterGain - 0.8) < 0.000_1)
    #expect(restored.settings.toneMapping == .agx)
    #expect(abs(restored.settings.exposure - 1.25) < 0.000_1)
    #expect(restored.assets.map(\.id) == project.assets.map(\.id))
    #expect(restored.assets.map(\.url) == project.assets.map(\.url))
    #expect(restored.assets.map(\.displayName) == project.assets.map(\.displayName))

    #expect(restored.timeline.visualItems.count == 2)
    let restoredFirst = restored.timeline.visualItems[0]
    #expect(abs(restoredFirst.duration - 5) < 0.000_1)
    #expect(abs(restoredFirst.sourceOffset - 0.5) < 0.000_1)
    #expect(abs(restoredFirst.gapBefore - 1) < 0.000_1)
    #expect(restoredFirst.canvasFit == .cover)
    #expect(abs(restoredFirst.translationX - 0.1) < 0.000_1)
    #expect(abs(restoredFirst.translationY + 0.2) < 0.000_1)
    #expect(abs(restoredFirst.uniformScale - 1.4) < 0.000_1)
    #expect(abs(restoredFirst.rotationRadians - 0.3) < 0.000_1)
    #expect(!restoredFirst.includesNativeAudio)
    #expect(!restoredFirst.isEnabled)
    #expect(restoredFirst.effects.map(\.kind) == [.filmGrain, .colorGrade])
    #expect(abs(restored.timeline.visualPlacements[0].startTime - 1) < 0.000_1)

    let restoredSecond = restored.timeline.visualItems[1]
    #expect(restoredSecond.transition?.kind == .dissolve)
    #expect(abs((restoredSecond.transition?.duration ?? 0) - 0.5) < 0.000_1)

    #expect(restored.timeline.audioItems.count == 1)
    let restoredAudio = restored.timeline.audioItems[0]
    #expect(abs(restoredAudio.startTime - 1.25) < 0.000_1)
    #expect(abs(restoredAudio.duration - 6) < 0.000_1)
    #expect(abs(restoredAudio.sourceOffset - 0.25) < 0.000_1)
    #expect(!restoredAudio.isEnabled)

    #expect(restored.timeline.textItems.count == 1)
    let restoredTitle = restored.timeline.textItems[0]
    #expect(restoredTitle.text == "Night market")
    #expect(!restoredTitle.isEnabled)
    #expect(abs(restoredTitle.startTime - 2) < 0.000_1)
    #expect(abs(restoredTitle.duration - 4) < 0.000_1)
    #expect(restoredTitle.style.fontFamily == "Helvetica")
    #expect(restoredTitle.style.fontPostScriptName == "Helvetica-Bold")
    #expect(restoredTitle.style.fontWeight == 700)
    #expect(abs(restoredTitle.style.fontSize - 64) < 0.000_1)
    #expect(restoredTitle.style.alignment == .leading)
    #expect(restoredTitle.style.wrap == .wrap)
    #expect(restoredTitle.style.boxWidth == 800)
    #expect(abs(restoredTitle.canvasTransform.translationX - 0.05) < 0.000_1)
    #expect(abs(restoredTitle.canvasTransform.scale - 1.1) < 0.000_1)
  }

  @Test("Unknown document keys survive encode/decode")
  func extrasRoundTrip() throws {
    let project = H3ddleProject(name: "Keep extras")
    let document = try InterchangeProjection.document(
      from: project,
      extras: [
        "sceneStudios": .object(["studio-1": .string("kept")]),
        "name": .string("should not overwrite"),
      ]
    )
    #expect(document.name == "Keep extras")
    #expect(document.extras["sceneStudios"] != nil)
    #expect(document.extras["name"] == nil)

    let data = try InterchangeProjection.encode(document)
    let decoded = try InterchangeProjection.decodeDocument(data)
    #expect(decoded.extras["sceneStudios"] == .object(["studio-1": .string("kept")]))
    #expect(decoded.name == "Keep extras")
  }

  @Test("A newer interchange schema is rejected")
  func rejectsFutureSchema() throws {
    let json = """
      {"schemaVersion":99,"id":"00000000-0000-0000-0000-000000000001","revision":0,"settings":{"width":1920,"height":1080,"fps":24},"assets":[],"sequences":[],"compositions":[]}
      """
    #expect(throws: InterchangeError.unsupportedSchemaVersion(99)) {
      _ = try InterchangeProjection.decodeDocument(Data(json.utf8))
    }
  }

  @Test("ACES tone mapping uses the interchange filmic identifier")
  func acesToneMappingIdentifier() throws {
    var project = H3ddleProject()
    project.settings.toneMapping = .aces
    let document = try InterchangeProjection.document(from: project) {
      $0.url.absoluteString
    }
    #expect(document.settings.toneMapping == "aces-filmic")
    let restored = try InterchangeProjection.project(from: document) { asset in
      URL(string: asset.src) ?? URL(fileURLWithPath: "/tmp/missing")
    }
    #expect(restored.settings.toneMapping == .aces)
  }

  private func roundTrip(_ project: H3ddleProject) throws -> H3ddleProject {
    let document = try InterchangeProjection.document(from: project) {
      $0.url.absoluteString
    }
    let data = try InterchangeProjection.encode(document)
    let decoded = try InterchangeProjection.decodeDocument(data)
    return try InterchangeProjection.project(from: decoded) { asset in
      guard let url = URL(string: asset.src) else {
        throw InterchangeError.invalidMediaURL
      }
      return url
    }
  }
}
