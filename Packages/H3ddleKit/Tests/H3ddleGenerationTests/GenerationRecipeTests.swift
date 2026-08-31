import Foundation
import H3ddleCore
import Testing

@testable import H3ddleGeneration

@Suite("Generation recipes")
struct GenerationRecipeTests {
  @Test("Recipes retain parameters without leaking input paths")
  func portableRecipe() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleRecipe-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceURL = root.appendingPathComponent("private-reference.png")
    try Data([1]).write(to: referenceURL)
    let reference = AssetReference(
      kind: .image,
      displayName: "Reference",
      url: referenceURL,
      duration: 3
    )
    let request = GenerationRequest(
      kind: .video,
      prompt: "composed engine prompt",
      duration: 5,
      quality: .preview,
      denoisingSteps: 4,
      seed: 42,
      firstFrameURL: referenceURL
    )
    let target = GenerationReplacementTarget(
      projectID: UUID(),
      clipID: UUID(),
      expectedAssetID: AssetID()
    )
    let job = GenerationQueueJob(
      request: request,
      modelID: "fast-h3-vsa",
      displayPrompt: "A woman walking through rain",
      settingsDescription: "FastH3 · 480p",
      studioSettings: .makeDefault(seed: 42),
      aspectRatio: "16:9",
      soundscape: "rain",
      music: "none",
      replacementTarget: target
    )

    let recipe = GenerationRecipe(
      job: job,
      projectAssets: [reference],
      parentAssetID: target.expectedAssetID
    )
    let json = try recipe.encodedJSON()
    let text = String(decoding: json, as: UTF8.self)

    #expect(!text.contains(root.path))
    #expect(text.contains("private-reference.png"))
    #expect(recipe.parentAssetID == target.expectedAssetID)
    #expect(recipe.modelID == "fast-h3-vsa")
    #expect(recipe.displayPrompt == "A woman walking through rain")
    #expect(recipe.soundscape == "rain")
    #expect(recipe.music == "none")
    #expect(try recipe.resolvedRequest(projectAssets: [reference]).firstFrameURL == referenceURL)
  }

  @Test("Recipe metadata survives an asset round trip")
  func assetMetadata() throws {
    let job = GenerationQueueJob(
      request: GenerationRequest(kind: .image, prompt: "portrait", duration: 3, seed: 7),
      modelID: "image-model",
      displayPrompt: "portrait",
      settingsDescription: "image",
      studioSettings: .makeDefault(seed: 7),
      aspectRatio: "1:1"
    )
    let recipe = GenerationRecipe(job: job, projectAssets: [])
    let asset = AssetReference(
      kind: .image,
      displayName: "Portrait",
      url: URL(fileURLWithPath: "/tmp/portrait.png"),
      duration: 3
    )

    let restored = try GenerationRecipe.recipe(from: recipe.attaching(to: asset))

    #expect(restored == recipe)
  }

  @Test("Unresolved reference inputs fail visibly")
  func missingReference() throws {
    let missing = URL(fileURLWithPath: "/private/missing/reference.png")
    let job = GenerationQueueJob(
      request: GenerationRequest(
        kind: .video,
        prompt: "reference",
        duration: 5,
        referenceImageURLs: [missing]
      ),
      displayPrompt: "reference",
      settingsDescription: "video",
      studioSettings: .makeDefault(seed: 9),
      aspectRatio: "16:9"
    )
    let recipe = GenerationRecipe(job: job, projectAssets: [])

    #expect(throws: GenerationRecipeError.missingInput("reference.png")) {
      try recipe.resolvedRequest(projectAssets: [])
    }
  }
}
