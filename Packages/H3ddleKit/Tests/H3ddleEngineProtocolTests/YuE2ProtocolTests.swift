import Foundation
import H3ddleEngineProtocol
import H3ddleGeneration
import Testing

@Suite("YuE2 request persistence")
struct YuE2ProtocolTests {
  @Test("Every selectable budget survives the protocol, queue, and recipe", arguments: YuE2SynthesisSteps.allCases, ["", "[Verse]\nA song by the sea"])
  func roundTrip(steps: YuE2SynthesisSteps, lyrics: String) throws {
    let music = EngineMusicOptions(lyrics: lyrics)
    let wire = EngineGenerationRequest(
      kind: .music, prompt: "Acoustic folk", duration: 360,
      denoisingSteps: steps.rawValue, music: music,
      outputURL: URL(fileURLWithPath: "/tmp/yue2-test.wav")
    )
    let decoded = try JSONDecoder().decode(
      EngineGenerationRequest.self, from: JSONEncoder().encode(wire))
    #expect(decoded.music == music)
    #expect(decoded.denoisingSteps == steps.rawValue)
    #expect(decoded.kind == .music)
    #expect(decoded.requiresExclusiveModelMemory)

    let request = GenerationRequest(
      kind: .audio, audioEngine: .yue2, music: music,
      prompt: "Acoustic folk", duration: 360, denoisingSteps: steps.rawValue, seed: 42)
    let job = GenerationQueueJob(
      request: request, modelID: "local-yue2", displayPrompt: request.prompt,
      settingsDescription: "YuE2", studioSettings: .makeDefault(seed: 42), aspectRatio: "1:1")
    let restored = try JSONDecoder().decode(
      GenerationQueueJob.self, from: JSONEncoder().encode(job))
    #expect(restored.request == request)
    let recipe = GenerationRecipe(job: restored, projectAssets: [])
    let restoredRecipe = try JSONDecoder().decode(
      GenerationRecipe.self, from: JSONEncoder().encode(recipe))
    #expect(restoredRecipe.request.music == music)
    #expect(restoredRecipe.request.audioEngine == .yue2)
    #expect(restoredRecipe.request.denoisingSteps == steps.rawValue)
  }

  @Test("Older audio jobs without music options still decode")
  func legacyRequest() throws {
    let request = GenerationRequest(kind: .audio, audioEngine: .stableAudio, prompt: "Rain", duration: 5)
    var object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any])
    object.removeValue(forKey: "music")
    let restored = try JSONDecoder().decode(
      GenerationRequest.self, from: JSONSerialization.data(withJSONObject: object))
    #expect(restored.audioEngine == .stableAudio)
    #expect(restored.music == nil)
    #expect(YuE2SynthesisSteps(rawValue: 24) == nil)
    #expect(YuE2SynthesisSteps.reference.rawValue == 32)
  }
}
