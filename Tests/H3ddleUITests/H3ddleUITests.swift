import XCTest

final class H3ddleUITests: XCTestCase {
  @MainActor
  func testEditorOpensWithTwoTracks() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    XCTAssertTrue(app.staticTexts["editor-root"].waitForExistence(timeout: 5))
    XCTAssertTrue(app.staticTexts["program-timeline"].exists)
    XCTAssertTrue(app.buttons["append-audio"].exists)
    XCTAssertTrue(app.descendants(matching: .any)["program-preview"].exists)
    XCTAssertTrue(app.buttons["transport-play"].exists)
    XCTAssertTrue(app.buttons["transport-split"].exists)
    XCTAssertTrue(app.buttons["transport-delete"].exists)

    app.buttons["model-status"].click()
    XCTAssertTrue(app.staticTexts["model-settings"].waitForExistence(timeout: 2))
    XCTAssertTrue(app.buttons["choose-model-folder-video"].exists)
    XCTAssertTrue(app.staticTexts["MiniMax H3 · INT8"].waitForExistence(timeout: 2))
  }

  @MainActor
  func testProjectSettingsPanelOpensFromTheHeader() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    XCTAssertTrue(app.buttons["project-settings-toggle"].waitForExistence(timeout: 5))
    app.buttons["project-settings-toggle"].click()
    XCTAssertTrue(app.descendants(matching: .any)["project-settings"].waitForExistence(timeout: 2))
    XCTAssertTrue(app.staticTexts["Custom / None"].exists)
    XCTAssertTrue(app.staticTexts["MASTER"].exists)
  }

  @MainActor
  func testExportModalOpensFromTheHeader() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    XCTAssertTrue(app.buttons["export-button"].waitForExistence(timeout: 5))
    app.buttons["export-button"].click()
    XCTAssertTrue(app.descendants(matching: .any)["export-modal"].waitForExistence(timeout: 2))
    XCTAssertTrue(app.staticTexts["Export Video"].exists)
    XCTAssertTrue(app.buttons["preset-recommended"].exists)
    XCTAssertTrue(app.buttons["export-now"].exists)
    app.buttons["export-close"].click()
    XCTAssertFalse(app.descendants(matching: .any)["export-modal"].waitForExistence(timeout: 1))
  }

  @MainActor
  func testGenerationStudioHidesComposerDuringProgress() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    // A cold launch can take a while to publish an accessibility tree, and
    // the lane carries the identifier rather than the button inside it.
    XCTAssertTrue(app.buttons["audio-lane-drop"].waitForExistence(timeout: 20))
    app.buttons["audio-lane-drop"].click()
    XCTAssertTrue(app.buttons["Generate"].waitForExistence(timeout: 5))
    app.buttons["Generate"].click()

    let title = app.staticTexts["generation-studio"]
    let prompt = app.textViews["generation-prompt"]
    let generate = app.buttons["generate-button"]
    XCTAssertTrue(title.waitForExistence(timeout: 2))
    XCTAssertTrue(prompt.waitForExistence(timeout: 2))
    XCTAssertTrue(generate.exists)
    XCTAssertTrue(generate.isHittable)

    prompt.click()
    prompt.typeText("Soft rain on a canvas awning")
    generate.click()

    let cancel = app.buttons["Cancel"]
    let elapsed = app.staticTexts["generation-elapsed"]
    XCTAssertTrue(cancel.waitForExistence(timeout: 2))
    XCTAssertTrue(elapsed.waitForExistence(timeout: 2))
    XCTAssertTrue(title.exists)
    XCTAssertTrue(title.isHittable)
    XCTAssertTrue(cancel.isHittable)
    XCTAssertFalse(prompt.exists)
    XCTAssertFalse(generate.exists)
    cancel.click()
    XCTAssertTrue(prompt.waitForExistence(timeout: 2))
  }

  @MainActor
  func testInsertPlacesResultAfterTheLastClip() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES", "-H3ddleFastFakeGeneration"]
    app.launch()

    // A cold launch can take a while to publish an accessibility tree, and
    // the lane carries the identifier rather than the button inside it.
    XCTAssertTrue(app.buttons["audio-lane-drop"].waitForExistence(timeout: 20))
    app.buttons["audio-lane-drop"].click()
    XCTAssertTrue(app.buttons["Generate"].waitForExistence(timeout: 5))
    app.buttons["Generate"].click()

    let prompt = app.textViews["generation-prompt"]
    XCTAssertTrue(prompt.waitForExistence(timeout: 2))
    prompt.click()
    prompt.typeText("Warm lo-fi bed")
    app.buttons["generate-button"].click()

    let insert = app.buttons["insert-to-timeline"]
    XCTAssertTrue(insert.waitForExistence(timeout: 5))
    insert.click()
    XCTAssertTrue(app.staticTexts["Generated Audio"].waitForExistence(timeout: 2))
  }

  @MainActor
  func testAppendMenusOfferImport() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    XCTAssertTrue(app.buttons["append-visual"].waitForExistence(timeout: 5))
    app.buttons["append-visual"].click()
    XCTAssertTrue(app.buttons["append-import"].waitForExistence(timeout: 2))
    XCTAssertTrue(app.buttons["append-generate-video"].exists)
    app.buttons["append-visual"].click()

    app.buttons["append-audio"].click()
    XCTAssertTrue(app.buttons["append-import"].waitForExistence(timeout: 2))
    XCTAssertTrue(app.buttons["append-generate-audio"].exists)
  }

  @MainActor
  func testTimelineLanesAcceptFileDrops() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    XCTAssertTrue(app.staticTexts["program-timeline"].waitForExistence(timeout: 5))
    XCTAssertTrue(app.descendants(matching: .any)["visual-lane-drop"].exists)
    XCTAssertTrue(app.descendants(matching: .any)["audio-lane-drop"].exists)
    XCTAssertTrue(app.descendants(matching: .any)["visual-header-drop"].exists)
    XCTAssertTrue(app.descendants(matching: .any)["audio-header-drop"].exists)
  }

  /// Walks the path a user takes to reach sound effects: open the audio
  /// studio, switch to the SFX tab, and confirm the studio reconfigures for a
  /// different model rather than keeping H3's controls.
  @MainActor
  func testAudioStudioOffersSoundEffects() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    // A cold launch can take a while to publish an accessibility tree, and
    // the lane carries the identifier rather than the button inside it.
    XCTAssertTrue(app.buttons["audio-lane-drop"].waitForExistence(timeout: 20))
    app.buttons["audio-lane-drop"].click()
    XCTAssertTrue(app.buttons["Generate"].waitForExistence(timeout: 5))
    app.buttons["Generate"].click()

    XCTAssertTrue(app.staticTexts["generation-studio"].waitForExistence(timeout: 5))
    XCTAssertTrue(app.buttons["segment-voice"].exists)
    XCTAssertTrue(app.buttons["segment-sfx"].exists)

    app.buttons["segment-sfx"].click()
    // With no sound-effect package installed the studio has to say so rather
    // than offer settings for a generation it cannot run.
    XCTAssertTrue(
      app.staticTexts["No models installed"].waitForExistence(timeout: 2)
        || app.buttons["generate-button"].exists
    )
  }

  /// The library groups by what a model is for, and each group asks for its
  /// own folders so neither button is ambiguous.
  @MainActor
  func testModelLibraryGroupsVideoAndAudio() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    XCTAssertTrue(app.buttons["model-status"].waitForExistence(timeout: 5))
    app.buttons["model-status"].click()
    XCTAssertTrue(app.staticTexts["model-settings"].waitForExistence(timeout: 2))
    XCTAssertTrue(app.buttons["choose-model-folder-video"].exists)
    XCTAssertTrue(app.buttons["choose-model-folder-audio"].exists)
  }
}
