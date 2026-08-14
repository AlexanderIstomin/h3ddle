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

    app.buttons["model-status"].click()
    XCTAssertTrue(app.staticTexts["model-settings"].waitForExistence(timeout: 2))
    XCTAssertTrue(app.buttons["choose-model-folder"].exists)
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
  func testGenerationStudioHidesComposerDuringProgress() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    XCTAssertTrue(app.buttons["append-audio"].waitForExistence(timeout: 5))
    app.buttons["append-audio"].click()
    XCTAssertTrue(app.buttons["Generate"].waitForExistence(timeout: 2))
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

    XCTAssertTrue(app.buttons["append-audio"].waitForExistence(timeout: 5))
    app.buttons["append-audio"].click()
    XCTAssertTrue(app.buttons["Generate"].waitForExistence(timeout: 2))
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
}
