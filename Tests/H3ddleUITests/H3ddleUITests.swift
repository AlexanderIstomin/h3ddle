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

    app.buttons["model-status"].click()
    XCTAssertTrue(app.staticTexts["model-settings"].waitForExistence(timeout: 2))
    XCTAssertTrue(app.buttons["choose-model-folder"].exists)
    XCTAssertTrue(app.staticTexts["MiniMax H3 · INT8"].waitForExistence(timeout: 2))
  }

  @MainActor
  func testGenerationSheetKeepsItsChromeVisibleDuringProgress() {
    let app = XCUIApplication()
    app.launchArguments += ["-ApplePersistenceIgnoreState", "YES"]
    app.launch()

    XCTAssertTrue(app.buttons["append-audio"].waitForExistence(timeout: 5))
    app.buttons["append-audio"].click()

    let title = app.staticTexts["generation-studio"]
    let prompt = app.textViews["generation-prompt"]
    let generate = app.buttons["generate-and-append"]
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
    cancel.click()
  }
}
