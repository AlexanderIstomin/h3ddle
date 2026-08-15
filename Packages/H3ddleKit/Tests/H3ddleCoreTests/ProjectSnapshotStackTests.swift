import Foundation
import Testing

@testable import H3ddleCore

@Suite("Project snapshot stack")
struct ProjectSnapshotStackTests {
  @Test("Checkpoint then undo restores the previous project")
  func undoRestoresCheckpoint() {
    var stack = ProjectSnapshotStack()
    var project = H3ddleProject(name: "Before")
    stack.checkpoint(project)
    project.name = "After"
    let restored = stack.popUndo(current: project)
    #expect(restored?.name == "Before")
    #expect(!stack.canUndo)
    #expect(stack.canRedo)
  }

  @Test("Redo returns the state that undo left")
  func redoReturnsForwardState() {
    var stack = ProjectSnapshotStack()
    var project = H3ddleProject(name: "A")
    stack.checkpoint(project)
    project.name = "B"
    project = stack.popUndo(current: project) ?? project
    #expect(project.name == "A")
    project = stack.popRedo(current: project) ?? project
    #expect(project.name == "B")
    #expect(!stack.canRedo)
    #expect(stack.canUndo)
  }

  @Test("A new checkpoint clears redo")
  func checkpointClearsRedo() {
    var stack = ProjectSnapshotStack()
    var project = H3ddleProject(name: "A")
    stack.checkpoint(project)
    project.name = "B"
    project = stack.popUndo(current: project) ?? project
    stack.checkpoint(project)
    #expect(!stack.canRedo)
    #expect(stack.popRedo(current: project) == nil)
  }

  @Test("The stack drops the oldest entry past 50")
  func capsAtFifty() {
    var stack = ProjectSnapshotStack()
    var project = H3ddleProject(name: "0")
    for index in 1...51 {
      stack.checkpoint(project)
      project.name = "\(index)"
    }
    #expect(stack.undo.count == 50)
    #expect(stack.undo.first?.name == "1")
    #expect(stack.undo.last?.name == "50")
  }

  @Test("Empty pop is a no-op")
  func emptyPop() {
    var stack = ProjectSnapshotStack()
    let project = H3ddleProject(name: "Now")
    #expect(stack.popUndo(current: project) == nil)
    #expect(stack.popRedo(current: project) == nil)
  }
}
