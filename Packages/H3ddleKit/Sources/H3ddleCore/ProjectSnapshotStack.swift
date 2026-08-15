import Foundation

/// Snapshot undo/redo of `H3ddleProject`. One entry is the state *before*
/// a mutation. Live gestures must not checkpoint until they commit.
public struct ProjectSnapshotStack: Equatable, Sendable {
  public static let limit = 50

  public private(set) var undo: [H3ddleProject] = []
  public private(set) var redo: [H3ddleProject] = []

  public init() {}

  public var canUndo: Bool { !undo.isEmpty }
  public var canRedo: Bool { !redo.isEmpty }

  /// Push `current` (the state before the mutation). Clears redo.
  public mutating func checkpoint(_ current: H3ddleProject) {
    undo.append(current)
    if undo.count > Self.limit {
      undo.removeFirst(undo.count - Self.limit)
    }
    redo.removeAll()
  }

  public mutating func popUndo(current: H3ddleProject) -> H3ddleProject? {
    guard let previous = undo.popLast() else { return nil }
    redo.append(current)
    return previous
  }

  public mutating func popRedo(current: H3ddleProject) -> H3ddleProject? {
    guard let next = redo.popLast() else { return nil }
    undo.append(current)
    return next
  }
}
