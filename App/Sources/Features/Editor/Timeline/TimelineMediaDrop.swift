import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct TimelineMediaDrop: ViewModifier {
  var lane: MediaImportLane
  var model: AppModel
  var accessibilityID: String
  var onLibraryAsset: ((AssetID, CGPoint) -> Bool)?

  @State private var isTargeted = false

  func body(content: Content) -> some View {
    content
      .overlay {
        if isTargeted {
          RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(H3Color.accent.opacity(0.14))
            .overlay {
              RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(H3Color.accent, lineWidth: 1.5)
            }
            .padding(2)
            .allowsHitTesting(false)
        }
      }
      .dropDestination(for: URL.self) { urls, _ in
        model.receiveDroppedFiles(urls, onto: lane)
      } isTargeted: { hovering in
        isTargeted = hovering && MediaImport.containsCompatible(Self.draggingFileURLs, onto: lane)
      }
      .dropDestination(for: String.self) { items, location in
        guard let raw = items.first, let uuid = UUID(uuidString: raw) else { return false }
        let id = AssetID(rawValue: uuid)
        guard let asset = model.project.asset(id: id) else { return false }
        guard lane.accepts(asset.kind) else { return false }
        if let onLibraryAsset {
          return onLibraryAsset(id, location)
        }
        model.insertLibraryAsset(id)
        return true
      }
      // The lane is a drop target that also contains controls of its own.
      // Naming the whole region merged it with the append button inside,
      // producing one element that answered to the lane's identifier and
      // the button's label — so the button could never be activated. The
      // name now rides on a marker of its own, which the drop tests can
      // still find while the controls keep their identities.
      .overlay {
        Color.clear
          .accessibilityElement()
          .accessibilityIdentifier(accessibilityID)
          .allowsHitTesting(false)
      }
  }

  private static var draggingFileURLs: [URL] {
    let pasteboard = NSPasteboard(name: .drag)
    let urls =
      pasteboard.readObjects(
        forClasses: [NSURL.self],
        options: [.urlReadingFileURLsOnly: true]
      ) as? [URL] ?? []
    return urls.map(\.standardizedFileURL)
  }
}

extension View {
  func timelineMediaDrop(
    lane: MediaImportLane,
    model: AppModel,
    accessibilityID: String,
    onLibraryAsset: ((AssetID, CGPoint) -> Bool)? = nil
  ) -> some View {
    modifier(
      TimelineMediaDrop(
        lane: lane,
        model: model,
        accessibilityID: accessibilityID,
        onLibraryAsset: onLibraryAsset
      )
    )
  }
}