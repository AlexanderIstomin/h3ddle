import AppKit

enum FontCatalog {
  static var families: [String] {
    NSFontManager.shared.availableFontFamilies.sorted { lhs, rhs in
      lhs.localizedCaseInsensitiveCompare(rhs) == .orderedAscending
    }
  }
}
