import AppKit
import CoreText
import H3ddleCore
import H3ddleMedia

enum FontCatalog {
  static var families: [String] {
    NSFontManager.shared.availableFontFamilies.sorted { lhs, rhs in
      lhs.localizedCaseInsensitiveCompare(rhs) == .orderedAscending
    }
  }

  static func postScriptName(for style: TextStyle) -> String? {
    let font = FontResolver.font(for: style)
    return CTFontCopyPostScriptName(font) as String
  }

  static func resolved(_ style: TextStyle) -> TextStyle {
    var next = style
    next.fontPostScriptName = postScriptName(for: next)
    return next
  }
}
