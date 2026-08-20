import CoreText
import Foundation
import H3ddleCore
import Testing

@testable import H3ddleMedia

@Suite("Font resolver")
struct FontResolverTests {
  @Test("Italic requests a slanted face")
  func italicSlants() {
    let roman = FontResolver.font(for: TextStyle(italic: false, fontSize: 24))
    let italic = FontResolver.font(for: TextStyle(italic: true, fontSize: 24))
    let romanSlant =
      (CTFontCopyTraits(roman) as NSDictionary)[kCTFontSlantTrait] as? NSNumber
    let italicSlant =
      (CTFontCopyTraits(italic) as NSDictionary)[kCTFontSlantTrait] as? NSNumber
    let romanValue = romanSlant?.doubleValue ?? 0
    let italicValue = italicSlant?.doubleValue ?? 0
    let romanSymbolic = CTFontGetSymbolicTraits(roman).contains(.traitItalic)
    let italicSymbolic = CTFontGetSymbolicTraits(italic).contains(.traitItalic)
    #expect(italicValue > romanValue || italicSymbolic && !romanSymbolic)
  }
}
