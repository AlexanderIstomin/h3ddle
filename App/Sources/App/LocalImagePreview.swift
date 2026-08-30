import CoreGraphics
import Foundation
import ImageIO
import SwiftUI

struct LocalImagePreview<FailureContent: View>: View {
  let url: URL
  var contentMode: ContentMode = .fit
  let failureContent: () -> FailureContent

  @State private var image: CGImage?
  @State private var loadFailed = false

  init(
    url: URL,
    contentMode: ContentMode = .fit,
    @ViewBuilder failureContent: @escaping () -> FailureContent
  ) {
    self.url = url
    self.contentMode = contentMode
    self.failureContent = failureContent
  }

  var body: some View {
    Group {
      if let image {
        Image(decorative: image, scale: 1)
          .resizable()
          .interpolation(.high)
          .aspectRatio(contentMode: contentMode)
      } else if loadFailed {
        failureContent()
      } else {
        ProgressView()
          .controlSize(.small)
      }
    }
    .task(id: url) {
      image = nil
      loadFailed = false
      let decoded = await LocalImageDecoder.decodeInBackground(url)
      guard !Task.isCancelled else { return }
      image = decoded
      loadFailed = decoded == nil
    }
  }
}

private enum LocalImageDecoder {
  nonisolated static func decodeInBackground(_ url: URL) async -> CGImage? {
    await Task.detached(priority: .utility) {
      decode(url)
    }.value
  }

  nonisolated static func decode(_ url: URL) -> CGImage? {
    autoreleasepool {
      let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
      guard let source = CGImageSourceCreateWithURL(url as CFURL, sourceOptions) else {
        return nil
      }
      let imageOptions = [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
      return CGImageSourceCreateImageAtIndex(source, 0, imageOptions)
    }
  }
}
