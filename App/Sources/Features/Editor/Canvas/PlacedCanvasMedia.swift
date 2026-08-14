import H3ddleCore
import SwiftUI

struct PlacedCanvasMedia<Content: View>: View {
  var source: CGSize
  var transform: VisualCanvasTransform
  @ViewBuilder var content: () -> Content

  var body: some View {
    GeometryReader { proxy in
      let dest = CanvasLayout.destination(
        sourceWidth: Double(source.width),
        sourceHeight: Double(source.height),
        canvasWidth: Double(proxy.size.width),
        canvasHeight: Double(proxy.size.height),
        transform: transform
      )
      let draw = CanvasLayout.unrotatedSize(
        destination: dest,
        rotationTurns: transform.rotationTurns
      )
      content()
        .frame(width: draw.width, height: draw.height)
        .clipped()
        .rotationEffect(.degrees(transform.rotationDegrees))
        .frame(width: dest.width, height: dest.height)
        .position(x: dest.midX, y: dest.midY)
    }
    .clipped()
  }
}
