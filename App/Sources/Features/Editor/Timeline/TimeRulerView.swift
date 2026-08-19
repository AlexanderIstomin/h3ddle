import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct TimeRulerView: View {
  var duration: TimeInterval
  var metrics: TimelineMetrics
  var visibleX: CGFloat
  var visibleWidth: CGFloat
  var onSeek: (TimeInterval) -> Void

  var body: some View {
    Canvas { context, size in
      let pps = Double(metrics.pointsPerSecond)
      let overscan = CGFloat(TimelineRuler.trailingLabelPadding)
      let visibleRange = TimelineRuler.visibleTimeRange(
        scrollOffset: Double(visibleX),
        viewportWidth: Double(visibleWidth),
        pointsPerSecond: pps,
        overscanPoints: Double(overscan)
      )

      for tick in TimelineRuler.ticks(
        duration: duration,
        pointsPerSecond: pps,
        visibleRange: visibleRange
      ) {
        let x = metrics.x(for: tick.time)
        if tick.isMajor {
          var path = Path()
          path.move(to: CGPoint(x: x, y: size.height))
          path.addLine(to: CGPoint(x: x, y: size.height - 11))
          context.stroke(path, with: .color(H3Color.tickMajor), lineWidth: 1)
          context.draw(
            Text(tick.label)
              .font(.system(size: 9, weight: .regular, design: .monospaced))
              .tracking(0.36)
              .foregroundColor(H3Color.textSecondary),
            at: CGPoint(x: x + 5, y: 5),
            anchor: .topLeading
          )
        } else {
          var path = Path()
          path.move(to: CGPoint(x: x, y: size.height))
          path.addLine(to: CGPoint(x: x, y: size.height - 6))
          context.stroke(path, with: .color(H3Color.tickMinor), lineWidth: 1)
        }
      }
    }
    .frame(height: TimelineChrome.rulerHeight)
    .background(H3Color.chrome)
    .overlay(alignment: .bottom) {
      Rectangle().fill(H3Color.line).frame(height: 1)
    }
    .contentShape(Rectangle())
    .gesture(
      DragGesture(minimumDistance: 0)
        .onChanged { value in
          onSeek(metrics.time(for: value.location.x))
        }
    )
  }
}
