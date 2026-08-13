import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct TimeRulerView: View {
  var duration: TimeInterval
  var metrics: TimelineMetrics
  var onSeek: (TimeInterval) -> Void

  var body: some View {
    Canvas { context, size in
      let pps = Double(metrics.pointsPerSecond)
      let minorSeconds = TimelineRuler.minorInterval(pointsPerSecond: pps)
      let minorSpan = CGFloat(minorSeconds) * metrics.pointsPerSecond

      if TimelineRuler.drawsMinorTicks(pointsPerSecond: pps), minorSpan > 0 {
        var x: CGFloat = 0
        while x <= size.width + 0.5 {
          var path = Path()
          path.move(to: CGPoint(x: x, y: size.height))
          path.addLine(to: CGPoint(x: x, y: size.height - 6))
          context.stroke(path, with: .color(H3Color.tickMinor), lineWidth: 1)
          x += minorSpan
        }
      }

      for tick in TimelineRuler.ticks(duration: duration, pointsPerSecond: pps) where tick.isMajor {
        let x = metrics.x(for: tick.time)
        var path = Path()
        path.move(to: CGPoint(x: x, y: size.height))
        path.addLine(to: CGPoint(x: x, y: size.height - 11))
        context.stroke(path, with: .color(H3Color.tickMajor), lineWidth: 1)
        if !tick.label.isEmpty {
          context.draw(
            Text(tick.label)
              .font(.system(size: 9, weight: .regular, design: .monospaced))
              .tracking(0.36)
              .foregroundColor(H3Color.textSecondary),
            at: CGPoint(x: x + 5, y: 5),
            anchor: .topLeading
          )
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
