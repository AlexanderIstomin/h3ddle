import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct EffectLaneItem: Identifiable, Equatable {
  var clipID: UUID
  var startTime: TimeInterval
  var duration: TimeInterval
  var effect: VisualEffectInstance
  var clipEnabled: Bool

  var id: UUID { effect.id }
}

struct TimelineEffectPill: View {
  var title: String
  var tint: Color
  var isSelected: Bool
  var isEnabled: Bool
  var width: CGFloat

  var body: some View {
    ZStack(alignment: .leading) {
      TimelineEffectHatch(tint: tint)
      Text(title)
        .font(.system(size: 8.5, weight: .bold, design: .monospaced))
        .foregroundStyle(H3Color.textPrimary)
        .lineLimit(1)
        .padding(.horizontal, 6)
    }
    .frame(width: width, height: TimelineChrome.effectLaneHeight - 6)
    .background(tint.opacity(0.2))
    .overlay {
      RoundedRectangle(cornerRadius: 5, style: .continuous)
        .stroke(isSelected ? tint : tint.opacity(0.5), lineWidth: isSelected ? 1.5 : 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
    .shadow(color: isSelected ? tint.opacity(0.45) : .clear, radius: 6)
    .saturation(isEnabled ? 1 : 0.25)
    .opacity(isEnabled ? 1 : 0.55)
    .accessibilityIdentifier("fx-pill")
  }
}

private struct TimelineEffectHatch: View {
  var tint: Color

  var body: some View {
    Canvas { context, size in
      var x: CGFloat = -size.height
      while x < size.width + size.height {
        var path = Path()
        path.move(to: CGPoint(x: x, y: size.height))
        path.addLine(to: CGPoint(x: x + size.height, y: 0))
        context.stroke(path, with: .color(tint.opacity(0.16)), lineWidth: 5)
        x += 10
      }
    }
  }
}

extension VisualEffectKind {
  var swatch: Color {
    switch self {
    case .colorGrade: Color(red: 224 / 255, green: 82 / 255, blue: 28 / 255)
    case .vignette: Color(red: 142 / 255, green: 92 / 255, blue: 196 / 255)
    case .filmGrain: Color(red: 160 / 255, green: 164 / 255, blue: 172 / 255)
    case .sharpen: Color(red: 214 / 255, green: 176 / 255, blue: 72 / 255)
    case .blur: Color(red: 91 / 255, green: 134 / 255, blue: 201 / 255)
    case .bloom: Color(red: 232 / 255, green: 126 / 255, blue: 64 / 255)
    case .chromaKey: Color(red: 70 / 255, green: 168 / 255, blue: 131 / 255)
    }
  }
}
