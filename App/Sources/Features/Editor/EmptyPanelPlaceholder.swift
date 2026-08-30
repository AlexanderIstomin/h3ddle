import H3ddleDesignSystem
import SwiftUI

struct EmptyPanelPlaceholder<Graphic: View>: View {
  var title: String
  var detail: String
  var actionTitle: String?
  var actionIdentifier: String?
  var action: (() -> Void)?
  @ViewBuilder var graphic: Graphic

  init(
    title: String,
    detail: String,
    actionTitle: String? = nil,
    actionIdentifier: String? = nil,
    action: (() -> Void)? = nil,
    @ViewBuilder graphic: () -> Graphic
  ) {
    self.title = title
    self.detail = detail
    self.actionTitle = actionTitle
    self.actionIdentifier = actionIdentifier
    self.action = action
    self.graphic = graphic()
  }

  var body: some View {
    VStack(spacing: 16) {
      graphic
      VStack(spacing: 6) {
        Text(title)
          .font(.system(size: 13, weight: .semibold))
        Text(detail)
          .font(.system(size: 11.5))
          .foregroundStyle(H3Color.textSecondary)
          .multilineTextAlignment(.center)
          .fixedSize(horizontal: false, vertical: true)
      }
      if let actionTitle, let action {
        Button(actionTitle, action: action)
          .buttonStyle(H3QuietButtonStyle())
          .accessibilityIdentifier(actionIdentifier ?? "empty-panel-action")
      }
    }
    .frame(maxWidth: 280)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .padding(28)
  }
}

struct TransitionEmptyGraphic: View {
  var body: some View {
    HStack(spacing: 8) {
      clipStub
      ZStack {
        Circle()
          .fill(H3Color.accent.opacity(0.16))
          .frame(width: 34, height: 34)
        Image(systemName: "plus")
          .font(.system(size: 14, weight: .bold))
          .foregroundStyle(H3Color.accent)
      }
      clipStub
    }
    .accessibilityHidden(true)
  }

  private var clipStub: some View {
    RoundedRectangle(cornerRadius: 6, style: .continuous)
      .fill(H3Color.controlFill)
      .overlay {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
          .strokeBorder(H3Color.line, style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
      }
      .frame(width: 52, height: 34)
  }
}

struct EmptyPanelGlyph: View {
  var systemName: String

  var body: some View {
    ZStack {
      Circle()
        .fill(H3Color.controlFill)
        .frame(width: 56, height: 56)
      Image(systemName: systemName)
        .font(.system(size: 22, weight: .medium))
        .foregroundStyle(H3Color.textSecondary.opacity(0.9))
    }
    .accessibilityHidden(true)
  }
}
