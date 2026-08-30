import H3ddleDesignSystem
import SwiftUI

enum LeftRailMetrics {
  static let width: CGFloat = 64
  static let tabHeight: CGFloat = 52
  static let iconSize: CGFloat = 16
  static let labelSize: CGFloat = 9
  static let panelWidth: CGFloat = 384
}

struct LeftRailView: View {
  @Bindable var model: AppModel

  var body: some View {
    ScrollView(.vertical) {
      VStack(spacing: 2) {
        ForEach(EditorPanel.railTabs) { tab in
          Button {
            model.toggleRail(tab)
          } label: {
            VStack(spacing: 4) {
              Image(systemName: tab.railSymbol)
                .font(.system(size: LeftRailMetrics.iconSize, weight: .medium))
                .frame(height: 18)
              Text(tab.railLabel)
                .font(.system(size: LeftRailMetrics.labelSize, weight: .medium))
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .minimumScaleFactor(0.78)
            }
            .foregroundStyle(isActive(tab) ? H3Color.accent : H3Color.textSecondary)
            .frame(maxWidth: .infinity)
            .frame(minHeight: LeftRailMetrics.tabHeight)
            .background(isActive(tab) ? H3Color.controlFill : Color.white.opacity(0.001))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(alignment: .leading) {
              if isActive(tab) {
                RoundedRectangle(cornerRadius: 1, style: .continuous)
                  .fill(H3Color.accent)
                  .frame(width: 2)
                  .padding(.vertical, 8)
              }
            }
            .overlay(alignment: .topTrailing) {
              if tab == .queue, model.queuedGenerationCount > 0 {
                Text("\(model.queuedGenerationCount)")
                  .font(.system(size: 8, weight: .bold, design: .monospaced))
                  .foregroundStyle(.white)
                  .padding(.horizontal, 4)
                  .frame(height: 14)
                  .background(H3Color.accent)
                  .clipShape(Capsule())
                  .offset(x: -1, y: 4)
              }
            }
            .contentShape(Rectangle())
          }
          .buttonStyle(LeftRailTabButtonStyle())
          .help(tab.panelTitle)
          .accessibilityIdentifier(railIdentifier(tab))
          .accessibilityAddTraits(isActive(tab) ? .isSelected : [])
        }
      }
      .padding(.vertical, 6)
      .padding(.horizontal, 4)
    }
    .scrollIndicators(.hidden)
    .frame(width: LeftRailMetrics.width)
    .background(H3Color.chrome)
    .overlay(alignment: .trailing) {
      Rectangle().fill(H3Color.line).frame(width: 1)
    }
    .accessibilityIdentifier("left-rail")
  }

  private func isActive(_ tab: EditorPanel) -> Bool {
    model.openPanel == tab || model.openPanel?.parentPanel == tab
  }

  private func railIdentifier(_ tab: EditorPanel) -> String {
    switch tab {
    case .queue: "generation-queue-toggle"
    case .models: "model-status"
    default: "rail-tab-\(tab.rawValue)"
    }
  }
}

struct LeftRailPanel<Content: View>: View {
  var title: String
  var navigationHelp = "Collapse panel"
  var onClose: () -> Void
  @ViewBuilder var content: Content

  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 8) {
        Text(title)
          .font(.system(size: 13, weight: .semibold))
        Spacer()
        Button(action: onClose) {
          Image(systemName: "chevron.left")
            .font(.system(size: 11, weight: .semibold))
            .frame(width: 26, height: 26)
            .background(H3Color.controlFill)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        }
        .buttonStyle(.plain)
        .help(navigationHelp)
        .accessibilityIdentifier("left-panel-close")
      }
      .padding(.horizontal, 12)
      .frame(height: 44)
      Divider().overlay(H3Color.line)
      content
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
    .frame(width: LeftRailMetrics.panelWidth)
    .background(H3Color.surface)
    .overlay(alignment: .trailing) {
      Rectangle().fill(H3Color.line).frame(width: 1)
    }
    .shadow(color: .black.opacity(0.35), radius: 22, x: 8, y: 0)
    .accessibilityElement(children: .contain)
    .accessibilityIdentifier("left-panel")
  }
}

private struct LeftRailTabButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    configuration.label
      .opacity(configuration.isPressed ? 0.72 : 1)
      .contentShape(Rectangle())
  }
}
