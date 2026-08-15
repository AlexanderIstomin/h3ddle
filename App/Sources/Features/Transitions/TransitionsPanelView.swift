import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct TransitionsPanelView: View {
  @Bindable var model: AppModel

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
      Divider().overlay(H3Color.line)
      if let host, host.transition != nil, !model.browsesTransitionCatalog {
        TransitionSettingsView(model: model, host: host)
      } else {
        catalog
      }
    }
    .frame(width: 320)
    .background(H3Color.surface)
    .overlay(alignment: .trailing) {
      Rectangle().fill(H3Color.line).frame(width: 1)
    }
    .accessibilityIdentifier("transitions-panel")
  }

  private var host: VisualItem? { model.transitionHostClip }

  private var header: some View {
    HStack(spacing: 8) {
      Text("Transitions")
        .font(.system(size: 13, weight: .semibold))
      Spacer()
      Text(host.flatMap { model.project.asset(id: $0.assetID)?.displayName } ?? "No cut")
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
        .lineLimit(1)
      Button {
        model.closeTransitionsPanel()
      } label: {
        Image(systemName: "xmark")
          .font(.system(size: 10, weight: .semibold))
          .frame(width: 26, height: 26)
          .background(H3Color.controlFill)
          .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
      }
      .buttonStyle(.plain)
      .foregroundStyle(H3Color.textSecondary)
      .help("Close Transitions")
      .accessibilityIdentifier("transitions-close")
    }
    .padding(.horizontal, 12)
    .frame(height: 44)
  }

  @ViewBuilder
  private var catalog: some View {
    if host == nil {
      VStack(spacing: 10) {
        Image(systemName: "plus")
          .font(.system(size: 20, weight: .medium))
          .foregroundStyle(H3Color.accent)
        Text("Select a cut between two clips to add a transition")
          .font(.system(size: 12.5))
          .foregroundStyle(H3Color.textSecondary)
          .multilineTextAlignment(.center)
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .padding(24)
    } else {
      ScrollView {
        VStack(alignment: .leading, spacing: 13) {
          Text("OVERLAP")
            .font(.system(size: 9, weight: .bold, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary)
          Text("A transition pulls the incoming clip over the outgoing tail.")
            .font(.system(size: 11.5))
            .foregroundStyle(H3Color.textSecondary)
          LazyVGrid(
            columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)],
            spacing: 10
          ) {
            ForEach(VisualTransitionKind.allCases) { kind in
              TransitionCatalogCard(kind: kind, isSelected: false) {
                model.applyVisualTransition(kind)
              }
            }
          }
        }
        .padding(12)
      }
    }
  }
}

private struct TransitionSettingsView: View {
  @Bindable var model: AppModel
  var host: VisualItem

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        Button {
          model.browseTransitionCatalog()
        } label: {
          Label("Catalog", systemImage: "chevron.left")
            .font(.system(size: 11, weight: .semibold))
        }
        .buttonStyle(.plain)
        .foregroundStyle(H3Color.textSecondary)
        .accessibilityIdentifier("transitions-back")

        Text(host.transition?.kind.label ?? "Transition")
          .font(.system(size: 15, weight: .semibold))

        LazyVGrid(
          columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)],
          spacing: 10
        ) {
          ForEach(VisualTransitionKind.allCases) { kind in
            TransitionCatalogCard(kind: kind, isSelected: host.transition?.kind == kind) {
              model.applyVisualTransition(kind)
            }
          }
        }

        VStack(alignment: .leading, spacing: 6) {
          HStack {
            Text("DURATION")
              .font(.system(size: 9, weight: .bold, design: .monospaced))
              .foregroundStyle(H3Color.textSecondary)
            Spacer()
            Text(String(format: "%.2fs", host.transition?.duration ?? 0))
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(H3Color.textSecondary)
          }
          Slider(
            value: durationBinding,
            in: 0.05...max(0.05, model.project.timeline.maximumVisualTransitionDuration(of: host.id))
          )
          .accessibilityIdentifier("transition-duration")
        }

        Button(role: .destructive) {
          model.removeVisualTransition(host.id)
        } label: {
          Text("Remove")
            .frame(maxWidth: .infinity)
            .frame(height: 32)
        }
        .buttonStyle(H3QuietButtonStyle())
        .foregroundStyle(H3Color.danger)
        .accessibilityIdentifier("transition-remove")
      }
      .padding(16)
    }
  }

  private var durationBinding: Binding<Double> {
    Binding(
      get: { host.transition?.duration ?? VisualTransitionMath.defaultDuration },
      set: { model.setVisualTransitionDuration(host.id, duration: $0) }
    )
  }
}

private struct TransitionCatalogCard: View {
  var kind: VisualTransitionKind
  var isSelected: Bool
  var action: () -> Void

  var body: some View {
    Button(action: action) {
      VStack(alignment: .leading, spacing: 0) {
        TransitionThumb(kind: kind)
          .aspectRatio(16 / 10, contentMode: .fit)
        HStack(spacing: 6) {
          Image(systemName: kind.symbol)
            .font(.system(size: 11, weight: .semibold))
          Text(kind.label)
            .font(.system(size: 11, weight: .semibold))
            .lineLimit(1)
          Spacer(minLength: 0)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 8)
      }
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 7, style: .continuous)
          .stroke(isSelected ? H3Color.accent : H3Color.line, lineWidth: isSelected ? 1.5 : 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
    }
    .buttonStyle(.plain)
    .accessibilityIdentifier("transition-card-\(kind.rawValue)")
    .accessibilityAddTraits(isSelected ? .isSelected : [])
  }
}

private struct TransitionThumb: View {
  var kind: VisualTransitionKind

  var body: some View {
    Canvas { context, size in
      let mid = size.width * (kind == .wipe ? 0.55 : 0.5)
      var outgoing = Path()
      outgoing.addRect(CGRect(x: 0, y: 0, width: mid + 8, height: size.height))
      context.fill(outgoing, with: .color(Color(red: 0.22, green: 0.28, blue: 0.38)))
      var incoming = Path()
      incoming.addRect(CGRect(x: mid - 8, y: 0, width: size.width - mid + 8, height: size.height))
      context.fill(incoming, with: .color(H3Color.accent.opacity(kind == .fade ? 0.35 : 0.7)))
      var slash = Path()
      slash.move(to: CGPoint(x: mid - 10, y: size.height))
      slash.addLine(to: CGPoint(x: mid + 10, y: 0))
      context.stroke(slash, with: .color(.white.opacity(0.55)), lineWidth: 1.5)
    }
  }
}
