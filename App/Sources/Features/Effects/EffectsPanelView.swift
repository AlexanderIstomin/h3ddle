import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct EffectsPanelView: View {
  @Bindable var model: AppModel
  var embedded = false
  @State private var query = ""
  @State private var category: VisualEffectCategory?

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      if !embedded {
        header
        Divider().overlay(H3Color.line)
      } else if let name = hostName {
        Text(name)
          .font(.system(size: 11))
          .foregroundStyle(H3Color.textSecondary)
          .lineLimit(1)
          .padding(.horizontal, 14)
          .padding(.top, 12)
          .padding(.bottom, 4)
      }
      if let effect = selectedEffect {
        EffectSettingsView(model: model, effect: effect)
      } else {
        catalog
      }
    }
    .frame(width: embedded ? nil : 320)
    .background(embedded ? Color.clear : H3Color.surface)
    .overlay(alignment: .trailing) {
      if !embedded {
        Rectangle().fill(H3Color.line).frame(width: 1)
      }
    }
    .accessibilityIdentifier("effects-panel")
  }

  private var hostName: String? {
    model.effectsHostClip.flatMap { model.project.asset(id: $0.assetID)?.displayName }
  }

  private var selectedEffect: VisualEffectInstance? {
    guard let host = model.effectsHostClip, let id = model.selectedEffectID else { return nil }
    return host.effects.first { $0.id == id }
  }

  private var header: some View {
    HStack(spacing: 8) {
      Text("Effects")
        .font(.system(size: 13, weight: .semibold))
      Spacer()
      Text(hostName ?? "No clip")
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
        .lineLimit(1)
      if !embedded {
      Button {
        model.showsEffectsPanel = false
        model.selectedEffectID = nil
      } label: {
        Image(systemName: "xmark")
          .font(.system(size: 10, weight: .semibold))
          .frame(width: 26, height: 26)
          .background(H3Color.controlFill)
          .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
          .frame(width: 44, height: 44)
          .contentShape(Rectangle())
      }
      .buttonStyle(.plain)
      .foregroundStyle(H3Color.textSecondary)
      .help("Close Effects")
      .accessibilityIdentifier("effects-close")
      }
    }
    .padding(.horizontal, 12)
    .frame(height: 44)
  }

  @ViewBuilder
  private var catalog: some View {
    if model.effectsHostClip == nil {
      EmptyPanelPlaceholder(
        title: "No clip selected",
        detail: "Select a visual clip on the timeline to add effects."
      ) {
        EmptyPanelGlyph(systemName: "sparkles")
      }
    } else {
      ScrollView {
        VStack(alignment: .leading, spacing: 13) {
          searchField
          categoryChips
          LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
            ForEach(filteredKinds) { kind in
              EffectCatalogCard(kind: kind) {
                model.addVisualEffect(kind)
              }
            }
          }
        }
        .padding(12)
      }
    }
  }

  private var searchField: some View {
    HStack(spacing: 8) {
      Image(systemName: "magnifyingglass")
        .foregroundStyle(H3Color.textSecondary)
      TextField("Search effects", text: $query)
        .textFieldStyle(.plain)
        .font(.system(size: 11, design: .monospaced))
      if !query.isEmpty {
        Button {
          query = ""
        } label: {
          Image(systemName: "xmark")
            .font(.system(size: 9, weight: .bold))
        }
        .buttonStyle(.plain)
        .foregroundStyle(H3Color.textSecondary)
      }
    }
    .padding(.horizontal, 10)
    .frame(height: 34)
    .background(H3Color.chrome)
    .overlay {
      RoundedRectangle(cornerRadius: 7, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
  }

  private var categoryChips: some View {
    ScrollView(.horizontal, showsIndicators: false) {
      HStack(spacing: 6) {
        chip(title: "All", selected: category == nil) { category = nil }
        ForEach(VisualEffectCategory.allCases) { item in
          chip(title: item.label, selected: category == item) {
            category = item
          }
        }
      }
    }
  }

  private func chip(title: String, selected: Bool, action: @escaping () -> Void) -> some View {
    Button(action: action) {
      Text(title.uppercased())
        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
        .tracking(0.5)
        .foregroundStyle(selected ? Color.white : H3Color.textSecondary)
        .padding(.horizontal, 11)
        .frame(height: 26)
        .background(selected ? H3Color.accent : Color.clear)
        .overlay {
          RoundedRectangle(cornerRadius: 6, style: .continuous)
            .stroke(selected ? H3Color.accent : H3Color.line, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
    .buttonStyle(.plain)
  }

  private var filteredKinds: [VisualEffectKind] {
    VisualEffectKind.allCases.filter { kind in
      if let category, kind.category != category { return false }
      if query.isEmpty { return true }
      return kind.label.localizedCaseInsensitiveContains(query)
    }
  }
}

private struct EffectCatalogCard: View {
  var kind: VisualEffectKind
  var action: () -> Void

  var body: some View {
    Button(action: action) {
      VStack(alignment: .leading, spacing: 0) {
        EffectThumb(kind: kind)
          .aspectRatio(16 / 10, contentMode: .fit)
        VStack(alignment: .leading, spacing: 2) {
          Text(kind.label)
            .font(.system(size: 11, weight: .semibold))
            .lineLimit(1)
          Text(kind.category.label.uppercased())
            .font(.system(size: 8.5, design: .monospaced))
            .tracking(0.6)
            .foregroundStyle(H3Color.textSecondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 8)
      }
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 7, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
    }
    .buttonStyle(.plain)
    .accessibilityIdentifier("effect-card-\(kind.rawValue)")
  }
}

private struct EffectThumb: View {
  var kind: VisualEffectKind

  var body: some View {
    ZStack {
      LinearGradient(
        colors: [Color(red: 0.15, green: 0.13, blue: 0.12), H3Color.canvas],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
      )
      Circle()
        .fill(H3Color.accent.opacity(kind == .blur || kind == .bloom ? 0.55 : 0.9))
        .frame(width: thumbSize, height: thumbSize)
        .blur(radius: kind == .vignette || kind == .blur ? 6 : 0)
        .saturation(kind == .colorGrade ? 1.4 : 1)
      if kind == .chromaKey {
        Circle()
          .fill(Color(red: 0.1, green: 0.75, blue: 0.25))
          .frame(width: 18, height: 18)
          .offset(x: 10, y: 6)
      }
    }
  }

  private var thumbSize: CGFloat {
    switch kind {
    case .vignette, .blur: 28
    case .bloom: 36
    default: 22
    }
  }
}

private struct EffectSettingsView: View {
  @Bindable var model: AppModel
  var effect: VisualEffectInstance

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        Button {
          model.selectedEffectID = nil
        } label: {
          Label("Catalog", systemImage: "chevron.left")
            .font(.system(size: 11, weight: .semibold))
        }
        .buttonStyle(.plain)
        .foregroundStyle(H3Color.textSecondary)
        .accessibilityIdentifier("effects-back")

        HStack {
          Text(effect.kind.label)
            .font(.system(size: 15, weight: .semibold))
          Spacer()
          Toggle("Enabled", isOn: enabledBinding)
            .toggleStyle(.switch)
            .labelsHidden()
            .accessibilityIdentifier("effect-enabled")
        }

        ForEach(VisualEffectCatalog.knobs(for: effect.kind)) { knob in
          VStack(alignment: .leading, spacing: 6) {
            HStack {
              Text(knob.label.uppercased())
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(H3Color.textSecondary)
              Spacer()
              Text(format(effect.value(knob.key, default: knob.range.lowerBound)))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(H3Color.textSecondary)
            }
            Slider(
              value: binding(for: knob),
              in: knob.range
            )
          }
        }

        Button(role: .destructive) {
          model.removeSelectedEffect()
        } label: {
          Text("Remove")
            .frame(maxWidth: .infinity)
            .frame(height: 32)
        }
        .buttonStyle(H3QuietButtonStyle())
        .foregroundStyle(H3Color.danger)
        .accessibilityIdentifier("effect-remove")
      }
      .padding(16)
    }
  }

  private var enabledBinding: Binding<Bool> {
    Binding(
      get: { effect.isEnabled },
      set: { value in
        var next = effect
        next.isEnabled = value
        model.updateVisualEffect(next)
      }
    )
  }

  private func binding(for knob: VisualEffectKnob) -> Binding<Double> {
    Binding(
      get: { effect.value(knob.key, default: knob.range.lowerBound) },
      set: { value in
        var next = effect
        next.parameters[knob.key] = min(max(value, knob.range.lowerBound), knob.range.upperBound)
        model.updateVisualEffect(next)
      }
    )
  }

  private func format(_ value: Double) -> String {
    String(format: abs(value) >= 10 ? "%.0f" : "%.2f", value)
  }
}
