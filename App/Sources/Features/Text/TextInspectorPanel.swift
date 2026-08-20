import AppKit
import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct TextInspectorPanel: View {
  @Bindable var model: AppModel
  @State private var contentDraft = ""
  @State private var familyQuery = ""
  @State private var showsAdvanced = false
  @State private var didCheckpointContent = false

  private var selected: TextItem? {
    guard case .text(let id) = model.selectedTimelineItem else { return nil }
    return model.project.timeline.textItems.first { $0.id == id }
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
      Divider().overlay(H3Color.line)
      if let item = selected {
        ScrollView {
          VStack(alignment: .leading, spacing: 16) {
            contentEditor
            familyPicker
            weightAndItalic(item)
            alignmentRow(item)
            fillRow(item)
            DisclosureGroup("Advanced", isExpanded: $showsAdvanced) {
              advanced(item)
            }
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(H3Color.textSecondary)
          }
          .padding(14)
        }
        .onAppear { contentDraft = item.text }
        .onChange(of: item.id) { _, _ in
          contentDraft = item.text
          didCheckpointContent = false
        }
      } else {
        emptyState
      }
    }
    .frame(width: 320)
    .background(H3Color.surface)
    .overlay(alignment: .trailing) {
      Rectangle().fill(H3Color.line).frame(width: 1)
    }
    .accessibilityIdentifier("text-panel")
  }

  private var header: some View {
    HStack(spacing: 8) {
      Text("Text")
        .font(.system(size: 13, weight: .semibold))
      Spacer()
      Text(headerTitle)
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
        .lineLimit(1)
      Button {
        model.closeTextPanel()
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
      .help("Close Text")
      .accessibilityIdentifier("text-close")
    }
    .padding(.horizontal, 12)
    .frame(height: 44)
  }

  private var headerTitle: String {
    let line = selected?.text.split(whereSeparator: \.isNewline).first.map(String.init) ?? "No clip"
    return line.isEmpty ? "Text" : line
  }

  private var emptyState: some View {
    VStack(spacing: 10) {
      Image(systemName: "textformat")
        .font(.system(size: 20, weight: .medium))
        .foregroundStyle(H3Color.accent)
      Text("Select a T1 clip to edit")
        .font(.system(size: 12.5))
        .foregroundStyle(H3Color.textSecondary)
      Button("Add text") {
        model.insertTextAtPlayhead(opensInspector: true)
      }
      .buttonStyle(H3QuietButtonStyle())
      .accessibilityIdentifier("text-panel-add")
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .padding(24)
  }

  private var contentEditor: some View {
    VStack(alignment: .leading, spacing: 6) {
      Text("Content")
        .font(.system(size: 10, weight: .semibold))
        .foregroundStyle(H3Color.textSecondary)
      TextEditor(text: $contentDraft)
        .font(.system(size: 13))
        .scrollContentBackground(.hidden)
        .padding(8)
        .frame(minHeight: 84)
        .background(H3Color.chrome)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
          RoundedRectangle(cornerRadius: 8, style: .continuous)
            .stroke(H3Color.line, lineWidth: 1)
        }
        .onChange(of: contentDraft) { _, next in
          guard let item = selected else { return }
          if !didCheckpointContent {
            model.setTextContent(item.id, next, registersUndo: true)
            didCheckpointContent = true
          } else {
            model.setTextContent(item.id, next, registersUndo: false)
          }
        }
        .accessibilityIdentifier("text-content")
    }
  }

  private var familyPicker: some View {
    VStack(alignment: .leading, spacing: 6) {
      Text("Family")
        .font(.system(size: 10, weight: .semibold))
        .foregroundStyle(H3Color.textSecondary)
      TextField("Search fonts", text: $familyQuery)
        .textFieldStyle(.plain)
        .font(.system(size: 11, design: .monospaced))
        .padding(.horizontal, 10)
        .frame(height: 30)
        .background(H3Color.chrome)
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 2) {
          ForEach(filteredFamilies, id: \.self) { family in
            Button {
              applyStyle { style in
                style.fontFamily = family
                style.fontPostScriptName = nil
              }
            } label: {
              Text(family)
                .font(.custom(family, size: 12))
                .foregroundStyle(H3Color.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(
                  RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(selected?.style.fontFamily == family ? H3Color.controlHover : Color.clear)
                )
            }
            .buttonStyle(.plain)
          }
        }
      }
      .frame(height: 120)
    }
  }

  private var filteredFamilies: [String] {
    let all = FontCatalog.families
    let query = familyQuery.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty else { return all }
    return all.filter { $0.localizedCaseInsensitiveContains(query) }
  }

  private func weightAndItalic(_ item: TextItem) -> some View {
    HStack(spacing: 8) {
      Picker(
        "Weight",
        selection: Binding(
          get: { item.style.fontWeight },
          set: { weight in
            applyStyle { $0.fontWeight = weight }
          }
        )
      ) {
        ForEach([100, 200, 300, 400, 500, 600, 700, 800, 900], id: \.self) { weight in
          Text(weightLabel(weight)).tag(weight)
        }
      }
      .labelsHidden()
      Toggle(
        "Italic",
        isOn: Binding(
          get: { item.style.italic },
          set: { italic in applyStyle { $0.italic = italic } }
        )
      )
      .toggleStyle(.checkbox)
    }
  }

  private func alignmentRow(_ item: TextItem) -> some View {
    HStack(spacing: 6) {
      alignmentButton(item, H3ddleCore.TextAlignment.leading, "text.alignleft")
      alignmentButton(item, H3ddleCore.TextAlignment.center, "text.aligncenter")
      alignmentButton(item, H3ddleCore.TextAlignment.trailing, "text.alignright")
      Spacer()
    }
  }

  private func alignmentButton(
    _ item: TextItem,
    _ alignment: H3ddleCore.TextAlignment,
    _ symbol: String
  ) -> some View {
    Button {
      applyStyle { $0.alignment = alignment }
    } label: {
      Image(systemName: symbol)
        .font(.system(size: 13, weight: .semibold))
        .frame(width: 32, height: 32)
        .background(
          item.style.alignment == alignment ? H3Color.accent.opacity(0.18) : H3Color.controlFill
        )
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        .frame(width: 44, height: 44)
        .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .foregroundStyle(H3Color.textPrimary)
  }

  private func fillRow(_ item: TextItem) -> some View {
    HStack(spacing: 10) {
      ColorPicker(
        "Fill",
        selection: Binding(
          get: { Color(textColor: item.style.fill) },
          set: { color in
            applyStyle { $0.fill = TextColor(color) }
          }
        ),
        supportsOpacity: true
      )
      .labelsHidden()
      Text("Fill")
        .font(.system(size: 12))
        .foregroundStyle(H3Color.textSecondary)
      Spacer()
    }
  }

  private func advanced(_ item: TextItem) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      Toggle(
        "Wrap",
        isOn: Binding(
          get: { item.style.wrap == .wrap },
          set: { wrap in
            applyStyle { style in
              style.wrap = wrap ? .wrap : .none
              if wrap, style.boxWidth == nil {
                style.boxWidth = 0.8 * Double(model.project.settings.width)
              }
            }
          }
        )
      )
      labeledSlider("Font size", value: item.style.fontSize, range: 8...240) { size in
        applyStyle { $0.fontSize = size }
      }
      labeledSlider("Line height", value: item.style.lineHeight, range: 0.8...2.4) { value in
        applyStyle { $0.lineHeight = value }
      }
      labeledSlider("Tracking", value: item.style.letterSpacing, range: -8...24) { value in
        applyStyle { $0.letterSpacing = value }
      }
      labeledSlider("Stroke", value: item.style.strokeWidth, range: 0...12) { value in
        applyStyle { $0.strokeWidth = value }
      }
      ColorPicker(
        "Stroke color",
        selection: Binding(
          get: { Color(textColor: item.style.strokeColor) },
          set: { color in applyStyle { $0.strokeColor = TextColor(color) } }
        ),
        supportsOpacity: true
      )
      labeledSlider("Shadow blur", value: item.style.shadowBlur, range: 0...24) { value in
        applyStyle { $0.shadowBlur = value }
      }
      ColorPicker(
        "Background",
        selection: Binding(
          get: { Color(textColor: item.style.backgroundColor) },
          set: { color in applyStyle { $0.backgroundColor = TextColor(color) } }
        ),
        supportsOpacity: true
      )
      labeledSlider("Padding", value: item.style.backgroundPadding, range: 0...48) { value in
        applyStyle { $0.backgroundPadding = value }
      }
      labeledSlider(
        "Corner",
        value: item.style.backgroundCornerRadius,
        range: 0...48
      ) { value in
        applyStyle { $0.backgroundCornerRadius = value }
      }
    }
    .padding(.top, 8)
  }

  private func labeledSlider(
    _ title: String,
    value: Double,
    range: ClosedRange<Double>,
    onChange: @escaping (Double) -> Void
  ) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack {
        Text(title)
          .font(.system(size: 11))
          .foregroundStyle(H3Color.textSecondary)
        Spacer()
        Text(String(format: "%.1f", value))
          .font(.system(size: 10, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
      }
      Slider(
        value: Binding(
          get: { value },
          set: onChange
        ),
        in: range
      )
    }
  }

  private func applyStyle(_ mutate: (inout TextStyle) -> Void) {
    guard var item = selected else { return }
    mutate(&item.style)
    model.setTextStyle(item.id, FontCatalog.resolved(item.style))
  }

  private func weightLabel(_ weight: Int) -> String {
    switch weight {
    case 100: "Thin"
    case 200: "Ultralight"
    case 300: "Light"
    case 400: "Regular"
    case 500: "Medium"
    case 600: "Semibold"
    case 700: "Bold"
    case 800: "Heavy"
    case 900: "Black"
    default: "\(weight)"
    }
  }
}

extension Color {
  fileprivate init(textColor: TextColor) {
    self.init(
      .sRGB,
      red: textColor.r,
      green: textColor.g,
      blue: textColor.b,
      opacity: textColor.a
    )
  }
}

extension TextColor {
  fileprivate init(_ color: Color) {
    let resolved = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
    var r: CGFloat = 1
    var g: CGFloat = 1
    var b: CGFloat = 1
    var a: CGFloat = 1
    resolved.getRed(&r, green: &g, blue: &b, alpha: &a)
    self.init(r: Double(r), g: Double(g), b: Double(b), a: Double(a))
  }
}
