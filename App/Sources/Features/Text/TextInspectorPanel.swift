import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct TextInspectorPanel: View {
  @Bindable var model: AppModel
  var embedded = false
  @State private var contentDraft = ""
  @State private var familyQuery = ""
  @State private var showsFamilyPicker = false
  @State private var showsAdvanced = false
  @State private var didCheckpointContent = false

  private var selected: TextItem? {
    guard case .text(let id) = model.selectedTimelineItem else { return nil }
    return model.project.timeline.textItems.first { $0.id == id }
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      if !embedded {
        header
        Divider().overlay(H3Color.line)
      }
      if let item = selected {
        ScrollView {
          VStack(alignment: .leading, spacing: 20) {
            contentEditor
            fontRow(item)
            alignmentRow(item)
            fillRow(item)
            H3Accordion("Advanced", isExpanded: $showsAdvanced) {
              advanced(item)
            }
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
    .frame(width: embedded ? nil : 320)
    .background(embedded ? Color.clear : H3Color.surface)
    .overlay(alignment: .trailing) {
      if !embedded {
        Rectangle().fill(H3Color.line).frame(width: 1)
      }
    }
    .accessibilityIdentifier("text-panel")
  }

  private var header: some View {
    HStack(spacing: 8) {
      Text("Text")
        .font(.system(size: 13, weight: .semibold))
      Spacer()
      if !embedded {
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
    }
    .padding(.horizontal, 12)
    .frame(height: 44)
  }

  private var emptyState: some View {
    EmptyPanelPlaceholder(
      title: "No text selected",
      detail: "Select a T1 clip to edit its style, or add a title at the playhead.",
      actionTitle: "Add text",
      actionIdentifier: "text-panel-add",
      action: { model.insertTextAtPlayhead(opensInspector: true) }
    ) {
      EmptyPanelGlyph(systemName: "textformat")
    }
  }

  private var contentEditor: some View {
    VStack(alignment: .leading, spacing: 6) {
      fieldLabel("Content")
      TextEditor(text: $contentDraft)
        .font(.system(size: 13))
        .scrollContentBackground(.hidden)
        .padding(8)
        .frame(minHeight: 72)
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

  private func fontRow(_ item: TextItem) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      fieldLabel("Font")
      familyButton(item)
      HStack(spacing: 8) {
        weightMenu(item)
        italicToggle(item)
      }
    }
  }

  private func familyButton(_ item: TextItem) -> some View {
    Button {
      familyQuery = ""
      showsFamilyPicker.toggle()
    } label: {
      HStack(spacing: 8) {
        Text(displayFamily(item.style.fontFamily))
          .font(.custom(item.style.fontFamily, size: 13))
          .foregroundStyle(H3Color.textPrimary)
          .lineLimit(1)
        Spacer(minLength: 0)
        Image(systemName: "chevron.up.chevron.down")
          .font(.system(size: 9, weight: .semibold))
          .foregroundStyle(H3Color.textSecondary)
      }
      .padding(.horizontal, 11)
      .frame(height: 34)
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
    .buttonStyle(.plain)
    .popover(isPresented: $showsFamilyPicker, arrowEdge: .trailing) {
      familyPopover(item)
    }
    .accessibilityIdentifier("text-font-family")
  }

  private func familyPopover(_ item: TextItem) -> some View {
    VStack(spacing: 0) {
      HStack(spacing: 8) {
        Image(systemName: "magnifyingglass")
          .foregroundStyle(H3Color.textSecondary)
        TextField("Search fonts", text: $familyQuery)
          .textFieldStyle(.plain)
          .font(.system(size: 12))
      }
      .padding(.horizontal, 10)
      .frame(height: 32)
      .background(H3Color.chrome)
      Divider().overlay(H3Color.line)
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 1) {
          ForEach(filteredFamilies, id: \.self) { family in
            Button {
              applyStyle { style in
                style.fontFamily = family
                style.fontPostScriptName = nil
              }
              showsFamilyPicker = false
            } label: {
              Text(family)
                .font(.custom(family, size: 13))
                .foregroundStyle(H3Color.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(
                  item.style.fontFamily == family ? H3Color.accent.opacity(0.16) : Color.clear
                )
            }
            .buttonStyle(.plain)
          }
        }
      }
    }
    .frame(width: 260, height: 320)
    .background(H3Color.surface)
  }

  private var filteredFamilies: [String] {
    let all = FontCatalog.families
    let query = familyQuery.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty else { return all }
    return all.filter { $0.localizedCaseInsensitiveContains(query) }
  }

  private func weightMenu(_ item: TextItem) -> some View {
    Menu {
      ForEach([100, 200, 300, 400, 500, 600, 700, 800, 900], id: \.self) { weight in
        Button(weightLabel(weight)) {
          applyStyle { $0.fontWeight = weight }
        }
      }
    } label: {
      HStack {
        Text(weightLabel(item.style.fontWeight))
          .font(.system(size: 12, weight: .medium))
          .foregroundStyle(H3Color.textPrimary)
        Spacer(minLength: 0)
        Image(systemName: "chevron.up.chevron.down")
          .font(.system(size: 9, weight: .semibold))
          .foregroundStyle(H3Color.textSecondary)
      }
      .padding(.horizontal, 11)
      .frame(height: 34)
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
    .buttonStyle(.plain)
    .accessibilityIdentifier("text-font-weight")
  }

  private func italicToggle(_ item: TextItem) -> some View {
    Button {
      applyStyle { $0.italic.toggle() }
    } label: {
      Text("I")
        .font(.system(size: 14, weight: .semibold, design: .serif).italic())
        .frame(width: 34, height: 34)
        .foregroundStyle(item.style.italic ? H3Color.accent : H3Color.textPrimary)
        .background(item.style.italic ? H3Color.accent.opacity(0.16) : H3Color.chrome)
        .overlay {
          RoundedRectangle(cornerRadius: 8, style: .continuous)
            .stroke(item.style.italic ? H3Color.accent : H3Color.line, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
    .buttonStyle(.plain)
    .help("Italic")
    .accessibilityIdentifier("text-italic")
    .accessibilityAddTraits(item.style.italic ? [.isSelected] : [])
  }

  private func alignmentRow(_ item: TextItem) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      fieldLabel("Align")
      HStack(spacing: 0) {
        alignmentSegment(item, .leading, "text.alignleft")
        alignmentSegment(item, .center, "text.aligncenter")
        alignmentSegment(item, .trailing, "text.alignright")
      }
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
  }

  private func alignmentSegment(
    _ item: TextItem,
    _ alignment: H3ddleCore.TextAlignment,
    _ symbol: String
  ) -> some View {
    Button {
      applyStyle { $0.alignment = alignment }
    } label: {
      Image(systemName: symbol)
        .font(.system(size: 13, weight: .semibold))
        .frame(maxWidth: .infinity)
        .frame(height: 32)
        .foregroundStyle(
          item.style.alignment == alignment ? H3Color.textPrimary : H3Color.textSecondary
        )
        .background(
          item.style.alignment == alignment ? H3Color.accent.opacity(0.18) : Color.clear
        )
    }
    .buttonStyle(.plain)
    .accessibilityIdentifier("text-align-\(alignment.rawValue)")
  }

  private func fillRow(_ item: TextItem) -> some View {
    HStack(spacing: 10) {
      fieldLabel("Fill")
      Spacer()
      squareSwatch(item.style.fill) { fill in
        applyStyle { $0.fill = fill }
      }
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
      .toggleStyle(.switch)
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
      HStack {
        Text("Stroke color")
          .font(.system(size: 11))
          .foregroundStyle(H3Color.textSecondary)
        Spacer()
        squareSwatch(item.style.strokeColor) { color in
          applyStyle { $0.strokeColor = color }
        }
      }
      labeledSlider("Shadow blur", value: item.style.shadowBlur, range: 0...24) { value in
        applyStyle { $0.shadowBlur = value }
      }
      HStack {
        Text("Background")
          .font(.system(size: 11))
          .foregroundStyle(H3Color.textSecondary)
        Spacer()
        squareSwatch(item.style.backgroundColor) { color in
          applyStyle { $0.backgroundColor = color }
        }
      }
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
    .padding(.top, 4)
  }

  private func squareSwatch(_ color: TextColor, onChange: @escaping (TextColor) -> Void) -> some View
  {
    ZStack {
      ColorPicker(
        "",
        selection: Binding(
          get: { Color(textColor: color) },
          set: { onChange(TextColor($0)) }
        ),
        supportsOpacity: true
      )
      .labelsHidden()
      .frame(width: 22, height: 22)
      Group {
        if color.a < 0.04 {
          H3Checkerboard(cell: 4)
        } else {
          Color(textColor: color)
        }
      }
      .allowsHitTesting(false)
      RoundedRectangle(cornerRadius: 4, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
        .allowsHitTesting(false)
    }
    .frame(width: 22, height: 22)
    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    .accessibilityLabel("Color")
  }

  private func fieldLabel(_ title: String) -> some View {
    Text(title.uppercased())
      .font(.system(size: 9, weight: .bold, design: .monospaced))
      .tracking(0.6)
      .foregroundStyle(H3Color.textSecondary)
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
    model.setTextStyle(item.id, FontResolver.resolved(item.style))
  }

  private func displayFamily(_ family: String) -> String {
    if family == ".AppleSystemUIFont" { return "System" }
    return family
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
