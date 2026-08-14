import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct ExportControlsView: View {
  @Binding var settings: ProgramExportSettings
  let seed: ProgramExportSettings
  let programDuration: TimeInterval

  var body: some View {
    VStack(alignment: .leading, spacing: 18) {
      if settings.isCustom {
        customFields
      } else {
        lockedSummary
      }

      bitrateSlider

      if settings.isCustom {
        labeled("Audio") {
          chipRow(ProgramAudioQuality.allCases, selected: settings.audioQuality) { quality in
            settings.updateCustom { $0.audioBitrateKbps = quality.kilobitsPerSecond }
          }
        }
      }

      toggleRow(
        symbol: "waveform",
        title: "Normalize loudness",
        detail: "−14 LUFS",
        isOn: settings.normalizeLoudness
      ) {
        settings.setAdditiveNormalize(!settings.normalizeLoudness)
      }
      .accessibilityIdentifier("export-normalize")

      toggleRow(
        symbol: "cpu",
        title: "Hardware acceleration",
        detail: "uses the GPU; turn off if export fails",
        isOn: settings.usesHardwareAcceleration
      ) {
        settings.setAdditiveHardwareAcceleration(!settings.usesHardwareAcceleration)
      }
      .accessibilityIdentifier("export-hwaccel")

      ExportRangeEditor(
        range: Binding(
          get: { settings.range },
          set: { settings.range = $0 }
        ),
        duration: programDuration,
        framesPerSecond: settings.framesPerSecond
      )
    }
  }

  private var lockedSummary: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        caption("Output")
        Spacer()
        HStack(spacing: 4) {
          Image(systemName: "lock.fill")
          Text("set by preset")
        }
        .font(.system(size: 10, weight: .semibold))
        .foregroundStyle(H3Color.textSecondary)
      }

      LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 11) {
        summaryCell("Format", settings.format.label)
        summaryCell("Resolution", settings.resolution.shortLabel)
        summaryCell("Frame rate", "\(Int(settings.framesPerSecond)) fps")
        summaryCell("Quality profile", settings.profile.label)
      }

      Button {
        settings.apply(preset: .custom, seed: seed)
      } label: {
        HStack(spacing: 6) {
          Image(systemName: "slider.horizontal.3")
          Text("Switch to Custom to change these")
        }
        .font(.system(size: 12, weight: .medium))
        .foregroundStyle(H3Color.textSecondary)
      }
      .buttonStyle(.plain)
      .accessibilityIdentifier("export-custom")
    }
    .padding(14)
    .background(H3Color.chrome)
    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 12, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
  }

  private var customFields: some View {
    VStack(alignment: .leading, spacing: 18) {
      labeled("Resolution") {
        chipRow(ProjectResolution.allCases, selected: settings.resolution) { resolution in
          settings.updateCustom { $0.resolution = resolution }
        }
      }
      labeled("Frame rate") {
        chipRow(ProjectSettings.frameRates, selected: settings.framesPerSecond) { rate in
          settings.updateCustom { $0.framesPerSecond = rate }
        } label: { rate in
          "\(Int(rate)) fps"
        }
      }
      labeled("Format") {
        chipRow(ProgramExportFormat.allCases, selected: settings.format) { format in
          settings.updateCustom { $0.format = format }
        }
      }
      if settings.format == .h264 {
        labeled("Quality profile") {
          chipRow(ProgramExportProfile.allCases, selected: settings.profile) { profile in
            settings.updateCustom { $0.profile = profile }
          }
        }
      }
    }
  }

  private var bitrateSlider: some View {
    VStack(alignment: .leading, spacing: 7) {
      HStack {
        caption("Video bitrate")
        Spacer()
        Text("\(Int(settings.videoBitrateKbps.rounded()))")
          .font(.system(size: 13, weight: .bold, design: .monospaced))
          .foregroundStyle(H3Color.accent)
          + Text(" kbps")
          .font(.system(size: 9, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
      }
      Slider(
        value: Binding(
          get: { settings.videoBitrateKbps },
          set: { value in
            let step = ProgramExportSettings.videoBitrateStepKbps
            settings.updateCustom { $0.videoBitrateKbps = (value / step).rounded() * step }
          }
        ),
        in: ProgramExportSettings.minimumVideoBitrateKbps...ProgramExportSettings.maximumVideoBitrateKbps
      )
      .tint(H3Color.accent)
      .disabled(settings.format == .proRes)
      .accessibilityIdentifier("export-bitrate")
      HStack {
        Text("Smaller file")
        Spacer()
        Text("Higher quality")
      }
      .font(.system(size: 9, design: .monospaced))
      .foregroundStyle(H3Color.textSecondary.opacity(0.7))
    }
  }

  private func labeled(_ title: String, @ViewBuilder content: () -> some View) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      caption(title)
      content()
    }
  }

  private func caption(_ title: String) -> some View {
    Text(title.uppercased())
      .font(.system(size: 10, weight: .bold))
      .tracking(1.0)
      .foregroundStyle(H3Color.textSecondary)
  }

  private func summaryCell(_ title: String, _ value: String) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(title)
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
      Text(value)
        .font(.system(size: 13, weight: .semibold))
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private func chipRow<Item: Hashable>(
    _ items: [Item],
    selected: Item,
    onSelect: @escaping (Item) -> Void,
    label: ((Item) -> String)? = nil
  ) -> some View {
    FlowChips(items: items, selected: selected, label: label, onSelect: onSelect)
  }

  private func toggleRow(
    symbol: String,
    title: String,
    detail: String,
    isOn: Bool,
    action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      HStack(spacing: 10) {
        Image(systemName: symbol)
          .foregroundStyle(H3Color.textSecondary)
        (Text(title).font(.system(size: 13, weight: .semibold))
          + Text("  \(detail)")
          .font(.system(size: 11))
          .foregroundColor(H3Color.textSecondary))
        Spacer()
        Capsule()
          .fill(isOn ? H3Color.accent : H3Color.controlFill)
          .frame(width: 34, height: 20)
          .overlay(alignment: isOn ? .trailing : .leading) {
            Circle()
              .fill(Color.white)
              .frame(width: 16, height: 16)
              .padding(2)
          }
      }
      .padding(.vertical, 4)
    }
    .buttonStyle(.plain)
  }
}

private struct FlowChips<Item: Hashable>: View {
  let items: [Item]
  let selected: Item
  let label: ((Item) -> String)?
  let onSelect: (Item) -> Void

  var body: some View {
    LazyVGrid(
      columns: [GridItem(.adaptive(minimum: 68), spacing: 7)],
      alignment: .leading,
      spacing: 7
    ) {
      ForEach(Array(items.enumerated()), id: \.offset) { _, item in
        let isSelected = item == selected
        Button {
          onSelect(item)
        } label: {
          Text(chipLabel(item))
            .font(.system(size: 11, weight: .semibold))
            .frame(maxWidth: .infinity)
            .frame(height: 28)
            .background(isSelected ? H3Color.accent : H3Color.chrome)
            .foregroundStyle(isSelected ? Color.white : H3Color.textPrimary)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .overlay {
              RoundedRectangle(cornerRadius: 7, style: .continuous)
                .stroke(isSelected ? H3Color.accent : H3Color.line, lineWidth: 1)
            }
        }
        .buttonStyle(.plain)
      }
    }
  }

  private func chipLabel(_ item: Item) -> String {
    if let provided = label { return provided(item) }
    if let resolution = item as? ProjectResolution { return resolution.shortLabel }
    if let format = item as? ProgramExportFormat { return format.label }
    if let profile = item as? ProgramExportProfile { return profile.label }
    if let quality = item as? ProgramAudioQuality { return quality.label }
    if let rate = item as? Double { return "\(Int(rate)) fps" }
    return "\(item)"
  }
}

struct ExportRangeEditor: View {
  @Binding var range: ProgramExportRange
  let duration: TimeInterval
  let framesPerSecond: Double

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("EXPORT RANGE")
        .font(.system(size: 10, weight: .bold))
        .tracking(1.0)
        .foregroundStyle(H3Color.textSecondary)

      HStack(spacing: 8) {
        modeButton(
          symbol: "film",
          title: "Full video",
          detail: ProgramExportSettings.formatClock(duration),
          selected: range.mode == .full
        ) {
          range = ProgramExportRange(mode: .full, inSec: 0, outSec: duration)
        }
        modeButton(
          symbol: "selection.pin.in.out",
          title: "Custom range",
          detail: nil,
          selected: range.mode == .custom
        ) {
          let span = range.resolved(in: duration)
          range = ProgramExportRange(mode: .custom, inSec: span.inSec, outSec: span.outSec)
        }
        .accessibilityIdentifier("export-range-custom")
      }

      if range.mode == .custom {
        customEditor
      }
    }
    .accessibilityIdentifier("export-range")
  }

  private var customEditor: some View {
    let span = range.resolved(in: duration)
    let inPct = duration > 0 ? span.inSec / duration : 0
    let outPct = duration > 0 ? span.outSec / duration : 1
    return VStack(spacing: 13) {
      GeometryReader { proxy in
        ZStack(alignment: .leading) {
          Capsule().fill(Color.black.opacity(0.45))
          Capsule()
            .fill(H3Color.accent.opacity(0.85))
            .frame(width: max(6, proxy.size.width * (outPct - inPct)))
            .offset(x: proxy.size.width * inPct)
          handle(at: inPct, width: proxy.size.width) { setIn($0, width: proxy.size.width) }
          handle(at: outPct, width: proxy.size.width) { setOut($0, width: proxy.size.width) }
        }
      }
      .frame(height: 30)

      HStack(alignment: .bottom, spacing: 12) {
        clockField("Start", span.inSec) { setInSeconds($0) }
        Image(systemName: "arrow.right")
          .foregroundStyle(H3Color.textSecondary)
          .padding(.bottom, 8)
        clockField("End", span.outSec) { setOutSeconds($0) }
        Spacer()
        VStack(alignment: .trailing, spacing: 3) {
          Text("DURATION")
            .font(.system(size: 9, weight: .bold))
            .foregroundStyle(H3Color.textSecondary)
          Text(ProgramExportSettings.formatClock(span.outSec - span.inSec))
            .font(.system(size: 14, weight: .bold, design: .monospaced))
            .foregroundStyle(H3Color.accent)
        }
      }
    }
    .padding(14)
    .background(H3Color.chrome)
    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 12, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
  }

  private func modeButton(
    symbol: String,
    title: String,
    detail: String?,
    selected: Bool,
    action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      HStack(spacing: 6) {
        Image(systemName: symbol)
        Text(title)
        if let detail {
          Text(detail)
            .font(.system(size: 10, design: .monospaced))
            .opacity(0.6)
        }
      }
      .font(.system(size: 12, weight: .semibold))
      .frame(maxWidth: .infinity)
      .frame(height: 34)
      .background(selected ? H3Color.accent.opacity(0.16) : H3Color.chrome)
      .foregroundStyle(selected ? H3Color.accent : H3Color.textPrimary)
      .overlay {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
          .stroke(selected ? H3Color.accent : H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
    .buttonStyle(.plain)
  }

  private func clockField(_ title: String, _ seconds: TimeInterval, onCommit: @escaping (TimeInterval) -> Void)
    -> some View
  {
    let frame = 1 / max(framesPerSecond, 1)
    return VStack(alignment: .leading, spacing: 4) {
      Text(title.uppercased())
        .font(.system(size: 9, weight: .bold))
        .foregroundStyle(H3Color.textSecondary)
      HStack(spacing: 0) {
        Button {
          onCommit(seconds - frame)
        } label: {
          Image(systemName: "minus")
            .frame(width: 26, height: 30)
        }
        .buttonStyle(.plain)
        Text(ProgramExportSettings.formatClock(seconds))
          .font(.system(size: 13, weight: .semibold, design: .monospaced))
          .frame(width: 52)
        Button {
          onCommit(seconds + frame)
        } label: {
          Image(systemName: "plus")
            .frame(width: 26, height: 30)
        }
        .buttonStyle(.plain)
      }
      .background(Color.black.opacity(0.25))
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
  }

  private func handle(at pct: Double, width: CGFloat, onDrag: @escaping (CGFloat) -> Void) -> some View {
    Capsule()
      .fill(Color.white)
      .frame(width: 6, height: 22)
      .offset(x: width * pct - 3)
      .gesture(
        DragGesture(minimumDistance: 1)
          .onChanged { value in
            onDrag(value.location.x)
          }
      )
  }

  private func setIn(_ x: CGFloat, width: CGFloat) {
    guard width > 0 else { return }
    setInSeconds(max(0, min(1, x / width)) * duration)
  }

  private func setOut(_ x: CGFloat, width: CGFloat) {
    guard width > 0 else { return }
    setOutSeconds(max(0, min(1, x / width)) * duration)
  }

  private func setInSeconds(_ seconds: TimeInterval) {
    let clamped = ProgramExportSettings.clampRange(
      inSec: seconds,
      outSec: range.outSec == 0 ? duration : range.outSec,
      duration: duration,
      framesPerSecond: framesPerSecond
    )
    range = ProgramExportRange(mode: .custom, inSec: clamped.inSec, outSec: clamped.outSec)
  }

  private func setOutSeconds(_ seconds: TimeInterval) {
    let clamped = ProgramExportSettings.clampRange(
      inSec: range.inSec,
      outSec: seconds,
      duration: duration,
      framesPerSecond: framesPerSecond
    )
    range = ProgramExportRange(mode: .custom, inSec: clamped.inSec, outSec: clamped.outSec)
  }
}