import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct ProjectSettingsPanel: View {
  @Bindable var model: AppModel

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        group("Platform") {
          platformMenu
        }

        if model.project.settings.isCustomPlatform {
          group("Aspect Ratio") {
            aspectChips
          }
          group("Resolution") {
            resolutionMenu
          }
          group("Frame Rate") {
            fpsChips
          }
        }

        group("Background") {
          backgroundSwatches
        }

        group("Output") {
          toneMapping
          exposureSlider
          Text("Tone mapping and exposure are stored with the project. Live preview still shows the source media.")
            .font(.system(size: 10))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }

        group("Master Output") {
          masterCard
        }

        if model.project.settings.framesPerSecond != 24 {
          Text("H3 clips are generated at 24 fps. The project rate only changes the playhead and timecode.")
            .font(.system(size: 10))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
      .padding(16)
    }
    .frame(width: 320)
    .background(H3Color.surface)
    .overlay(alignment: .trailing) {
      Rectangle().fill(H3Color.line).frame(width: 1)
    }
    .accessibilityIdentifier("project-settings")
  }

  private var settings: ProjectSettings {
    model.project.settings
  }

  private var platformMenu: some View {
    Menu {
      ForEach(ProjectPlatform.allCases) { platform in
        Button(platform.label) {
          model.updateProjectSettings { $0.apply(platform: platform) }
        }
      }
    } label: {
      HStack {
        VStack(alignment: .leading, spacing: 2) {
          Text(settings.platform.label)
            .font(.system(size: 13, weight: .semibold))
          Text(
            settings.platform == .custom
              ? settings.platform.detail
              : "\(settings.width)×\(settings.height) · \(settings.platform.detail)"
          )
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
        }
        Spacer()
        Image(systemName: "chevron.up.chevron.down")
          .font(.system(size: 10, weight: .semibold))
          .foregroundStyle(H3Color.textSecondary)
      }
      .padding(.horizontal, 11)
      .padding(.vertical, 9)
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
    .buttonStyle(.plain)
  }

  private var aspectChips: some View {
    HStack(spacing: 6) {
      ForEach(ProjectAspect.allCases) { aspect in
        let selected = settings.aspect == aspect
        Button {
          model.updateProjectSettings { $0.apply(aspect: aspect) }
        } label: {
          HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2, style: .continuous)
              .stroke(selected ? H3Color.accent : H3Color.line, lineWidth: 1)
              .frame(width: chipSize(aspect).width, height: chipSize(aspect).height)
            Text(aspect.rawValue)
              .font(.system(size: 11, weight: .semibold, design: .monospaced))
          }
          .padding(.horizontal, 8)
          .frame(height: 28)
          .background(selected ? H3Color.accent.opacity(0.14) : H3Color.chrome)
          .overlay {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
              .stroke(selected ? H3Color.accent : H3Color.line, lineWidth: 1)
          }
          .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        }
        .buttonStyle(.plain)
      }
    }
  }

  private var resolutionMenu: some View {
    Menu {
      ForEach(ProjectResolution.allCases) { resolution in
        Button(resolution.label) {
          model.updateProjectSettings { $0.apply(resolution: resolution) }
        }
      }
    } label: {
      HStack {
        Text(settings.resolutionLabel)
          .font(.system(size: 12, weight: .medium, design: .monospaced))
        Spacer()
        Image(systemName: "chevron.up.chevron.down")
          .font(.system(size: 10, weight: .semibold))
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
  }

  private var fpsChips: some View {
    HStack(spacing: 4) {
      ForEach(ProjectSettings.frameRates, id: \.self) { rate in
        let selected = abs(settings.framesPerSecond - rate) < 0.01
        Button {
          model.updateProjectSettings { $0.apply(frameRate: rate) }
        } label: {
          Text(rate.formatted(.number.precision(.fractionLength(0))))
            .font(.system(size: 11, weight: .semibold, design: .monospaced))
            .frame(maxWidth: .infinity)
            .frame(height: 28)
            .background(selected ? H3Color.accent : H3Color.chrome)
            .foregroundStyle(selected ? Color.white : H3Color.textPrimary)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
      }
    }
  }

  private var backgroundSwatches: some View {
    HStack(spacing: 8) {
      ForEach(ProjectBackground.allCases) { background in
        Button {
          model.updateProjectSettings { $0.background = background }
        } label: {
          ZStack {
            swatchFill(background)
            if settings.background == background {
              Image(systemName: "checkmark")
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(background == .white || background == .clear ? Color.black : Color.white)
            }
          }
          .frame(width: 28, height: 28)
          .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
          .overlay {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
              .stroke(settings.background == background ? H3Color.accent : H3Color.line, lineWidth: 1.5)
          }
        }
        .buttonStyle(.plain)
        .help(background == .clear ? "Transparent" : background.rawValue)
      }
    }
  }

  private var toneMapping: some View {
    HStack(spacing: 4) {
      ForEach(ProjectToneMapping.allCases) { mode in
        let selected = settings.toneMapping == mode
        Button {
          model.updateProjectSettings { $0.toneMapping = mode }
        } label: {
          Text(mode.label)
            .font(.system(size: 11, weight: .semibold))
            .frame(maxWidth: .infinity)
            .frame(height: 28)
            .background(selected ? H3Color.accent : H3Color.chrome)
            .foregroundStyle(selected ? Color.white : H3Color.textPrimary)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
      }
    }
  }

  private var exposureSlider: some View {
    let disabled = settings.toneMapping == .none
    return HStack(spacing: 8) {
      Text("Exposure")
        .font(.system(size: 10, weight: .semibold, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
        .frame(width: 56, alignment: .leading)
      Slider(
        value: Binding(
          get: { settings.exposureStops },
          set: { stops in
            model.updateProjectSettings { $0.exposure = pow(2, stops) }
          }
        ),
        in: -3...3
      )
      .tint(H3Color.accent)
      .disabled(disabled)
      Text(exposureLabel)
        .font(.system(size: 10, weight: .semibold, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
        .frame(width: 48, alignment: .trailing)
    }
    .opacity(disabled ? 0.35 : 1)
  }

  private var masterCard: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("MASTER")
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .tracking(1.6)
          .foregroundStyle(H3Color.textSecondary)
        Spacer()
        Text("−∞")
          .font(.system(size: 15, weight: .bold, design: .monospaced))
        Text("dB")
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
      }
      meterRow("L")
      meterRow("R")
      HStack {
        Text("−∞").frame(maxWidth: .infinity, alignment: .leading)
        Text("−24").frame(maxWidth: .infinity)
        Text("−12").frame(maxWidth: .infinity)
        Text("−6").frame(maxWidth: .infinity)
        Text("0").frame(maxWidth: .infinity, alignment: .trailing)
      }
      .font(.system(size: 7.5, design: .monospaced))
      .foregroundStyle(H3Color.textSecondary.opacity(0.7))
      .padding(.leading, 18)

      HStack(spacing: 9) {
        Image(systemName: "speaker.wave.2")
          .foregroundStyle(H3Color.textSecondary)
        Slider(
          value: Binding(
            get: { settings.masterGain },
            set: { gain in
              model.updateProjectSettings { $0.masterGain = gain }
            }
          ),
          in: 0...1
        )
        .tint(H3Color.accent)
        Text("\(Int((settings.masterGain * 100).rounded()))%")
          .font(.system(size: 10, weight: .semibold, design: .monospaced))
          .frame(width: 36, alignment: .trailing)
      }
      Text("Peak meters stay idle until playback has an audio tap.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
    }
    .padding(13)
    .background(H3Color.chrome)
    .overlay {
      RoundedRectangle(cornerRadius: 10, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
  }

  private func meterRow(_ label: String) -> some View {
    HStack(spacing: 8) {
      Text(label)
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
        .frame(width: 10)
      HStack(spacing: 2) {
        ForEach(0..<28, id: \.self) { index in
          let fraction = (Double(index) + 0.5) / 28
          RoundedRectangle(cornerRadius: 1)
            .fill(meterColor(fraction).opacity(0.16))
            .frame(height: 9)
        }
      }
    }
  }

  private func meterColor(_ fraction: Double) -> Color {
    if fraction > 0.88 { return H3Color.danger }
    if fraction > 0.72 { return Color(red: 229 / 255, green: 162 / 255, blue: 60 / 255) }
    return H3Color.clipAudio
  }

  private func group<Content: View>(
    _ title: String,
    @ViewBuilder content: () -> Content
  ) -> some View {
    VStack(alignment: .leading, spacing: 11) {
      Text(title)
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .tracking(1.6)
        .textCase(.uppercase)
        .foregroundStyle(H3Color.textSecondary)
      content()
    }
  }

  private func chipSize(_ aspect: ProjectAspect) -> CGSize {
    let maxEdge: CGFloat = 16
    if aspect.fraction >= 1 {
      return CGSize(width: maxEdge, height: max(4, maxEdge / aspect.fraction))
    }
    return CGSize(width: max(4, maxEdge * aspect.fraction), height: maxEdge)
  }

  private func swatchFill(_ background: ProjectBackground) -> some View {
    Group {
      if background.isClear {
        H3Checkerboard()
      } else {
        Color(h3Hex: background.rawValue)
      }
    }
  }

  private var exposureLabel: String {
    let stops = settings.exposureStops
    let prefix = stops >= 0 ? "+" : ""
    return "\(prefix)\(stops.formatted(.number.precision(.fractionLength(1)))) EV"
  }
}

