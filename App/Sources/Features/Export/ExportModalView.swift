import AVFoundation
import AVKit
import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI
import UniformTypeIdentifiers

private enum ExportRenderState: Equatable {
  case idle
  case active(phase: String, fraction: Double, elapsed: TimeInterval)
  case done(url: URL, sizeLabel: String)
  case failed(String)
}

struct ExportModalView: View {
  @Bindable var model: AppModel

  @State private var settings = ProgramExportSettings()
  @State private var seed = ProgramExportSettings()
  @State private var render = ExportRenderState.idle
  @State private var showsTrailingWarning = false
  @State private var exportTask: Task<Void, Never>?
  @State private var exportGeneration = 0
  @State private var startedAt: Date?
  @State private var previewPlayer: AVPlayer?
  @State private var livePreview: NSImage?

  var body: some View {
    ZStack {
      Color.black.opacity(0.62)
        .ignoresSafeArea()
        .onTapGesture {
          if !isRendering { close() }
        }

      VStack(spacing: 0) {
        header
        Divider().overlay(H3Color.line)
        presetStrip
        bodyColumns
        Divider().overlay(H3Color.line)
        footer
      }
      .frame(maxWidth: 1_360, maxHeight: 900)
      .background(Color(red: 13 / 255, green: 15 / 255, blue: 19 / 255))
      .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
      .overlay {
        RoundedRectangle(cornerRadius: 14, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .shadow(color: .black.opacity(0.55), radius: 40, y: 18)
      .padding(24)
    }
    .foregroundStyle(H3Color.textPrimary)
    .accessibilityIdentifier("export-modal")
    .onAppear(perform: seedFromProject)
    .onDisappear { exportTask?.cancel() }
    .alert("Audio continues past the picture", isPresented: $showsTrailingWarning) {
      Button("Cancel", role: .cancel) {}
      Button("Export anyway") { presentSaveAndStart() }
    } message: {
      Text(
        "The audio track continues \(ProgramExportSettings.formatClock(trailingAudioDuration)) past the program end. The file will end there."
      )
    }
  }

  private var project: H3ddleProject { model.project }
  private var plan: ProgramCompositionPlan { ProgramCompositionPlan(project: project) }
  private var programDuration: TimeInterval {
    plan.exportDuration(includeTextLane: settings.includeTextLane)
  }
  private var canExport: Bool {
    programDuration > 0.001 && !isRendering
  }
  private var isRendering: Bool {
    if case .active = render { return true }
    return false
  }

  private var trailingAudioDuration: TimeInterval {
    guard model.audioLaneAudible else { return 0 }
    return plan.trailingAudioPast(range: settings.range)
  }

  private var header: some View {
    HStack(spacing: 12) {
      RoundedRectangle(cornerRadius: 9, style: .continuous)
        .fill(H3Color.accent.opacity(0.14))
        .overlay {
          RoundedRectangle(cornerRadius: 9, style: .continuous)
            .stroke(H3Color.accent.opacity(0.4), lineWidth: 1)
        }
        .overlay {
          Image(systemName: "square.and.arrow.up")
            .font(.system(size: 15, weight: .semibold))
            .foregroundStyle(H3Color.accent)
        }
        .frame(width: 34, height: 34)

      VStack(alignment: .leading, spacing: 2) {
        Text("Export Video")
          .font(.system(size: 16, weight: .semibold))
        Text(headerMeta)
          .font(.system(size: 10.5, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
          .lineLimit(1)
      }
      Spacer()
      Button {
        close()
      } label: {
        Image(systemName: "xmark")
          .font(.system(size: 12, weight: .semibold))
          .frame(width: 34, height: 34)
          .background(H3Color.controlFill)
          .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
          .frame(width: 44, height: 44)
          .contentShape(Rectangle())
      }
      .buttonStyle(.plain)
      .disabled(isRendering)
      .accessibilityIdentifier("export-close")
    }
    .padding(.horizontal, 24)
    .padding(.vertical, 16)
  }

  private var headerMeta: String {
    let size = settings.outputPixelSize(project: project.settings)
    return "\(project.name) · \(ProgramExportSettings.formatClock(programDuration)) · \(size.width)×\(size.height) · \(project.settings.aspect.rawValue)"
  }

  private var presetStrip: some View {
    VStack(alignment: .leading, spacing: 9) {
      HStack(spacing: 8) {
        Text("QUALITY PRESET")
          .font(.system(size: 10, weight: .bold))
          .tracking(1.0)
          .foregroundStyle(H3Color.textSecondary)
        Text("project stays \(project.settings.width)×\(project.settings.height) · \(project.settings.aspect.rawValue)")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary.opacity(0.8))
      }
      HStack(spacing: 12) {
        ForEach(ProgramExportPreset.allCases) { preset in
          presetCard(preset)
        }
      }
    }
    .padding(.horizontal, 24)
    .padding(.top, 18)
    .padding(.bottom, 6)
  }

  private func presetCard(_ preset: ProgramExportPreset) -> some View {
    let selected = settings.preset == preset
    return Button {
      settings.apply(preset: preset, seed: seed)
    } label: {
      VStack(alignment: .leading, spacing: 6) {
        Image(systemName: preset.symbolName)
          .font(.system(size: 16, weight: .semibold))
          .foregroundStyle(selected ? H3Color.accent : H3Color.textSecondary)
        Text(preset.label)
          .font(.system(size: 13, weight: .semibold))
        Text(preset.subtitle)
          .font(.system(size: 9.5, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
      }
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(12)
      .background(selected ? H3Color.accent.opacity(0.12) : H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 12, style: .continuous)
          .stroke(selected ? H3Color.accent : H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
    .buttonStyle(.plain)
    .accessibilityIdentifier("preset-\(preset.rawValue)")
  }

  private var bodyColumns: some View {
    HStack(alignment: .top, spacing: 22) {
      leftColumn
        .frame(maxWidth: .infinity, alignment: .top)
      ScrollView {
        ExportControlsView(
          settings: $settings,
          seed: seed,
          programDuration: programDuration
        )
        .padding(.trailing, 4)
      }
      .frame(maxWidth: .infinity, alignment: .top)
    }
    .padding(.horizontal, 24)
    .padding(.vertical, 16)
  }

  private var leftColumn: some View {
    VStack(alignment: .leading, spacing: 16) {
      ZStack {
        Color.black
        previewSurface
      }
      .aspectRatio(max(project.settings.aspectFraction, 0.3), contentMode: .fit)
      .frame(maxWidth: .infinity, maxHeight: 520)
      .layoutPriority(1)
      .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
      .overlay {
        RoundedRectangle(cornerRadius: 12, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .accessibilityIdentifier("export-preview")

      if trailingAudioDuration > 0.05 {
        HStack(alignment: .top, spacing: 8) {
          Image(systemName: "exclamationmark.triangle.fill")
            .foregroundStyle(H3Color.accent)
          Text(
            "Audio continues \(ProgramExportSettings.formatClock(trailingAudioDuration)) past the picture. Export ends at the program end."
          )
          .font(.system(size: 11.5))
          .fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .background(H3Color.accent.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .accessibilityIdentifier("export-trailing-audio")
      }

      progressPanel
    }
  }

  @ViewBuilder
  private var previewSurface: some View {
    if case .done(let url, _) = render {
      ExportResultPlayer(url: url, player: $previewPlayer)
    } else if let livePreview {
      Image(nsImage: livePreview)
        .resizable()
        .scaledToFit()
    } else {
      ExportPosterView(
        project: project,
        time: settings.range.resolved(in: programDuration).inSec
      )
    }
  }

  private var progressPanel: some View {
    VStack(alignment: .leading, spacing: 0) {
      HStack {
        Text("Render progress")
          .font(.system(size: 12.5, weight: .semibold))
        Spacer()
        Text(progressTitle)
          .font(.system(size: 10, weight: .semibold))
          .foregroundStyle(progressTint)
          .padding(.horizontal, 8)
          .frame(height: 20)
          .background(progressTint.opacity(0.14))
          .clipShape(Capsule())
      }
      .padding(.horizontal, 16)
      .padding(.vertical, 12)
      Divider().overlay(H3Color.line.opacity(0.6))

      switch render {
      case .idle:
        VStack(spacing: 8) {
          Image(systemName: "dot.radiowaves.left.and.right")
            .font(.system(size: 22))
            .foregroundStyle(H3Color.textSecondary.opacity(0.6))
          Text("Ready to export")
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(H3Color.textSecondary)
          Text("Progress, phase and time remaining show here once you start.")
            .font(.system(size: 11.5))
            .foregroundStyle(H3Color.textSecondary.opacity(0.8))
            .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(24)

      case .active(let phase, let fraction, let elapsed):
        VStack(alignment: .leading, spacing: 12) {
          HStack {
            Text(phase)
              .font(.system(size: 13.5, weight: .semibold))
            Spacer()
            Text("\(Int((fraction * 100).rounded()))")
              .font(.system(size: 24, weight: .bold, design: .monospaced))
              + Text("%")
              .font(.system(size: 13))
              .foregroundColor(H3Color.textSecondary)
          }
          GeometryReader { proxy in
            ZStack(alignment: .leading) {
              Capsule().fill(Color.white.opacity(0.08))
              Capsule()
                .fill(H3Color.accent)
                .frame(width: max(6, proxy.size.width * fraction))
            }
          }
          .frame(height: 8)
          HStack {
            timeCell("Elapsed", ProgramExportSettings.formatClock(elapsed))
            timeCell("Remaining", remainingLabel(fraction: fraction, elapsed: elapsed))
            timeCell("Speed", speedLabel(fraction: fraction, elapsed: elapsed))
          }
          Button(role: .destructive) {
            cancelExport()
          } label: {
            HStack {
              Image(systemName: "stop.circle")
              Text("Cancel render")
            }
            .frame(maxWidth: .infinity)
            .frame(height: 32)
          }
          .buttonStyle(H3QuietButtonStyle())
          .accessibilityIdentifier("export-cancel")
        }
        .padding(16)

      case .done(let url, let sizeLabel):
        VStack(alignment: .leading, spacing: 14) {
          HStack(spacing: 12) {
            Image(systemName: "checkmark")
              .font(.system(size: 16, weight: .bold))
              .foregroundStyle(Color(red: 70 / 255, green: 168 / 255, blue: 131 / 255))
              .frame(width: 40, height: 40)
              .background(Color(red: 70 / 255, green: 168 / 255, blue: 131 / 255).opacity(0.16))
              .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
              Text("Export complete")
                .font(.system(size: 14, weight: .semibold))
              Text("\(url.lastPathComponent) · \(sizeLabel)")
                .font(.system(size: 10.5, design: .monospaced))
                .foregroundStyle(H3Color.textSecondary)
                .lineLimit(1)
            }
          }
          HStack(spacing: 8) {
            Button {
              NSWorkspace.shared.activateFileViewerSelecting([url])
            } label: {
              Label("Reveal", systemImage: "folder")
                .frame(maxWidth: .infinity)
                .frame(height: 32)
            }
            .buttonStyle(H3QuietButtonStyle())
            .accessibilityIdentifier("export-reveal")

            Button {
              resetRender()
            } label: {
              Label("Export another", systemImage: "arrow.counterclockwise")
                .frame(maxWidth: .infinity)
                .frame(height: 32)
            }
            .buttonStyle(H3QuietButtonStyle())
          }
        }
        .padding(16)

      case .failed(let message):
        VStack(spacing: 10) {
          Image(systemName: "exclamationmark.triangle")
            .font(.system(size: 20))
            .foregroundStyle(H3Color.danger)
          Text(message)
            .font(.system(size: 12.5))
            .multilineTextAlignment(.center)
          Button("Try again") { resetRender() }
            .buttonStyle(H3QuietButtonStyle())
        }
        .frame(maxWidth: .infinity)
        .padding(18)
      }
    }
    .background(H3Color.chrome)
    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 14, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .accessibilityIdentifier("render-progress")
  }

  private var footer: some View {
    HStack {
      Text("Exporting ")
        .foregroundStyle(H3Color.textSecondary)
        + Text(specLine)
        .fontWeight(.semibold)
      Spacer()
      Button("Cancel") { close() }
        .buttonStyle(H3QuietButtonStyle())
        .disabled(isRendering)
      Button {
        requestExport()
      } label: {
        HStack(spacing: 6) {
          Image(systemName: "square.and.arrow.up.fill")
          Text(isRendering ? "Rendering…" : "Export now")
        }
      }
      .buttonStyle(H3PrimaryButtonStyle())
      .disabled(!canExport)
      .accessibilityIdentifier("export-now")
    }
    .font(.system(size: 12.5))
    .padding(.horizontal, 24)
    .padding(.vertical, 14)
    .background(Color(red: 15 / 255, green: 18 / 255, blue: 23 / 255))
  }

  private var specLine: String {
    "\(settings.format.label) · \(settings.resolution.shortLabel) · \(Int(settings.framesPerSecond)) fps"
  }

  private var progressTitle: String {
    switch render {
    case .idle: "Idle"
    case .active: "Rendering"
    case .done: "Done"
    case .failed: "Error"
    }
  }

  private var progressTint: Color {
    switch render {
    case .failed: H3Color.danger
    case .done: Color(red: 70 / 255, green: 168 / 255, blue: 131 / 255)
    case .active: H3Color.accent
    case .idle: H3Color.textSecondary
    }
  }

  private func timeCell(_ title: String, _ value: String) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(title)
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
      Text(value)
        .font(.system(size: 13, weight: .semibold, design: .monospaced))
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private func remainingLabel(fraction: Double, elapsed: TimeInterval) -> String {
    guard fraction > 0.02 else { return "—" }
    let remaining = elapsed * (1 - fraction) / fraction
    return ProgramExportSettings.formatClock(remaining)
  }

  private func speedLabel(fraction: Double, elapsed: TimeInterval) -> String {
    guard elapsed > 0.2 else { return "—" }
    let range = max(0.001, settings.rangeDuration(programDuration: programDuration))
    let encodedSeconds = range * fraction
    let fps = encodedSeconds / elapsed * settings.framesPerSecond
    return "\(max(0, Int(fps.rounded()))) fps"
  }

  private func seedFromProject() {
    var next = ProgramExportSettings.makeDefault(project: project)
    next.includeAudioLane = model.audioLaneAudible
    next.includeTextLane = model.textLaneAudible
    seed = next
    settings = next
    livePreview = nil
    render = .idle
  }

  private func requestExport() {
    guard canExport else { return }
    settings.includeAudioLane = model.audioLaneAudible
    settings.includeTextLane = model.textLaneAudible
    if settings.includeAudioLane,
      plan.requiresTrailingAudioWarning(range: settings.range)
    {
      showsTrailingWarning = true
      return
    }
    presentSaveAndStart()
  }

  private func presentSaveAndStart() {
    let panel = NSSavePanel()
    panel.canCreateDirectories = true
    panel.isExtensionHidden = false
    panel.nameFieldStringValue = settings.suggestedFileName(projectName: project.name)
    panel.allowedContentTypes = settings.format == .proRes ? [.quickTimeMovie] : [.mpeg4Movie]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    startExport(to: url)
  }

  private func startExport(to url: URL) {
    exportTask?.cancel()
    exportGeneration += 1
    let generation = exportGeneration
    previewPlayer?.pause()
    previewPlayer = nil
    livePreview = nil
    startedAt = Date()
    render = .active(phase: "Preparing media", fraction: 0, elapsed: 0)
    let project = project
    let settings = settings
    exportTask = Task {
      do {
        for try await event in ProgramExporter().export(
          project: project,
          settings: settings,
          destination: url
        ) {
          guard generation == exportGeneration else { return }
          let elapsed = Date().timeIntervalSince(startedAt ?? Date())
          switch event {
          case .preparing:
            render = .active(phase: "Preparing media", fraction: 0, elapsed: elapsed)
          case .progress(let phase, let fraction):
            render = .active(phase: phase, fraction: fraction, elapsed: elapsed)
          case .preview(let frame):
            livePreview = NSImage(cgImage: frame.image, size: .zero)
          case .completed(let finished):
            let size = (try? finished.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            render = .done(url: finished, sizeLabel: formattedFileSize(size))
          }
        }
      } catch is CancellationError {
        applyExportOutcome(generation: generation) {
          livePreview = nil
          render = .idle
        }
      } catch MediaExportError.cancelled {
        applyExportOutcome(generation: generation) {
          livePreview = nil
          render = .idle
        }
      } catch MediaExportError.emptyProgram {
        applyExportOutcome(generation: generation) {
          render = .failed("Add a visual or title before exporting.")
        }
      } catch MediaExportError.failed(let message) {
        applyExportOutcome(generation: generation) {
          render = .failed(message)
        }
      } catch {
        applyExportOutcome(generation: generation) {
          render = .failed(error.localizedDescription)
        }
      }
    }
  }

  private func applyExportOutcome(generation: Int, _ update: () -> Void) {
    guard generation == exportGeneration else { return }
    update()
  }

  private func formattedFileSize(_ bytes: Int) -> String {
    let megabytes = Double(bytes) / 1_024 / 1_024
    if megabytes >= 1024 {
      return String(format: "%.2f GB", megabytes / 1024)
    }
    if megabytes < 1 {
      return String(format: "%.0f KB", Double(bytes) / 1_024)
    }
    return "\(Int(megabytes.rounded())) MB"
  }

  private func cancelExport() {
    exportGeneration += 1
    exportTask?.cancel()
    exportTask = nil
    livePreview = nil
    render = .idle
  }

  private func resetRender() {
    previewPlayer?.pause()
    previewPlayer = nil
    livePreview = nil
    render = .idle
  }

  private func close() {
    if isRendering { return }
    exportTask?.cancel()
    previewPlayer?.pause()
    model.showsExport = false
  }
}

private struct ExportPosterView: View {
  let project: H3ddleProject
  let time: TimeInterval
  @State private var presenter = ProgramFramePresenter()
  @State private var posterSize: CGSize = .zero

  var body: some View {
    ZStack {
      posterBackground
      if let image = presenter.image {
        Image(nsImage: image)
          .resizable()
          .interpolation(.high)
          .scaledToFit()
      }
    }
    .background {
      GeometryReader { proxy in
        Color.clear.preference(key: ExportPosterSizeKey.self, value: proxy.size)
      }
    }
    .onPreferenceChange(ExportPosterSizeKey.self) { posterSize = $0 }
    .onAppear(perform: compose)
    .onChange(of: time) { _, _ in compose() }
    .onChange(of: posterSize) { _, _ in compose() }
    .onChange(of: project.settings) { _, _ in compose() }
    .task(id: project.timeline) { compose() }
  }

  @ViewBuilder
  private var posterBackground: some View {
    if project.settings.background.isClear {
      H3Checkerboard(cell: 8)
    } else {
      Color(h3Hex: project.settings.background.rawValue)
    }
  }

  private func compose() {
    presenter.render(
      frame: ProgramPreview.frame(at: time, project: project),
      canvas: posterSize,
      scale: NSScreen.main?.backingScaleFactor ?? 2,
      background: project.settings.background,
      videoFrame: nil,
      layoutWidth: project.settings.width,
      layoutHeight: project.settings.height
    )
  }
}

private struct ExportPosterSizeKey: PreferenceKey {
  static let defaultValue: CGSize = .zero

  static func reduce(value: inout CGSize, nextValue: () -> CGSize) {
    value = nextValue()
  }
}

private struct ExportResultPlayer: View {
  let url: URL
  @Binding var player: AVPlayer?

  var body: some View {
    VideoPlayer(player: player)
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .onAppear {
        if player == nil {
          player = AVPlayer(url: url)
        }
      }
  }
}
