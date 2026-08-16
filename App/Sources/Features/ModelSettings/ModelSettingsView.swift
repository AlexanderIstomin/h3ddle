import H3ddleDesignSystem
import H3ddleEngineProtocol
import H3ddleModels
import SwiftUI
import UniformTypeIdentifiers

struct ModelSettingsView: View {
  @Bindable var model: AppModel

  @State private var isChoosingModel = false
  @State private var manifestPendingDownload: ModelPackageManifest?
  @State private var detailsExpanded = false

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
      Divider().overlay(H3Color.line)
      ScrollView {
        VStack(alignment: .leading, spacing: 0) {
          modelList
          Divider().overlay(H3Color.line)
          status
          Divider().overlay(H3Color.line)
          details
        }
      }
      Divider().overlay(H3Color.line)
      footer
    }
    .frame(width: 610, height: 680)
    .background(H3Color.surface)
    .foregroundStyle(H3Color.textPrimary)
    .fileImporter(
      isPresented: $isChoosingModel,
      allowedContentTypes: [.folder],
      allowsMultipleSelection: false
    ) { result in
      if case .success(let urls) = result, let directory = urls.first {
        model.addLocalModelFolder(directory)
      }
    }
    .confirmationDialog(
      "Get \(manifestPendingDownload?.displayName ?? "model")?",
      isPresented: Binding(
        get: { manifestPendingDownload != nil },
        set: { if !$0 { manifestPendingDownload = nil } }
      ),
      titleVisibility: .visible
    ) {
      if let manifest = manifestPendingDownload {
        Button(confirmActionTitle(for: manifest)) {
          model.downloadManagedModel(manifest)
          manifestPendingDownload = nil
        }
      }
      Button("Cancel", role: .cancel) {}
    } message: {
      if let manifest = manifestPendingDownload {
        Text(licenseMessage(for: manifest))
      }
    }
  }

  /// Each package carries its own licence, so the agreement being accepted
  /// has to be named from the manifest rather than assumed to be H3's.
  private func licenseMessage(for manifest: ModelPackageManifest) -> String {
    let reuse = manifest.generationProfile == .turbo
      ? "Files already on this Mac are reused; only missing files are downloaded. "
      : "The files come from Hugging Face. "
    return reuse + "By installing you agree to the linked \(manifest.licenseName)."
  }

  private func confirmActionTitle(for manifest: ModelPackageManifest) -> String {
    manifest.generationProfile == .turbo
      ? "Agree & Install"
      : "Agree & Download "
        + ByteCountFormatter.string(
          fromByteCount: model.pendingDownloadBytes(for: manifest), countStyle: .file)
  }

  private var header: some View {
    HStack(alignment: .top) {
      VStack(alignment: .leading, spacing: 4) {
        Text("LOCAL ENGINE")
          .font(.system(size: 10, weight: .bold))
          .tracking(1.1)
          .foregroundStyle(H3Color.accent)
        Text("Models")
          .font(.system(size: 18, weight: .semibold))
          .accessibilityIdentifier("model-settings")
      }
      Spacer()
      Button {
        model.showsModelSettings = false
      } label: {
        Image(systemName: "xmark")
      }
      .buttonStyle(.plain)
    }
    .padding(H3Spacing.large)
  }

  private var modelList: some View {
    VStack(alignment: .leading, spacing: H3Spacing.large) {
      // A category with nothing in it is not shown: an empty heading reads
      // like something failed to load.
      ForEach(ModelCapability.allCases, id: \.self) { capability in
        let choices = model.modelChoices.filter { $0.capability == capability }
        if !choices.isEmpty {
          VStack(alignment: .leading, spacing: H3Spacing.small) {
            Text(capability.sectionTitle.uppercased())
              .font(.system(size: 10, weight: .bold))
              .tracking(1.1)
              .foregroundStyle(H3Color.textSecondary)
            ForEach(choices) { choice in
              ModelChoiceRow(
                choice: choice,
                isSelected: model.selectedModelID(for: capability) == choice.id,
                status: managedStatus(for: choice),
                installedMemoryBytes: Int64(ProcessInfo.processInfo.physicalMemory),
                select: { model.selectModel(choice.id) },
                install: {
                  if case .managed(let manifest) = choice.source {
                    manifestPendingDownload = manifest
                  }
                },
                pause: { model.cancelManagedModelDownload() },
                discard: {
                  if case .managed(let manifest) = choice.source {
                    model.discardManagedModelDownload(manifest)
                  }
                },
                remove: choice.isLocalFolder ? { model.removeLocalModel(choice.id) } : nil
              )
            }
          }
        }
      }

      Button {
        isChoosingModel = true
      } label: {
        HStack(spacing: 8) {
          Image(systemName: "plus")
            .font(.system(size: 11, weight: .semibold))
          Text("Add Local Folder")
            .font(.system(size: 13, weight: .semibold))
          Spacer()
        }
        .padding(.horizontal, 12)
        .frame(height: 40)
        .background(H3Color.canvas, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
          RoundedRectangle(cornerRadius: 10, style: .continuous)
            .strokeBorder(H3Color.line, style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
        }
      }
      .buttonStyle(.plain)
      .accessibilityIdentifier("choose-model-folder")
    }
    .padding(H3Spacing.large)
  }

  private func managedStatus(for choice: ModelChoice) -> ManagedPackageStatus? {
    if case .managed(let manifest) = choice.source {
      return model.status(for: manifest)
    }
    return nil
  }

  private var status: some View {
    HStack(alignment: .top, spacing: H3Spacing.medium) {
      statusSymbol
        .frame(width: 22, height: 22)
      VStack(alignment: .leading, spacing: 5) {
        Text(model.modelStatusTitle)
          .font(.system(size: 13, weight: .semibold))
        Text(model.modelValidationMessage)
          .font(.system(size: 12))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      Spacer(minLength: 0)
      if model.modelDirectory != nil {
        Button("Check again") {
          model.validateSelectedModel()
        }
        .buttonStyle(H3QuietButtonStyle())
        .disabled(model.modelValidationState == .validating)
      }
    }
    .padding(H3Spacing.large)
    .animation(.easeOut(duration: 0.18), value: model.modelValidationState)
  }

  @ViewBuilder
  private var statusSymbol: some View {
    switch model.modelValidationState {
    case .notSelected:
      Image(systemName: "externaldrive")
        .foregroundStyle(H3Color.textSecondary)
    case .validating:
      ProgressView()
        .controlSize(.small)
        .tint(H3Color.accent)
    case .ready:
      Image(systemName: "checkmark.circle.fill")
        .foregroundStyle(H3Color.accent)
        .symbolEffect(.bounce, value: model.modelValidationState)
    case .failed:
      Image(systemName: "exclamationmark.triangle.fill")
        .foregroundStyle(H3Color.danger)
    }
  }

  private var details: some View {
    DisclosureGroup(isExpanded: $detailsExpanded) {
      VStack(spacing: 0) {
        if let report = model.modelReport, let capabilities = model.engineCapabilities {
          DetailRow(
            label: "Engine", value: "\(capabilities.engineName) \(capabilities.engineVersion)")
          DetailRow(label: "Device", value: report.device.name)
          DetailRow(
            label: "Model files",
            value: ByteCountFormatter.string(
              fromByteCount: Int64(report.totalBytes), countStyle: .file))
          DetailRow(label: "Format", value: report.format.displayName)
          DetailRow(
            label: "Reference model",
            value: report.hasReferenceTransformer ? "Available" : "Not installed"
          )
          DetailRow(
            label: "Video output",
            value: capabilities.supports(.videoGeneration) && report.supportsGeneration
              ? "Available" : "Adapter required")
          DetailRow(
            label: "Audio-only output",
            value: capabilities.supports(.standaloneAudioGeneration)
              ? "Available" : "Requires a separate provider"
          )
          if let directory = model.modelDirectory {
            DetailRow(label: "Location", value: directory.path(percentEncoded: false))
          }
        } else {
          Text("Select a ready model to inspect engine and device details.")
            .font(.system(size: 12))
            .foregroundStyle(H3Color.textSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 10)
        }
      }
    } label: {
      Text("Details")
        .font(.system(size: 13, weight: .semibold))
    }
    .tint(H3Color.textSecondary)
    .padding(H3Spacing.large)
  }

  private var footer: some View {
    HStack {
      if let manifest = selectedManifest {
        Link("Model license", destination: manifest.licenseURL)
          .font(.system(size: 11, weight: .medium))
      }
      Spacer()
    }
    .padding(.horizontal, H3Spacing.large)
    .frame(height: 50)
  }

  private var selectedManifest: ModelPackageManifest? {
    if case .managed(let manifest) = model.selectedModelChoice?.source {
      return manifest
    }
    return nil
  }
}

/// One selectable model card: name and subtitle on the left, install state or
/// selection tick on the right, mirroring the studio dropdown rows.
private struct ModelChoiceRow: View {
  var choice: ModelChoice
  var isSelected: Bool
  var status: ManagedPackageStatus?
  /// This Mac's unified memory, compared against what the package asks for.
  var installedMemoryBytes: Int64
  var select: () -> Void
  var install: () -> Void
  var pause: () -> Void
  var discard: (() -> Void)?
  var remove: (() -> Void)?

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(alignment: .top, spacing: 10) {
        VStack(alignment: .leading, spacing: 4) {
          HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(choice.displayName)
              .font(.system(size: 13, weight: .semibold))
            Spacer(minLength: 0)
            trailingBadge
          }
          Text(choice.subtitle)
            .font(.system(size: 11))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
          if !chips.isEmpty {
            HStack(spacing: 6) {
              ForEach(chips, id: \.text) { chip in
                Text(chip.text)
                  .font(.system(size: 10, weight: .medium))
                  .padding(.horizontal, 6)
                  .padding(.vertical, 2)
                  .background(
                    chip.warning ? H3Color.danger.opacity(0.16) : H3Color.controlFill,
                    in: Capsule()
                  )
                  .foregroundStyle(chip.warning ? H3Color.danger : H3Color.textSecondary)
              }
            }
            .padding(.top, 2)
          }
        }

        actions
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 10)

      if let status, status.downloadIsActive || (status.progress > 0 && status.state != .installed)
      {
        VStack(spacing: 4) {
          ProgressView(value: status.progress)
            .tint(H3Color.accent)
          HStack {
            Text(status.message)
            Spacer()
            Text("\(Int(status.progress * 100))%")
          }
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 10)
      }
    }
    .contentShape(Rectangle())
    .onTapGesture(perform: selectIfPossible)
    .background(
      isSelected ? H3Color.accent.opacity(0.12) : H3Color.canvas,
      in: RoundedRectangle(cornerRadius: H3Radius.medium, style: .continuous)
    )
    .overlay {
      RoundedRectangle(cornerRadius: H3Radius.medium, style: .continuous)
        .strokeBorder(isSelected ? H3Color.accent.opacity(0.5) : H3Color.line, lineWidth: 1)
    }
    .animation(.easeOut(duration: 0.18), value: isSelected)
  }

  /// Installed packages confirm themselves; everything else shows its price,
  /// so the cost is visible right up until it is paid.
  @ViewBuilder
  private var trailingBadge: some View {
    if choice.isInstalled {
      Text("Installed")
        .font(.system(size: 10, weight: .semibold))
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(H3Color.accent.opacity(0.18), in: Capsule())
        .foregroundStyle(H3Color.accent)
    } else if choice.downloadBytes > 0 {
      Text(byteText(choice.downloadBytes))
        .font(.system(size: 11, weight: .medium))
        .foregroundStyle(H3Color.textSecondary)
    }
  }

  private struct Chip {
    var text: String
    var warning = false
  }

  private var chips: [Chip] {
    var result: [Chip] = []
    if choice.generationProfile == .turbo {
      result.append(Chip(text: "Turbo"))
    }
    // Disk is the real cost of a package: the weights stream from the file
    // rather than being held in RAM, so what it occupies is the drive.
    if choice.isInstalled, choice.installedBytes > 0 {
      result.append(Chip(text: "\(byteText(choice.installedBytes)) on disk"))
    }
    // Memory only earns a chip when this Mac is short of it; otherwise it
    // is a number nobody has to act on.
    if choice.requiredMemoryBytes > 0, installedMemoryBytes > 0,
      installedMemoryBytes < choice.requiredMemoryBytes
    {
      result.append(
        Chip(text: "Needs \(byteText(choice.requiredMemoryBytes)) memory", warning: true)
      )
    }
    if choice.isLocalFolder {
      result.append(Chip(text: "Local folder"))
    }
    return result
  }

  private func byteText(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
  }

  private func selectIfPossible() {
    if choice.isInstalled {
      select()
    }
  }

  @ViewBuilder
  private var actions: some View {
    if choice.isInstalled {
      if let remove {
        Button(action: remove) {
          Image(systemName: "xmark.circle")
            .foregroundStyle(H3Color.textSecondary)
        }
        .buttonStyle(.plain)
        .help("Remove from the list (files stay on disk)")
      }
    } else if let status {
      switch status.state {
      case .checking:
        ProgressView().controlSize(.small)
      case .downloading, .verifying, .installing:
        Button("Pause", action: pause)
          .buttonStyle(H3QuietButtonStyle())
      case .available, .cancelled, .failed, .installed:
        // A paused download keeps its partial files so resuming is free;
        // discarding is the explicit way to get that disk back.
        if status.state == .cancelled || status.progress > 0, let discard {
          Button(action: discard) {
            Image(systemName: "trash")
              .foregroundStyle(H3Color.textSecondary)
          }
          .buttonStyle(.plain)
          .help("Discard the paused download and free its partial files")
          .accessibilityIdentifier("discard-managed-download")
        }
        Button(installTitle, action: install)
          .buttonStyle(H3PrimaryButtonStyle())
          .accessibilityIdentifier("download-managed-model")
      }
    }
  }

  private var installTitle: String {
    if status?.state == .cancelled {
      return "Resume"
    }
    return choice.generationProfile == .turbo ? "Install" : "Download"
  }
}

private extension EngineModelFormat {
  var displayName: String {
    switch self {
    case .unknown: "Unknown"
    case .releasedDirectory: "Released directory"
    case .optimizedINT8SingleFile: "Optimized INT8"
    }
  }
}

private struct DetailRow: View {
  var label: String
  var value: String

  var body: some View {
    HStack(alignment: .firstTextBaseline) {
      Text(label)
        .foregroundStyle(H3Color.textSecondary)
      Spacer()
      Text(value)
        .multilineTextAlignment(.trailing)
        .textSelection(.enabled)
    }
    .font(.system(size: 12, weight: .medium))
    .frame(minHeight: 35)
    .overlay(alignment: .bottom) {
      Divider().overlay(H3Color.line.opacity(0.65))
    }
  }
}
