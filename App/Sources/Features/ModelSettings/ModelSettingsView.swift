import H3ddleDesignSystem
import H3ddleModels
import SwiftUI
import UniformTypeIdentifiers

struct ModelSettingsView: View {
  @Bindable var model: AppModel

  @State private var isChoosingModel = false
  /// Which list the folder about to be chosen should join.
  @State private var capabilityBeingAdded: ModelCapability = .video
  @State private var manifestPendingDownload: ModelPackageManifest?
  @State private var manifestPendingRemoval: ModelPackageManifest?

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
      Divider().overlay(H3Color.line)
      ScrollView {
        modelList
      }
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
        model.addLocalModelFolder(directory, capability: capabilityBeingAdded)
      }
    }
    .confirmationDialog(
      "Delete \(manifestPendingRemoval?.displayName ?? "model")?",
      isPresented: Binding(
        get: { manifestPendingRemoval != nil },
        set: { if !$0 { manifestPendingRemoval = nil } }
      ),
      titleVisibility: .visible
    ) {
      if let manifest = manifestPendingRemoval {
        Button("Delete Files", role: .destructive) {
          model.removeManagedModel(manifest)
          manifestPendingRemoval = nil
        }
      }
      Button("Cancel", role: .cancel) {}
    } message: {
      if let manifest = manifestPendingRemoval {
        Text(
          "Removes \(ByteCountFormatter.string(fromByteCount: manifest.totalByteCount, countStyle: .file)) "
            + "from this Mac. Weights shared with another installed package are kept."
        )
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
      // Every category is shown even when empty: the section carries its own
      // way to add a folder, so an empty one is an invitation rather than a
      // sign something failed to load.
      ForEach(ModelCapability.allCases, id: \.self) { capability in
        let choices = model.modelChoices.filter { $0.capability == capability }
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
          addFolderButton(for: capability)
        }
      }
    }
    .padding(H3Spacing.large)
  }

  private func addFolderButton(for capability: ModelCapability) -> some View {
    Button {
      capabilityBeingAdded = capability
      isChoosingModel = true
    } label: {
      HStack(spacing: 8) {
        Image(systemName: "plus")
          .font(.system(size: 11, weight: .semibold))
        Text("Add \(capability.sectionTitle.lowercased()) model from this Mac")
          .font(.system(size: 12, weight: .medium))
        Spacer()
      }
      .foregroundStyle(H3Color.textSecondary)
      .padding(.horizontal, 12)
      .frame(height: 36)
      .background(
        H3Color.canvas,
        in: RoundedRectangle(cornerRadius: H3Radius.medium, style: .continuous)
      )
      .overlay {
        RoundedRectangle(cornerRadius: H3Radius.medium, style: .continuous)
          .strokeBorder(H3Color.line, style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
      }
    }
    .buttonStyle(.plain)
    .accessibilityIdentifier("choose-model-folder-\(capability.rawValue)")
  }

  /// Names whichever package in the same category is already downloading.
  private func blockingPackageName(for choice: ModelChoice) -> String? {
    guard case .managed(let manifest) = choice.source else { return nil }
    return model.packageBlockingDownload(of: manifest)?.displayName
  }

  private func managedStatus(for choice: ModelChoice) -> ManagedPackageStatus? {
    if case .managed(let manifest) = choice.source {
      return model.status(for: manifest)
    }
    return nil
  }

}

/// One model in the library: what it is, what it costs, and the licence it
/// arrives under, which differs per package and so belongs on the card
/// rather than in a footer that could only ever name one of them.
private struct ModelChoiceRow: View {
  var choice: ModelChoice
  var isSelected: Bool
  var status: ManagedPackageStatus?
  /// Name of the package already downloading in this category, which this
  /// one has to wait for.
  var blockedBy: String?
  /// This Mac's unified memory, compared against what the package asks for.
  var installedMemoryBytes: Int64
  var select: () -> Void
  var install: () -> Void
  var pause: () -> Void
  var discard: (() -> Void)?
  var remove: (() -> Void)?  // trash for managed, unlist for local

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
      .padding(.top, 10)
      .padding(.bottom, choice.licenseURL == nil ? 10 : 0)

      if let licenseURL = choice.licenseURL {
        Link(choice.licenseName ?? "Model license", destination: licenseURL)
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .padding(.horizontal, 12)
          .padding(.bottom, 10)
      }

      if let blockedBy {
        Text("Waiting on \(blockedBy): they share weights, and fetching "
          + "both at once would download the shared files twice.")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
          .frame(maxWidth: .infinity, alignment: .leading)
          .padding(.horizontal, 12)
          .padding(.bottom, 10)
      } else if let status,
        status.downloadIsActive || (status.progress > 0 && status.state != .installed)
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
      H3Color.canvas,
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
      Text(isSelected ? "In use" : "Installed")
        .font(.system(size: 10, weight: .semibold))
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(
          isSelected ? H3Color.accent.opacity(0.18) : H3Color.controlFill,
          in: Capsule()
        )
        .foregroundStyle(isSelected ? H3Color.accent : H3Color.textSecondary)
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
          Image(systemName: choice.isLocalFolder ? "xmark.circle" : "trash")
            .foregroundStyle(H3Color.textSecondary)
        }
        .buttonStyle(.plain)
        .help(
          choice.isLocalFolder
            ? "Remove from the list (files stay on disk)"
            : "Delete the downloaded weights from this Mac"
        )
        .accessibilityIdentifier("remove-model")
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
          .disabled(blockedBy != nil)
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
