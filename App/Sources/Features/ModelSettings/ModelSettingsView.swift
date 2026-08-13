import H3ddleDesignSystem
import H3ddleEngineProtocol
import H3ddleModels
import SwiftUI
import UniformTypeIdentifiers

struct ModelSettingsView: View {
  @Bindable var model: AppModel

  @State private var isChoosingModel = false
  @State private var isConfirmingManagedDownload = false

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
      Divider().overlay(H3Color.line)
      ScrollView {
        VStack(spacing: 0) {
          status
          Divider().overlay(H3Color.line)
          managedPackage
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
        model.selectModelDirectory(directory)
      }
    }
    .confirmationDialog(
      "Download MiniMax H3 · INT8?",
      isPresented: $isConfirmingManagedDownload,
      titleVisibility: .visible
    ) {
      Button("Agree & Download 53.9 GB") {
        model.downloadManagedModel()
      }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text(
        "The files come from Hugging Face. By downloading them, you agree to the linked MiniMax H3 Community License Agreement."
      )
    }
  }

  private var managedPackage: some View {
    VStack(alignment: .leading, spacing: H3Spacing.medium) {
      HStack(alignment: .top, spacing: H3Spacing.medium) {
        managedDownloadSymbol
          .frame(width: 22, height: 22)
        VStack(alignment: .leading, spacing: 4) {
          HStack(alignment: .firstTextBaseline) {
            Text(model.managedModel.displayName)
              .font(.system(size: 13, weight: .semibold))
            Text("32 GB+")
              .font(.system(size: 9, weight: .bold))
              .tracking(0.6)
              .foregroundStyle(H3Color.accent)
              .padding(.horizontal, 7)
              .padding(.vertical, 3)
              .background(H3Color.accent.opacity(0.12), in: Capsule())
          }
          Text(model.managedModel.detail)
            .font(.system(size: 11))
            .foregroundStyle(H3Color.textSecondary)
          Text(model.managedModelStatusTitle)
            .font(.system(size: 12, weight: .semibold))
            .padding(.top, 3)
          Text(model.managedModelStatusMessage)
            .font(.system(size: 11))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        Spacer(minLength: 8)
        Text(
          ByteCountFormatter.string(
            fromByteCount: model.managedModel.totalByteCount,
            countStyle: .file
          )
        )
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
      }

      if model.managedModelProgress > 0 || model.managedModelDownloadIsActive {
        VStack(spacing: 6) {
          ProgressView(value: model.managedModelProgress)
            .tint(H3Color.accent)
          HStack {
            Text(
              ByteCountFormatter.string(
                fromByteCount: model.managedModelCompletedBytes,
                countStyle: .file
              )
            )
            Spacer()
            Text(model.managedModelProgress.formatted(.percent.precision(.fractionLength(1))))
          }
          .font(.system(size: 10, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
        }
      }

      HStack {
        Link("Model license", destination: model.managedModel.licenseURL)
          .font(.system(size: 11, weight: .medium))
        Spacer()
        if model.managedModelDownloadIsActive {
          Button("Pause") {
            model.cancelManagedModelDownload()
          }
          .buttonStyle(H3QuietButtonStyle())
          .accessibilityIdentifier("cancel-model-download")
        } else if model.managedModelState != .installed {
          Button(model.managedModelState == .cancelled ? "Resume Download" : "Download Model") {
            isConfirmingManagedDownload = true
          }
          .buttonStyle(H3PrimaryButtonStyle())
          .disabled(model.managedModelState == .checking)
          .accessibilityIdentifier("download-managed-model")
        }
      }

      if model.managedModel.compatibility == .ready {
        Text(
          "Ready for prompt-only FL2VA video generation with embedded audio. The app uses a native 256×256, four-pass preview preset; visual conditioning is not available yet."
        )
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary.opacity(0.82))
        .fixedSize(horizontal: false, vertical: true)
      } else {
        Text(
          "This package can be downloaded and inspected, but this app build cannot generate with its checkpoint layout."
        )
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary.opacity(0.82))
        .fixedSize(horizontal: false, vertical: true)
      }
    }
    .padding(H3Spacing.large)
  }

  @ViewBuilder
  private var managedDownloadSymbol: some View {
    switch model.managedModelState {
    case .checking:
      ProgressView().controlSize(.small)
    case .available:
      Image(systemName: "arrow.down.circle")
        .foregroundStyle(H3Color.accent)
    case .downloading, .verifying, .installing:
      ProgressView().controlSize(.small).tint(H3Color.accent)
    case .installed:
      Image(systemName: "checkmark.circle.fill")
        .foregroundStyle(H3Color.accent)
    case .cancelled:
      Image(systemName: "pause.circle")
        .foregroundStyle(H3Color.textSecondary)
    case .failed:
      Image(systemName: "exclamationmark.triangle.fill")
        .foregroundStyle(H3Color.danger)
    }
  }

  private var header: some View {
    HStack(alignment: .top) {
      VStack(alignment: .leading, spacing: 4) {
        Text("LOCAL ENGINE")
          .font(.system(size: 10, weight: .bold))
          .tracking(1.1)
          .foregroundStyle(H3Color.accent)
        Text("MiniMax H3 model")
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
        if let modelDirectory = model.modelDirectory {
          Text(modelDirectory.path(percentEncoded: false))
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary.opacity(0.78))
            .lineLimit(2)
            .textSelection(.enabled)
        }
      }
      Spacer(minLength: 0)
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

  @ViewBuilder
  private var details: some View {
    if let report = model.modelReport, let capabilities = model.engineCapabilities {
      VStack(spacing: 0) {
        DetailRow(
          label: "Engine", value: "\(capabilities.engineName) \(capabilities.engineVersion)")
        DetailRow(label: "Device", value: report.device.name)
        DetailRow(
          label: "Model files",
          value: ByteCountFormatter.string(
            fromByteCount: Int64(report.totalBytes), countStyle: .file))
        DetailRow(
          label: "Format",
          value: report.format.displayName
        )
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
      }
      .padding(.horizontal, H3Spacing.large)
      .transition(.opacity.combined(with: .move(edge: .top)))
    } else {
      VStack(alignment: .leading, spacing: 8) {
        Text("Use an existing model")
          .font(.system(size: 12, weight: .semibold))
        Text(
          "Select the directory containing FL2VA. Ref2VA may be installed alongside it for reference-conditioned generation."
        )
        .font(.system(size: 12))
        .foregroundStyle(H3Color.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
      }
      .padding(H3Spacing.large)
      .transition(.opacity)
    }
  }

  private var footer: some View {
    HStack {
      if model.modelDirectory != nil {
        Button("Forget") {
          model.clearModelDirectory()
        }
        .buttonStyle(H3QuietButtonStyle())

        Button("Check again") {
          model.validateSelectedModel()
        }
        .buttonStyle(H3QuietButtonStyle())
        .disabled(model.modelValidationState == .validating)
      }
      Spacer()
      Button(model.modelDirectory == nil ? "Choose Model Folder" : "Choose Another Folder") {
        isChoosingModel = true
      }
      .buttonStyle(H3PrimaryButtonStyle())
      .disabled(model.modelValidationState == .validating)
      .accessibilityIdentifier("choose-model-folder")
    }
    .padding(.horizontal, H3Spacing.large)
    .frame(height: 62)
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
    }
    .font(.system(size: 12, weight: .medium))
    .frame(height: 35)
    .overlay(alignment: .bottom) {
      Divider().overlay(H3Color.line.opacity(0.65))
    }
  }
}
