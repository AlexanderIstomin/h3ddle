import AppKit
import H3ddleDesignSystem
import H3ddleGeneration
import SwiftUI

struct GenerationQueueView: View {
  @Bindable var model: AppModel
  let queue: GenerationJobQueue
  @State private var pausedJobPendingEdit: GenerationQueueJob?

  private var queueJobs: [GenerationQueueJob] {
    queue.jobs.filter {
      $0.state != .completed && $0.state != .cancelled
    }
  }

  private var historyJobs: [GenerationQueueJob] {
    queue.jobs.reversed().filter {
      $0.state == .completed || $0.state == .cancelled
    }
  }

  private var hasWaitingJobs: Bool {
    queue.jobs.contains { $0.state == .queued && !$0.isScheduled }
  }

  var body: some View {
    VStack(spacing: 0) {
      header
      Divider().overlay(H3Color.line)
      queueProgress
      Divider().overlay(H3Color.line)

      ScrollView {
        LazyVStack(alignment: .leading, spacing: 12) {
          sectionTitle("QUEUE", count: queueJobs.count)
          if queueJobs.isEmpty {
            emptyState(
              icon: "text.line.first.and.arrowtriangle.forward",
              title: "Nothing waiting",
              detail: "Add a generation or open a completed job to run it again."
            )
          } else {
            ForEach(queueJobs) { job in
              GenerationQueueRow(
                job: job,
                activeJobID: model.activeQueueJobID,
                remaining: job.id == model.activeQueueJobID
                  ? model.generationRemainingDescription : nil,
                onEdit: { edit(job) },
                onRunNext: { model.runGenerationJobNext(job.id) },
                onPause: { model.pauseGenerationJob(job.id) },
                onCancel: { model.cancelGenerationJob(job.id) },
                onRetry: { model.retryGenerationJob(job.id) },
                onRemove: { model.removeGenerationJob(job.id) },
                onMoveUp: { model.moveGenerationJob(job.id, by: -1) },
                onMoveDown: { model.moveGenerationJob(job.id, by: 1) },
                onInsert: { insert(job) },
                onReveal: { reveal(job) }
              )
            }
          }

          if !historyJobs.isEmpty {
            HStack {
              sectionTitle("HISTORY", count: historyJobs.count)
              Spacer()
              Button("Clear") { model.clearCompletedGenerationJobs() }
                .buttonStyle(.plain)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(H3Color.textSecondary)
                .help("Remove completed and cancelled jobs from this list")
            }
            .padding(.top, 8)

            ForEach(historyJobs) { job in
              GenerationQueueRow(
                job: job,
                activeJobID: model.activeQueueJobID,
                remaining: nil,
                onEdit: { edit(job) },
                onRunNext: { model.runGenerationJobNext(job.id) },
                onPause: { model.pauseGenerationJob(job.id) },
                onCancel: { model.cancelGenerationJob(job.id) },
                onRetry: { model.retryGenerationJob(job.id) },
                onRemove: { model.removeGenerationJob(job.id) },
                onMoveUp: {},
                onMoveDown: {},
                onInsert: { insert(job) },
                onReveal: { reveal(job) }
              )
            }
          }
        }
        .padding(14)
      }
    }
    .frame(minWidth: 360, idealWidth: 390, maxWidth: 420)
    .background(H3Color.surface)
    .foregroundStyle(H3Color.textPrimary)
    .alert(item: $pausedJobPendingEdit) { job in
      Alert(
        title: Text("Edit paused generation?"),
        message: Text(
          "Changing its inputs or settings discards the saved denoising checkpoint."
        ),
        primaryButton: .destructive(Text("Discard Progress and Edit")) {
          model.editGenerationJob(job.id)
        },
        secondaryButton: .cancel()
      )
    }
  }

  private var header: some View {
    HStack(spacing: 10) {
      Image(systemName: "list.bullet.rectangle.portrait")
        .font(.system(size: 15, weight: .semibold))
        .foregroundStyle(H3Color.accent)
      Text("GENERATION QUEUE")
        .font(.system(size: 10, weight: .bold, design: .monospaced))
        .tracking(1.2)
        .accessibilityIdentifier("generation-queue")
      Spacer()
      Button {
        model.runAllGenerationJobs()
      } label: {
        Label("Run All", systemImage: "play.fill")
      }
      .buttonStyle(H3QuietButtonStyle())
      .disabled(!hasWaitingJobs)
      .help("Schedule every waiting job in its current order")
      .accessibilityIdentifier("generation-queue-run-all")
      Button {
        model.cancelAllGenerationJobs()
      } label: {
        Text("Cancel All")
      }
      .buttonStyle(H3QuietButtonStyle())
      .foregroundStyle(H3Color.danger)
      .disabled(!queue.hasCancellableJobs)
      .help("Cancel the current generation and every waiting or paused job")
      .accessibilityIdentifier("generation-queue-cancel-all")
      Button {
        model.showsGenerationQueue = false
      } label: {
        Image(systemName: "xmark")
      }
      .buttonStyle(H3IconButtonStyle(size: 30))
      .help("Close Queue")
      .accessibilityIdentifier("generation-queue-close")
    }
    .padding(.horizontal, 14)
    .frame(height: 50)
    .background(H3Color.chrome)
  }

  @ViewBuilder private var queueProgress: some View {
    if let position = model.generationQueuePosition {
      VStack(alignment: .leading, spacing: 7) {
        HStack {
          let current = min(position.total, position.completed + 1)
          Text("Job \(current) of \(position.total)")
            .font(.system(size: 11, weight: .semibold))
          Spacer()
          if let progress = model.generationQueueProgress {
            Text("\(Int((progress * 100).rounded()))%")
              .font(.system(size: 10, weight: .medium, design: .monospaced))
              .foregroundStyle(H3Color.textSecondary)
          } else {
            Text("Calculating")
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(H3Color.textSecondary)
          }
        }
        if let progress = model.generationQueueProgress {
          ProgressView(value: progress)
            .tint(H3Color.accent)
        } else {
          ProgressView()
            .controlSize(.small)
        }
      }
      .padding(.horizontal, 14)
      .padding(.vertical, 11)
      .background(H3Color.chrome.opacity(0.55))
    } else {
      HStack {
        Text(model.queuedGenerationCount == 0
          ? "Queue is ready"
          : "\(model.queuedGenerationCount) job\(model.queuedGenerationCount == 1 ? "" : "s")")
          .font(.system(size: 10, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
        Spacer()
      }
      .padding(.horizontal, 14)
      .frame(height: 38)
      .background(H3Color.chrome.opacity(0.55))
    }
  }

  private func sectionTitle(_ title: String, count: Int) -> some View {
    Text("\(title) · \(count)")
      .font(.system(size: 9, weight: .bold, design: .monospaced))
      .tracking(1.4)
      .foregroundStyle(H3Color.textSecondary.opacity(0.75))
  }

  private func emptyState(icon: String, title: String, detail: String) -> some View {
    VStack(spacing: 8) {
      Image(systemName: icon)
        .font(.system(size: 22))
        .foregroundStyle(H3Color.textSecondary.opacity(0.6))
      Text(title).font(.system(size: 12, weight: .semibold))
      Text(detail)
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
        .multilineTextAlignment(.center)
    }
    .frame(maxWidth: .infinity)
    .padding(.vertical, 26)
  }

  private func edit(_ job: GenerationQueueJob) {
    if job.state == .paused {
      pausedJobPendingEdit = job
    } else {
      model.editGenerationJob(job.id)
    }
  }

  private func insert(_ job: GenerationQueueJob) {
    guard let asset = job.result else { return }
    model.insertToTimeline(
      GenerationResult(
        id: UUID(),
        asset: asset,
        kind: job.request.kind,
        prompt: job.displayPrompt,
        createdAt: job.finishedAt ?? Date()
      )
    )
  }

  private func reveal(_ job: GenerationQueueJob) {
    guard let url = job.result?.url else { return }
    NSWorkspace.shared.activateFileViewerSelecting([url])
  }
}

private struct GenerationQueueRow: View {
  let job: GenerationQueueJob
  let activeJobID: UUID?
  let remaining: String?
  let onEdit: () -> Void
  let onRunNext: () -> Void
  let onPause: () -> Void
  let onCancel: () -> Void
  let onRetry: () -> Void
  let onRemove: () -> Void
  let onMoveUp: () -> Void
  let onMoveDown: () -> Void
  let onInsert: () -> Void
  let onReveal: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack(alignment: .top, spacing: 10) {
        Image(systemName: kindIcon)
          .font(.system(size: 14, weight: .medium))
          .foregroundStyle(stateColor)
          .frame(width: 22, height: 22)
          .background(stateColor.opacity(0.12))
          .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))

        VStack(alignment: .leading, spacing: 4) {
          Text(job.displayPrompt)
            .font(.system(size: 11, weight: .semibold))
            .lineLimit(2)
          Text(job.settingsDescription)
            .font(.system(size: 9, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary)
            .lineLimit(2)
        }
        Spacer(minLength: 4)
        Text(stateLabel)
          .font(.system(size: 8.5, weight: .bold, design: .monospaced))
          .foregroundStyle(stateColor)
          .padding(.horizontal, 7)
          .frame(height: 20)
          .background(stateColor.opacity(0.11))
          .clipShape(Capsule())
      }

      if job.state.isActive {
        VStack(alignment: .leading, spacing: 5) {
          ProgressView(value: job.overallProgress)
            .tint(H3Color.accent)
          HStack {
            Text(job.phase)
              .lineLimit(1)
            Spacer()
            if let remaining { Text(remaining) }
          }
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
        }
      } else if let error = job.errorMessage {
        Text(error)
          .font(.system(size: 9.5, weight: .medium))
          .foregroundStyle(H3Color.danger)
          .fixedSize(horizontal: false, vertical: true)
      }

      actions
    }
    .padding(11)
    .background(H3Color.chrome.opacity(job.id == activeJobID ? 0.95 : 0.62))
    .overlay {
      RoundedRectangle(cornerRadius: 10, style: .continuous)
        .stroke(job.id == activeJobID ? H3Color.accent.opacity(0.7) : H3Color.line, lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    .accessibilityIdentifier("generation-job-\(job.id.uuidString)")
  }

  @ViewBuilder private var actions: some View {
    HStack(spacing: 7) {
      switch job.state {
      case .queued:
        Button("Edit", action: onEdit)
        Button(job.isScheduled ? "Run Next" : (activeJobID == nil ? "Run Now" : "Run Next"),
          action: onRunNext)
        Spacer()
        Button(action: onMoveUp) { Image(systemName: "chevron.up") }
          .help("Move up")
        Button(action: onMoveDown) { Image(systemName: "chevron.down") }
          .help("Move down")
        Button(action: onRemove) { Image(systemName: "trash") }
          .help("Remove job")
      case .preparing, .running:
        if job.supportsPause {
          Button("Pause", action: onPause)
            .accessibilityIdentifier("generation-job-pause")
        } else {
          Button("Cancel", action: onCancel)
            .accessibilityIdentifier("generation-job-cancel")
        }
        Spacer()
      case .paused:
        Button("Resume", action: onRunNext)
        Button("Edit", action: onEdit)
        Spacer()
        Button(action: onRemove) { Image(systemName: "trash") }
      case .blocked, .failed:
        Button("Retry", action: onRetry)
        Button("Edit", action: onEdit)
        Spacer()
        Button(action: onRemove) { Image(systemName: "trash") }
      case .completed:
        Button("Edit & Run Again", action: onEdit)
        Spacer()
        if job.result != nil {
          Button(action: onReveal) { Image(systemName: "folder") }
            .help("Show in Finder")
          Button("Insert", action: onInsert)
        }
      case .cancelled:
        Button("Retry", action: onRetry)
        Button("Edit", action: onEdit)
        Spacer()
        Button(action: onRemove) { Image(systemName: "trash") }
      }
    }
    .buttonStyle(H3QuietButtonStyle())
    .controlSize(.small)
  }

  private var kindIcon: String {
    switch job.request.kind {
    case .image: "photo"
    case .video: "film"
    case .audio: "waveform"
    }
  }

  private var stateLabel: String {
    switch job.state {
    case .queued: job.isScheduled ? "WAITING" : "QUEUED"
    case .preparing: "PREPARING"
    case .running: "RUNNING"
    case .paused: "PAUSED"
    case .blocked: "BLOCKED"
    case .failed: "FAILED"
    case .completed: "DONE"
    case .cancelled: "CANCELLED"
    }
  }

  private var stateColor: Color {
    switch job.state {
    case .preparing, .running: H3Color.accent
    case .failed, .blocked: H3Color.danger
    case .completed: Color.green
    case .paused: Color.orange
    case .queued, .cancelled: H3Color.textSecondary
    }
  }
}
