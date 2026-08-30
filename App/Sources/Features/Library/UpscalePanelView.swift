import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct UpscalePanelView: View {
  @Bindable var model: AppModel

  private var selectedAsset: AssetReference? {
    if let id = model.selectedLibraryAssetID,
       let asset = model.project.asset(id: id),
       asset.kind.isVisual
    {
      return asset
    }
    guard case .visual(let id) = model.selectedTimelineItem else { return nil }
    return model.project.timeline.visualItems.first { $0.id == id }
      .flatMap { model.project.asset(id: $0.assetID) }
  }

  var body: some View {
    Group {
      if let asset = selectedAsset {
        ScrollView {
          VStack(alignment: .leading, spacing: 16) {
            assetSummary(asset)
            AssetUpscaleControls(model: model, asset: asset)
          }
          .padding(14)
        }
      } else {
        VStack(spacing: 11) {
          Image(systemName: "arrow.up.left.and.arrow.down.right")
            .font(.system(size: 28, weight: .medium))
            .foregroundStyle(H3Color.textSecondary.opacity(0.55))
          Text("Select an image or video to upscale")
            .font(.system(size: 12))
            .foregroundStyle(H3Color.textSecondary)
            .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(46)
      }
    }
    .accessibilityIdentifier("upscale-panel")
  }

  private func assetSummary(_ asset: AssetReference) -> some View {
    HStack(spacing: 10) {
      Image(systemName: asset.kind == .video ? "film" : "photo")
        .font(.system(size: 13, weight: .semibold))
        .foregroundStyle(H3Color.accent)
        .frame(width: 30, height: 30)
        .background(H3Color.accent.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
      VStack(alignment: .leading, spacing: 2) {
        Text(asset.displayName)
          .font(.system(size: 12, weight: .semibold))
          .foregroundStyle(H3Color.textPrimary)
          .lineLimit(1)
        Text(asset.kind == .video ? "Video · Processed locally" : "Image · Processed locally")
          .font(.system(size: 9))
          .foregroundStyle(H3Color.textSecondary)
      }
      Spacer()
    }
    .padding(.bottom, 14)
    .overlay(alignment: .bottom) {
      Rectangle().fill(H3Color.line).frame(height: 1)
    }
  }
}
