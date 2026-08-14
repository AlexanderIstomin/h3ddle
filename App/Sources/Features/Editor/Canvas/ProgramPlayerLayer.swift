import AVFoundation
import AppKit
import SwiftUI

struct ProgramPlayerLayer: NSViewRepresentable {
  let player: AVPlayer
  var videoGravity: AVLayerVideoGravity = .resize

  func makeNSView(context: Context) -> PlayerHostView {
    let view = PlayerHostView()
    view.playerLayer.player = player
    view.playerLayer.videoGravity = videoGravity
    view.playerLayer.isOpaque = false
    view.playerLayer.backgroundColor = CGColor.clear
    return view
  }

  func updateNSView(_ view: PlayerHostView, context: Context) {
    if view.playerLayer.player !== player {
      view.playerLayer.player = player
    }
    view.playerLayer.videoGravity = videoGravity
    view.playerLayer.isOpaque = false
    view.playerLayer.backgroundColor = CGColor.clear
  }
}

final class PlayerHostView: NSView {
  override init(frame frameRect: NSRect) {
    super.init(frame: frameRect)
    wantsLayer = true
    layer = AVPlayerLayer()
  }

  @available(*, unavailable)
  required init?(coder: NSCoder) {
    fatalError("init(coder:) is unused")
  }

  var playerLayer: AVPlayerLayer {
    layer as! AVPlayerLayer
  }

  override func layout() {
    super.layout()
    playerLayer.frame = bounds
  }

  override func hitTest(_ point: NSPoint) -> NSView? {
    nil
  }
}
