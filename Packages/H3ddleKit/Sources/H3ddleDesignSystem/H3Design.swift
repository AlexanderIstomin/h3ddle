import Foundation
import SwiftUI

public enum H3Color {
  public static let canvas = Color(red: 13 / 255, green: 15 / 255, blue: 19 / 255)
  public static let gradientTop = Color(red: 26 / 255, green: 30 / 255, blue: 37 / 255)
  public static let surface = Color(red: 23 / 255, green: 26 / 255, blue: 32 / 255)
  public static let surfaceRaised = Color(red: 23 / 255, green: 26 / 255, blue: 32 / 255)
  public static let chrome = Color(red: 17 / 255, green: 20 / 255, blue: 26 / 255)
  public static let transport = Color(red: 31 / 255, green: 36 / 255, blue: 44 / 255)
  public static let line = Color.white.opacity(0.12)
  public static let hairSoft = Color.white.opacity(0.065)
  public static let textPrimary = Color(red: 236 / 255, green: 238 / 255, blue: 243 / 255)
  public static let textSecondary = Color.white.opacity(0.56)
  public static let accent = Color(red: 224 / 255, green: 82 / 255, blue: 28 / 255)
  public static let danger = Color(red: 229 / 255, green: 72 / 255, blue: 77 / 255)
  public static let clipVideo = Color(red: 91 / 255, green: 134 / 255, blue: 201 / 255)
  public static let clipAudio = Color(red: 70 / 255, green: 168 / 255, blue: 131 / 255)
  public static let tickMajor = Color.white.opacity(0.42)
  public static let tickMinor = Color.white.opacity(0.18)
  public static let controlFill = Color.white.opacity(0.07)
  public static let controlHover = Color.white.opacity(0.09)
}

public enum H3Spacing {
  public static let xSmall: CGFloat = 6
  public static let small: CGFloat = 10
  public static let medium: CGFloat = 16
  public static let large: CGFloat = 24
  public static let xLarge: CGFloat = 32
}

public enum H3Radius {
  public static let small: CGFloat = 6
  public static let medium: CGFloat = 10
  public static let large: CGFloat = 16
}

public struct H3PrimaryButtonStyle: ButtonStyle {
  public init() {}

  public func makeBody(configuration: Configuration) -> some View {
    configuration.label
      .font(.system(size: 13, weight: .semibold))
      .foregroundStyle(Color.white)
      .padding(.horizontal, 14)
      .frame(height: 32)
      .background(H3Color.accent.opacity(configuration.isPressed ? 0.72 : 1))
      .clipShape(RoundedRectangle(cornerRadius: H3Radius.small, style: .continuous))
      .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
  }
}

public struct H3QuietButtonStyle: ButtonStyle {
  public init() {}

  public func makeBody(configuration: Configuration) -> some View {
    configuration.label
      .font(.system(size: 12, weight: .medium))
      .foregroundStyle(H3Color.textPrimary)
      .padding(.horizontal, 10)
      .frame(height: 30)
      .background(configuration.isPressed ? H3Color.controlHover : H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: H3Radius.small, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: H3Radius.small, style: .continuous))
  }
}

public struct H3IconButtonStyle: ButtonStyle {
  public var isActive: Bool
  public var size: CGFloat

  public init(isActive: Bool = false, size: CGFloat = 36) {
    self.isActive = isActive
    self.size = size
  }

  public func makeBody(configuration: Configuration) -> some View {
    configuration.label
      .font(.system(size: 14, weight: .medium))
      .foregroundStyle(isActive ? H3Color.accent : H3Color.textSecondary)
      .frame(width: size, height: size)
      .background(
        isActive
          ? H3Color.accent.opacity(0.16)
          : (configuration.isPressed ? H3Color.controlHover : Color.clear)
      )
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
  }
}

public struct H3Checkerboard: View {
  public var cell: CGFloat

  public init(cell: CGFloat = 6) {
    self.cell = cell
  }

  public var body: some View {
    Canvas { context, size in
      var row = 0
      var y: CGFloat = 0
      while y < size.height {
        var col = 0
        var x: CGFloat = 0
        while x < size.width {
          context.fill(
            Path(CGRect(x: x, y: y, width: cell, height: cell)),
            with: .color((row + col).isMultiple(of: 2) ? .white : Color(white: 0.73))
          )
          x += cell
          col += 1
        }
        y += cell
        row += 1
      }
    }
  }
}

public extension Color {
  init(h3Hex: String) {
    let cleaned = h3Hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
    var value: UInt64 = 0
    Scanner(string: cleaned).scanHexInt64(&value)
    self.init(
      red: Double((value >> 16) & 0xFF) / 255,
      green: Double((value >> 8) & 0xFF) / 255,
      blue: Double(value & 0xFF) / 255
    )
  }
}

/// A compact segmented control: one filled segment for the active choice and
/// quiet ones either side, sized for a panel header rather than a form.
public struct H3SegmentedControl<Value: Hashable>: View {
  public struct Segment: Identifiable {
    public var value: Value
    public var title: String
    public var systemImage: String?

    public var id: Value { value }

    public init(value: Value, title: String, systemImage: String? = nil) {
      self.value = value
      self.title = title
      self.systemImage = systemImage
    }
  }

  @Binding private var selection: Value
  private let segments: [Segment]
  private let isEnabled: Bool

  public init(selection: Binding<Value>, segments: [Segment], isEnabled: Bool = true) {
    self._selection = selection
    self.segments = segments
    self.isEnabled = isEnabled
  }

  public var body: some View {
    HStack(spacing: 3) {
      ForEach(segments) { segment in
        let active = segment.value == selection
        Button {
          selection = segment.value
        } label: {
          HStack(spacing: 6) {
            if let systemImage = segment.systemImage {
              Image(systemName: systemImage)
                .font(.system(size: 11))
            }
            Text(segment.title)
              .font(.system(size: 10, weight: .semibold, design: .monospaced))
          }
          .padding(.horizontal, 12)
          .frame(height: 24)
          .foregroundStyle(active ? Color.white : H3Color.textSecondary)
          .background(
            active ? H3Color.accent : .clear,
            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
          )
          .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("segment-\(segment.title.lowercased())")
      }
    }
    .padding(3)
    .background(H3Color.surface, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 8, style: .continuous)
        .strokeBorder(H3Color.line, lineWidth: 1)
    }
    .opacity(isEnabled ? 1 : 0.5)
    .disabled(!isEnabled)
    .animation(.easeOut(duration: 0.15), value: selection)
  }
}
