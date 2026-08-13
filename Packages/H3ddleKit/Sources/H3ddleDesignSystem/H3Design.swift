import SwiftUI

public enum H3Color {
  public static let canvas = Color(red: 0.055, green: 0.052, blue: 0.049)
  public static let surface = Color(red: 0.094, green: 0.090, blue: 0.082)
  public static let surfaceRaised = Color(red: 0.137, green: 0.129, blue: 0.114)
  public static let line = Color(red: 0.216, green: 0.200, blue: 0.180)
  public static let textPrimary = Color(red: 0.957, green: 0.933, blue: 0.886)
  public static let textSecondary = Color(red: 0.667, green: 0.635, blue: 0.588)
  public static let accent = Color(red: 0.925, green: 0.612, blue: 0.200)
  public static let danger = Color(red: 0.878, green: 0.329, blue: 0.286)
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
}

public struct H3PrimaryButtonStyle: ButtonStyle {
  public init() {}

  public func makeBody(configuration: Configuration) -> some View {
    configuration.label
      .font(.system(size: 13, weight: .semibold))
      .foregroundStyle(H3Color.canvas)
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
      .background(configuration.isPressed ? H3Color.surfaceRaised : H3Color.surface)
      .overlay {
        RoundedRectangle(cornerRadius: H3Radius.small, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: H3Radius.small, style: .continuous))
  }
}
