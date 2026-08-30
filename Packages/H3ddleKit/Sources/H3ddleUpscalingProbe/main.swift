import Foundation
import H3ddleCore
import H3ddleUpscaling

@main
enum H3ddleUpscalingProbeCommand {
  static func main() throws {
    let arguments = CommandLine.arguments.dropFirst()
    let width = value(after: "--width", in: arguments).flatMap(Int.init) ?? 640
    let height = value(after: "--height", in: arguments).flatMap(Int.init) ?? 360
    let kind = value(after: "--kind", in: arguments).flatMap(MediaKind.init(rawValue:)) ?? .video

    let snapshot = AppleUpscalingCapabilityProbe.inspect(
      sourceKind: kind,
      sourcePixelSize: UpscalingPixelSize(width: width, height: height)
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(snapshot)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
  }

  private static func value(
    after flag: String,
    in arguments: ArraySlice<String>
  ) -> String? {
    guard let index = arguments.firstIndex(of: flag) else { return nil }
    let valueIndex = arguments.index(after: index)
    guard valueIndex < arguments.endIndex else { return nil }
    return arguments[valueIndex]
  }
}
