import Foundation

enum EngineProcessConfiguration {
  static func apply(to process: Process, executableURL: URL) {
    let helperDirectory = executableURL.deletingLastPathComponent()
    let bundledShaderDirectory =
      helperDirectory
      .deletingLastPathComponent()
      .appendingPathComponent("Resources", isDirectory: true)
      .appendingPathComponent("H3Engine", isDirectory: true)
    if FileManager.default.fileExists(
      atPath: bundledShaderDirectory.appendingPathComponent("h3_shaders.metal").path
    ) {
      process.currentDirectoryURL = bundledShaderDirectory
    } else {
      process.currentDirectoryURL = helperDirectory
    }
    var environment = ProcessInfo.processInfo.environment
    addTool(named: "ffmpeg", override: "H3_FFMPEG", to: &environment)
    addTool(named: "ffprobe", override: "H3_FFPROBE", to: &environment)
    process.environment = environment
  }

  private static func addTool(
    named name: String,
    override key: String,
    to environment: inout [String: String]
  ) {
    if let existing = environment[key], FileManager.default.isExecutableFile(atPath: existing) {
      return
    }

    let pathDirectories = (environment["PATH"] ?? "")
      .split(separator: ":")
      .map(String.init)
    let candidates = pathDirectories + ["/opt/homebrew/bin", "/usr/local/bin"]
    if let path =
      candidates
      .map({ URL(fileURLWithPath: $0).appendingPathComponent(name).path })
      .first(where: FileManager.default.isExecutableFile(atPath:))
    {
      environment[key] = path
    }
  }
}
