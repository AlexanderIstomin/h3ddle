import Foundation
import H3ddleEngineProtocol

public protocol EngineInspecting: Sendable {
  func capabilities() async throws -> EngineCapabilities
  func inspectModel(at directory: URL) async throws -> EngineModelReport
}

public enum EngineClientError: LocalizedError, Equatable, Sendable {
  case executableMissing(URL)
  case launchFailed(String)
  case noResponse
  case invalidResponse
  case rejected(String)
  case busy

  public var errorDescription: String? {
    switch self {
    case .executableMissing:
      "The H3 engine helper is not installed."
    case .launchFailed(let message):
      "The H3 engine helper could not start: \(message)"
    case .noResponse:
      "The H3 engine helper did not respond."
    case .invalidResponse:
      "The H3 engine helper returned an invalid response."
    case .rejected(let message):
      message
    case .busy:
      "The H3 engine is already generating."
    }
  }
}

public struct EngineProcessClient: EngineInspecting, Sendable {
  public let executableURL: URL

  public init(executableURL: URL) {
    self.executableURL = executableURL
  }

  public func capabilities() async throws -> EngineCapabilities {
    let event = try await exchange(EngineCommand(kind: .handshake))
    guard event.kind == .ready, let capabilities = event.capabilities else {
      throw EngineClientError.invalidResponse
    }
    return capabilities
  }

  public func inspectModel(at directory: URL) async throws -> EngineModelReport {
    let event = try await exchange(
      EngineCommand(
        kind: .inspectModel,
        modelInspection: EngineModelInspectionRequest(modelDirectory: directory)
      )
    )
    guard event.kind == .modelInspected, let model = event.model else {
      throw EngineClientError.invalidResponse
    }
    return model
  }

  private func exchange(_ command: EngineCommand) async throws -> EngineEvent {
    let executableURL = executableURL
    return try await Task.detached(priority: .userInitiated) {
      try Self.performExchange(command, executableURL: executableURL)
    }.value
  }

  private static func performExchange(
    _ command: EngineCommand,
    executableURL: URL
  ) throws -> EngineEvent {
    guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
      throw EngineClientError.executableMissing(executableURL)
    }

    let process = Process()
    let standardInput = Pipe()
    let standardOutput = Pipe()
    let standardError = Pipe()
    process.executableURL = executableURL
    process.standardInput = standardInput
    process.standardOutput = standardOutput
    process.standardError = standardError
    EngineProcessConfiguration.apply(to: process, executableURL: executableURL)

    do {
      try process.run()
    } catch {
      throw EngineClientError.launchFailed(error.localizedDescription)
    }

    var input = try EngineLineCodec.encode(command)
    input.append(try EngineLineCodec.encode(EngineCommand(kind: .shutdown)))
    standardInput.fileHandleForWriting.write(input)
    try? standardInput.fileHandleForWriting.close()

    let output = standardOutput.fileHandleForReading.readDataToEndOfFile()
    let errorData = standardError.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()

    let events = output.split(separator: 0x0A).compactMap { line in
      try? EngineLineCodec.decode(EngineEvent.self, from: Data(line))
    }
    guard let event = events.first(where: { $0.requestID == command.requestID }) else {
      if !errorData.isEmpty, let message = String(data: errorData, encoding: .utf8) {
        throw EngineClientError.launchFailed(
          message.trimmingCharacters(in: .whitespacesAndNewlines)
        )
      }
      throw output.isEmpty ? EngineClientError.noResponse : EngineClientError.invalidResponse
    }
    if event.kind == .failed {
      throw EngineClientError.rejected(event.message ?? "The H3 engine rejected the request.")
    }
    return event
  }
}

public enum EngineExecutableLocator {
  public static func bundled(in bundle: Bundle = .main) -> URL {
    bundled(at: bundle.bundleURL)
  }

  public static func bundled(at bundleURL: URL) -> URL {
    bundleURL
      .appendingPathComponent("Contents", isDirectory: true)
      .appendingPathComponent("Helpers", isDirectory: true)
      .appendingPathComponent("H3ddleEngineService", isDirectory: false)
  }
}
