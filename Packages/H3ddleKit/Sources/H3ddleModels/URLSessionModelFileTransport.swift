import Foundation

public final class URLSessionModelFileTransport: ModelFileTransport, @unchecked Sendable {
  private let configuration: URLSessionConfiguration

  public init(configuration: URLSessionConfiguration = .default) {
    self.configuration = configuration.copy() as! URLSessionConfiguration
  }

  public func download(
    request: URLRequest,
    to destination: URL,
    existingBytes: Int64,
    progress: @escaping @Sendable (Int64) -> Void
  ) async throws -> ModelHTTPResponse {
    let operation = StreamingDownloadOperation(
      request: request,
      destination: destination,
      existingBytes: existingBytes,
      configuration: configuration,
      progress: progress
    )
    return try await withTaskCancellationHandler {
      try await operation.start()
    } onCancel: {
      operation.cancel()
    }
  }
}

private final class StreamingDownloadOperation: NSObject, URLSessionDataDelegate,
  @unchecked Sendable
{
  private let lock = NSLock()
  private let request: URLRequest
  private let destination: URL
  private let existingBytes: Int64
  private let configuration: URLSessionConfiguration
  private let progress: @Sendable (Int64) -> Void

  private var continuation: CheckedContinuation<ModelHTTPResponse, any Error>?
  private var session: URLSession?
  private var task: URLSessionDataTask?
  private var fileHandle: FileHandle?
  private var statusCode = 0
  private var baseBytes: Int64 = 0
  private var receivedBytes: Int64 = 0
  private var isFinished = false
  private var isCancelled = false

  init(
    request: URLRequest,
    destination: URL,
    existingBytes: Int64,
    configuration: URLSessionConfiguration,
    progress: @escaping @Sendable (Int64) -> Void
  ) {
    self.request = request
    self.destination = destination
    self.existingBytes = existingBytes
    self.configuration = configuration
    self.progress = progress
  }

  func start() async throws -> ModelHTTPResponse {
    try await withCheckedThrowingContinuation { continuation in
      lock.lock()
      if isCancelled {
        lock.unlock()
        continuation.resume(throwing: CancellationError())
        return
      }
      self.continuation = continuation
      let queue = OperationQueue()
      queue.maxConcurrentOperationCount = 1
      let configuration = configuration.copy() as! URLSessionConfiguration
      configuration.waitsForConnectivity = true
      configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
      configuration.timeoutIntervalForResource = 7 * 24 * 60 * 60
      let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
      let task = session.dataTask(with: request)
      self.session = session
      self.task = task
      lock.unlock()
      task.resume()
    }
  }

  func cancel() {
    lock.lock()
    isCancelled = true
    let task = task
    lock.unlock()
    task?.cancel()
  }

  func urlSession(
    _ session: URLSession,
    dataTask: URLSessionDataTask,
    didReceive response: URLResponse,
    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
  ) {
    guard let httpResponse = response as? HTTPURLResponse else {
      completionHandler(.cancel)
      finish(throwing: URLError(.badServerResponse))
      return
    }
    statusCode = httpResponse.statusCode
    guard (200...299).contains(statusCode) else {
      completionHandler(.cancel)
      finish(returning: ModelHTTPResponse(statusCode: statusCode, bytesWritten: 0))
      return
    }

    do {
      if !FileManager.default.fileExists(atPath: destination.path) {
        _ = FileManager.default.createFile(atPath: destination.path, contents: nil)
      }
      let handle = try FileHandle(forWritingTo: destination)
      if existingBytes > 0 && statusCode == 206 {
        try handle.seekToEnd()
        baseBytes = existingBytes
      } else {
        try handle.truncate(atOffset: 0)
        baseBytes = 0
      }
      fileHandle = handle
      progress(baseBytes)
      completionHandler(.allow)
    } catch {
      completionHandler(.cancel)
      finish(throwing: error)
    }
  }

  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    do {
      try fileHandle?.write(contentsOf: data)
      receivedBytes += Int64(data.count)
      progress(baseBytes + receivedBytes)
    } catch {
      dataTask.cancel()
      finish(throwing: error)
    }
  }

  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    didCompleteWithError error: (any Error)?
  ) {
    if let error {
      if (error as? URLError)?.code == .cancelled || isCancelled {
        finish(throwing: CancellationError())
      } else {
        finish(throwing: error)
      }
      return
    }
    finish(
      returning: ModelHTTPResponse(
        statusCode: statusCode,
        bytesWritten: baseBytes + receivedBytes
      )
    )
  }

  private func finish(returning response: ModelHTTPResponse) {
    finish(with: .success(response))
  }

  private func finish(throwing error: any Error) {
    finish(with: .failure(error))
  }

  private func finish(with result: Result<ModelHTTPResponse, any Error>) {
    lock.lock()
    guard !isFinished else {
      lock.unlock()
      return
    }
    isFinished = true
    let continuation = continuation
    self.continuation = nil
    let handle = fileHandle
    fileHandle = nil
    let session = session
    self.session = nil
    task = nil
    lock.unlock()

    try? handle?.close()
    session?.invalidateAndCancel()
    continuation?.resume(with: result)
  }
}
