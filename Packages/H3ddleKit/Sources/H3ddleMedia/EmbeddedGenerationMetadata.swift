import Foundation

public enum EmbeddedGenerationMetadataError: LocalizedError, Equatable, Sendable {
  case unsupportedFormat(String)
  case invalidContainer
  case metadataTooLarge

  public var errorDescription: String? {
    switch self {
    case .unsupportedFormat(let ext):
      "Generation metadata is not supported for .\(ext) files."
    case .invalidContainer:
      "The generated media container is invalid."
    case .metadataTooLarge:
      "The generation metadata is too large to embed."
    }
  }
}

/// A small, lossless provenance payload carried by the media itself. PNG uses
/// an iTXt entry, WAV uses a private RIFF chunk, and ISO-BMFF movies use the
/// standard UUID extension box. Unknown readers ignore all three.
public enum EmbeddedGenerationMetadata {
  public static let keyword = "h3ddle"
  public static let maximumByteCount = 1_048_576

  public static func write(_ json: Data, to url: URL) throws {
    guard json.count <= maximumByteCount else {
      throw EmbeddedGenerationMetadataError.metadataTooLarge
    }
    switch url.pathExtension.lowercased() {
    case "png":
      try writePNG(json, to: url)
    case "wav", "wave":
      try writeWAV(json, to: url)
    case "mp4", "mov", "m4v":
      try writeISOBox(json, to: url)
    default:
      throw EmbeddedGenerationMetadataError.unsupportedFormat(url.pathExtension.lowercased())
    }
  }

  public static func read(from url: URL) throws -> Data? {
    switch url.pathExtension.lowercased() {
    case "png":
      try readPNG(from: url)
    case "wav", "wave":
      try readWAV(from: url)
    case "mp4", "mov", "m4v":
      try readISOBox(from: url)
    default:
      nil
    }
  }

  // MARK: - PNG iTXt

  private static let pngSignature = Data([137, 80, 78, 71, 13, 10, 26, 10])

  private static func writePNG(_ json: Data, to url: URL) throws {
    let source = try Data(contentsOf: url)
    guard source.count >= pngSignature.count,
      source.prefix(pngSignature.count) == pngSignature
    else { throw EmbeddedGenerationMetadataError.invalidContainer }
    guard let iend = pngChunks(in: source).last(where: { $0.type == "IEND" }) else {
      throw EmbeddedGenerationMetadataError.invalidContainer
    }
    var text = Data(keyword.utf8)
    // keyword terminator, compression flag/method, language terminator, and
    // translated-keyword terminator. The JSON that follows is UTF-8.
    text.append(contentsOf: [0, 0, 0, 0, 0])
    text.append(json)
    var output = Data(source[..<iend.offset])
    output.append(pngChunk(type: "iTXt", payload: text))
    output.append(source[iend.offset...])
    try output.write(to: url, options: .atomic)
  }

  private static func readPNG(from url: URL) throws -> Data? {
    let source = try Data(contentsOf: url)
    guard source.count >= pngSignature.count,
      source.prefix(pngSignature.count) == pngSignature
    else { throw EmbeddedGenerationMetadataError.invalidContainer }
    var result: Data?
    for chunk in pngChunks(in: source) where chunk.type == "iTXt" {
      let payload = source[chunk.payloadRange]
      guard let separator = payload.firstIndex(of: 0),
        String(decoding: payload[..<separator], as: UTF8.self) == keyword
      else { continue }
      let fieldsStart = payload.index(after: separator)
      guard payload.distance(from: fieldsStart, to: payload.endIndex) >= 4 else { continue }
      var cursor = fieldsStart
      let compressionFlag = payload[cursor]
      cursor = payload.index(after: cursor)
      _ = payload[cursor] // compression method
      cursor = payload.index(after: cursor)
      guard compressionFlag == 0,
        let languageEnd = payload[cursor...].firstIndex(of: 0)
      else { continue }
      cursor = payload.index(after: languageEnd)
      guard let translatedEnd = payload[cursor...].firstIndex(of: 0) else { continue }
      cursor = payload.index(after: translatedEnd)
      let candidate = Data(payload[cursor...])
      if candidate.count <= maximumByteCount { result = candidate }
    }
    return result
  }

  private struct PNGChunk {
    var offset: Int
    var type: String
    var payloadRange: Range<Data.Index>
  }

  private static func pngChunks(in data: Data) -> [PNGChunk] {
    guard data.count >= 8 else { return [] }
    var chunks: [PNGChunk] = []
    var offset = 8
    while offset + 12 <= data.count {
      let length = Int(readUInt32BE(data, at: offset))
      let end = offset + 12 + length
      guard length >= 0, end <= data.count else { break }
      let typeStart = offset + 4
      let type = String(decoding: data[typeStart..<(typeStart + 4)], as: UTF8.self)
      chunks.append(
        PNGChunk(
          offset: offset,
          type: type,
          payloadRange: (offset + 8)..<(offset + 8 + length)
        )
      )
      offset = end
    }
    return chunks
  }

  private static func pngChunk(type: String, payload: Data) -> Data {
    let typeData = Data(type.utf8)
    var result = Data()
    appendUInt32BE(UInt32(payload.count), to: &result)
    result.append(typeData)
    result.append(payload)
    var checksumInput = typeData
    checksumInput.append(payload)
    appendUInt32BE(crc32(checksumInput), to: &result)
    return result
  }

  private static func crc32(_ data: Data) -> UInt32 {
    var crc: UInt32 = 0xffff_ffff
    for byte in data {
      crc ^= UInt32(byte)
      for _ in 0..<8 {
        crc = (crc >> 1) ^ ((crc & 1) == 1 ? 0xedb8_8320 : 0)
      }
    }
    return crc ^ 0xffff_ffff
  }

  // MARK: - RIFF/WAV

  private static let wavChunkType = Data("h3dL".utf8)

  private static func writeWAV(_ json: Data, to url: URL) throws {
    var source = try Data(contentsOf: url)
    guard source.count >= 12,
      source.prefix(4) == Data("RIFF".utf8),
      source[8..<12] == Data("WAVE".utf8)
    else { throw EmbeddedGenerationMetadataError.invalidContainer }
    source.append(wavChunkType)
    appendUInt32LE(UInt32(json.count), to: &source)
    source.append(json)
    if json.count.isMultiple(of: 2) == false { source.append(0) }
    guard source.count - 8 <= Int(UInt32.max) else {
      throw EmbeddedGenerationMetadataError.metadataTooLarge
    }
    replaceUInt32LE(UInt32(source.count - 8), in: &source, at: 4)
    try source.write(to: url, options: .atomic)
  }

  private static func readWAV(from url: URL) throws -> Data? {
    let source = try Data(contentsOf: url)
    guard source.count >= 12,
      source.prefix(4) == Data("RIFF".utf8),
      source[8..<12] == Data("WAVE".utf8)
    else { throw EmbeddedGenerationMetadataError.invalidContainer }
    var offset = 12
    var result: Data?
    while offset + 8 <= source.count {
      let size = Int(readUInt32LE(source, at: offset + 4))
      let payloadStart = offset + 8
      let payloadEnd = payloadStart + size
      guard payloadEnd <= source.count else { break }
      if source[offset..<(offset + 4)] == wavChunkType, size <= maximumByteCount {
        result = Data(source[payloadStart..<payloadEnd])
      }
      offset = payloadEnd + (size.isMultiple(of: 2) ? 0 : 1)
    }
    return result
  }

  // MARK: - ISO base media UUID box

  private static let isoUUID = Data([
    0x48, 0x33, 0x44, 0x4c, 0x45, 0x2d, 0x47, 0x45,
    0x8e, 0x45, 0x52, 0x41, 0x54, 0x49, 0x4f, 0x4e,
  ])

  private static func writeISOBox(_ json: Data, to url: URL) throws {
    let size = 8 + isoUUID.count + json.count
    guard size <= Int(UInt32.max) else {
      throw EmbeddedGenerationMetadataError.metadataTooLarge
    }
    var box = Data()
    appendUInt32BE(UInt32(size), to: &box)
    box.append(Data("uuid".utf8))
    box.append(isoUUID)
    box.append(json)
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    try handle.seekToEnd()
    try handle.write(contentsOf: box)
  }

  private static func readISOBox(from url: URL) throws -> Data? {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    guard let number = attributes[.size] as? NSNumber else {
      throw EmbeddedGenerationMetadataError.invalidContainer
    }
    let fileSize = number.uint64Value
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var offset: UInt64 = 0
    var result: Data?
    while offset + 8 <= fileSize {
      try handle.seek(toOffset: offset)
      guard let header = try handle.read(upToCount: 8), header.count == 8 else { break }
      let shortSize = UInt64(readUInt32BE(header, at: 0))
      let type = String(decoding: header[4..<8], as: UTF8.self)
      var headerSize: UInt64 = 8
      var boxSize = shortSize
      if shortSize == 1 {
        guard let extended = try handle.read(upToCount: 8), extended.count == 8 else { break }
        boxSize = readUInt64BE(extended, at: 0)
        headerSize = 16
      } else if shortSize == 0 {
        boxSize = fileSize - offset
      }
      guard boxSize >= headerSize, offset + boxSize <= fileSize else { break }
      if type == "uuid", boxSize >= headerSize + 16 {
        guard let uuid = try handle.read(upToCount: 16), uuid.count == 16 else { break }
        if uuid == isoUUID {
          let payloadSize = boxSize - headerSize - 16
          if payloadSize <= UInt64(maximumByteCount),
            let payload = try handle.read(upToCount: Int(payloadSize)),
            payload.count == Int(payloadSize)
          {
            result = payload
          }
        }
      }
      offset += boxSize
    }
    return result
  }

  // MARK: - Integer coding

  private static func readUInt32BE(_ data: Data, at offset: Int) -> UInt32 {
    data[offset..<(offset + 4)].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
  }

  private static func readUInt64BE(_ data: Data, at offset: Int) -> UInt64 {
    data[offset..<(offset + 8)].reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
  }

  private static func readUInt32LE(_ data: Data, at offset: Int) -> UInt32 {
    var value: UInt32 = 0
    for index in 0..<4 {
      value |= UInt32(data[offset + index]) << UInt32(index * 8)
    }
    return value
  }

  private static func appendUInt32BE(_ value: UInt32, to data: inout Data) {
    data.append(UInt8((value >> 24) & 0xff))
    data.append(UInt8((value >> 16) & 0xff))
    data.append(UInt8((value >> 8) & 0xff))
    data.append(UInt8(value & 0xff))
  }

  private static func appendUInt32LE(_ value: UInt32, to data: inout Data) {
    data.append(UInt8(value & 0xff))
    data.append(UInt8((value >> 8) & 0xff))
    data.append(UInt8((value >> 16) & 0xff))
    data.append(UInt8((value >> 24) & 0xff))
  }

  private static func replaceUInt32LE(_ value: UInt32, in data: inout Data, at offset: Int) {
    data[offset] = UInt8(value & 0xff)
    data[offset + 1] = UInt8((value >> 8) & 0xff)
    data[offset + 2] = UInt8((value >> 16) & 0xff)
    data[offset + 3] = UInt8((value >> 24) & 0xff)
  }
}
