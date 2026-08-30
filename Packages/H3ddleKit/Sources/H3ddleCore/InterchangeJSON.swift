import Foundation

/// Untyped JSON so interchange documents can round-trip keys H3ddle does not model.
public enum JSONValue: Hashable, Sendable {
  case null
  case bool(Bool)
  case int(Int)
  case double(Double)
  case string(String)
  case array([JSONValue])
  case object([String: JSONValue])
}

extension JSONValue: Codable {
  public init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() {
      self = .null
      return
    }
    if let value = try? container.decode(Bool.self) {
      self = .bool(value)
      return
    }
    if let value = try? container.decode(Int.self) {
      self = .int(value)
      return
    }
    if let value = try? container.decode(Double.self) {
      self = .double(value)
      return
    }
    if let value = try? container.decode(String.self) {
      self = .string(value)
      return
    }
    if let value = try? container.decode([JSONValue].self) {
      self = .array(value)
      return
    }
    if let value = try? container.decode([String: JSONValue].self) {
      self = .object(value)
      return
    }
    throw DecodingError.dataCorruptedError(
      in: container,
      debugDescription: "Unsupported JSON value."
    )
  }

  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .null:
      try container.encodeNil()
    case .bool(let value):
      try container.encode(value)
    case .int(let value):
      try container.encode(value)
    case .double(let value):
      try container.encode(value)
    case .string(let value):
      try container.encode(value)
    case .array(let value):
      try container.encode(value)
    case .object(let value):
      try container.encode(value)
    }
  }

  public var object: [String: JSONValue]? {
    if case .object(let value) = self { return value }
    return nil
  }

  public var array: [JSONValue]? {
    if case .array(let value) = self { return value }
    return nil
  }

  public var string: String? {
    if case .string(let value) = self { return value }
    return nil
  }

  public var bool: Bool? {
    if case .bool(let value) = self { return value }
    return nil
  }

  public var number: Double? {
    switch self {
    case .int(let value): Double(value)
    case .double(let value): value
    default: nil
    }
  }

  public var int: Int? {
    switch self {
    case .int(let value): value
    case .double(let value) where value.rounded() == value: Int(value)
    default: nil
    }
  }
}

public enum InterchangeError: Error, Equatable, Sendable {
  case notAnObject
  case missingKey(String)
  case unexpectedType(String)
  case unsupportedSchemaVersion(Int)
  case missingAsset
  case invalidMediaURL
}

struct JSONObject {
  var fields: [String: JSONValue]

  init(_ value: JSONValue) throws {
    guard let fields = value.object else { throw InterchangeError.notAnObject }
    self.fields = fields
  }

  init(_ fields: [String: JSONValue] = [:]) {
    self.fields = fields
  }

  func requiredString(_ key: String) throws -> String {
    guard let value = fields[key]?.string else {
      throw InterchangeError.missingKey(key)
    }
    return value
  }

  func string(_ key: String) -> String? {
    fields[key]?.string
  }

  func requiredInt(_ key: String) throws -> Int {
    guard let value = fields[key]?.int else {
      throw InterchangeError.missingKey(key)
    }
    return value
  }

  func int(_ key: String, default fallback: Int) -> Int {
    fields[key]?.int ?? fallback
  }

  func requiredDouble(_ key: String) throws -> Double {
    guard let value = fields[key]?.number else {
      throw InterchangeError.missingKey(key)
    }
    return value
  }

  func double(_ key: String, default fallback: Double = 0) -> Double {
    fields[key]?.number ?? fallback
  }

  func double(_ key: String) -> Double? {
    fields[key]?.number
  }

  func bool(_ key: String, default fallback: Bool) -> Bool {
    fields[key]?.bool ?? fallback
  }

  func bool(_ key: String) -> Bool? {
    fields[key]?.bool
  }

  func object(_ key: String) throws -> JSONObject {
    guard let value = fields[key] else { throw InterchangeError.missingKey(key) }
    return try JSONObject(value)
  }

  func optionalObject(_ key: String) throws -> JSONObject? {
    guard let value = fields[key], value != .null else { return nil }
    return try JSONObject(value)
  }

  func array(_ key: String) -> [JSONValue] {
    fields[key]?.array ?? []
  }

  func extras(excluding known: Set<String>) -> [String: JSONValue] {
    fields.filter { !known.contains($0.key) }
  }

  mutating func set(_ key: String, _ value: JSONValue?) {
    if let value {
      fields[key] = value
    } else {
      fields.removeValue(forKey: key)
    }
  }

  mutating func set(_ key: String, _ value: String?) {
    set(key, value.map(JSONValue.string))
  }

  mutating func set(_ key: String, _ value: Int?) {
    set(key, value.map(JSONValue.int))
  }

  mutating func set(_ key: String, _ value: Double?) {
    set(key, value.map(JSONValue.double))
  }

  mutating func set(_ key: String, _ value: Bool?) {
    set(key, value.map(JSONValue.bool))
  }

  mutating func mergeExtras(_ extras: [String: JSONValue], known: Set<String>) {
    for (key, value) in extras where !known.contains(key) {
      fields[key] = value
    }
  }

  var json: JSONValue { .object(fields) }
}

func jsonArray<T>(_ values: [T], _ encode: (T) -> JSONValue) -> JSONValue {
  .array(values.map(encode))
}

func jsonArray<T>(_ values: [T], _ keyPath: KeyPath<T, JSONValue>) -> JSONValue {
  .array(values.map { $0[keyPath: keyPath] })
}

func decodeArray<T>(_ values: [JSONValue], _ decode: (JSONValue) throws -> T) throws -> [T] {
  try values.map(decode)
}

func jsonVec(_ values: [Double]) -> JSONValue {
  .array(values.map(JSONValue.double))
}

func decodeVec(_ value: JSONValue?, count: Int, default fallback: [Double]) -> [Double] {
  guard let items = value?.array, items.count == count else { return fallback }
  let numbers = items.compactMap(\.number)
  return numbers.count == count ? numbers : fallback
}

extension Array {
  subscript(safe index: Int) -> Element? {
    indices.contains(index) ? self[index] : nil
  }
}
