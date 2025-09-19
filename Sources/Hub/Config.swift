//
//  Config.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 06.03.25.
//

import Foundation

public enum ConfigError: Error {
    case typeMismatch(expected: String, actual: String)
    case typeConversionFailed(value: Sendable, targetType: Sendable.Type)
}

@dynamicMemberLookup
public struct Config: Hashable, Sendable, CustomStringConvertible {
    public struct Key: Hashable, Sendable, CustomStringConvertible, Comparable {
        public let value: String

        public init(_ value: String) { self.value = value }

        public var description: String { value }

        public func hash(into hasher: inout Hasher) {
            for codeUnit in value.utf16 {
                hasher.combine(codeUnit)
            }
        }

        public static func == (lhs: Self, rhs: Self) -> Bool {
            (lhs.value as NSString).isEqual(to: rhs.value)
        }

        public static func < (lhs: Config.Key, rhs: Config.Key) -> Bool {
            lhs.value < rhs.value
        }
    }

    public let properties: [Key: Value]

    public init(_ properties: [Key: Value] = [:]) {
        self.properties = properties
    }

    public init(_ dictionary: [String: Any]) {
        self.properties = Config.convertKeys(dictionary)
    }

    public init(_ dictionary: [NSString: Any]) {
        self.properties = Config.convertKeys(dictionary)
    }

    private static func convertKeys(_ object: Any) -> [Key: Value] {
        if let dict = object as? [String: Any] {
            return Dictionary(uniqueKeysWithValues: dict.map { key, value in
                (Key(key), Value.fromAny(value))
            })
        }
        if let dict = object as? [NSString: Any] {
            return Dictionary(uniqueKeysWithValues: dict.map { key, value in
                (Key(key as String), Value.fromAny(value))
            })
        }
        fatalError("Top-level config must be a dictionary")
    }

    public subscript(index: String) -> Value? {
        let key = Key(index)
        if let value = properties[key] {
            return value
        }

        let uncamelCased = uncamelCase(key)
        if key != uncamelCased, let value = properties[uncamelCased] {
            return value
        }

        return nil
    }

    public subscript(dynamicMember member: String) -> Value? {
        self[member]
    }

    private func uncamelCase(_ string: Key) -> Key {
        let scalars = string.value.unicodeScalars
        var result = ""

        var previousCharacterIsLowercase = false
        for scalar in scalars {
            if CharacterSet.uppercaseLetters.contains(scalar) {
                if previousCharacterIsLowercase {
                    result += "_"
                }
                result += Character(scalar).lowercased()
                previousCharacterIsLowercase = false
            } else {
                result += String(scalar)
                previousCharacterIsLowercase = true
            }
        }
        return Key(result)
    }

    public var description: String {
        "\(properties)"
    }
}

public extension Config {
    @dynamicMemberLookup
    enum Value: Hashable, Sendable, CustomStringConvertible {
        case null
        case string(String)
        case integer(Int)
        case boolean(Bool)
        case floating(Float)
        case dictionary(Config)
        case array([Value])
        case token(UInt, String)

        static func fromAny(_ object: Any) -> Value {
            switch object {
            case let obj as String:
                .string(obj)
            case let obj as Int:
                .integer(obj)
            case let obj as Float:
                .floating(obj)
            case let obj as Double:
                .floating(Float(obj))
            case let obj as Bool:
                .boolean(obj)
            case let obj as NSNumber:
                if CFNumberIsFloatType(obj) {
                    .floating(obj.floatValue)
                } else {
                    .integer(obj.intValue)
                }
            case is NSNull:
                .null
            case let obj as [Any]:
                .array(obj.map { Value.fromAny($0) })
            case let obj as [String: Any]:
                .dictionary(Config(obj))
            case let obj as (UInt, String):
                .token(obj.0, obj.1)
            case let config as Config:
                .dictionary(config)
            case let value as Value:
                value
            default:
                fatalError("unknown type: \(type(of: object)) \(object)")
            }
        }

        public var description: String {
            switch self {
            case .null: "null"
            case let .string(value): "\"\(value)\""
            case let .integer(value): "\(value)"
            case let .boolean(value): "\(value)"
            case let .floating(value): "\(value)"
            case let .array(arr): "\(arr)"
            case let .dictionary(val): "\(val)"
            case let .token(id, token): "(\(id), \(token))"
            }
        }

        // MARK: getters - string

        public var string: String? {
            if case let .string(val) = self { return val }
            return nil
        }

        public func string(or defaultValue: String) -> String { self.string ?? defaultValue }

        // MARK: getters - boolean

        public var boolean: Bool? {
            if case let .boolean(val) = self { return val }
            if case let .integer(val) = self { return val == 1 }
            if case let .string(val) = self {
                switch val.lowercased() {
                case "true", "t", "1": return true
                case "false", "f", "0": return false
                default: return nil
                }
            }
            return nil
        }

        public func boolean(or defaultValue: Bool) -> Bool { self.boolean ?? defaultValue }

        // MARK: getters - integer

        public var integer: Int? {
            if case let .integer(val) = self { return val }
            return nil
        }

        public func integer(or defaultValue: Int) -> Int { self.integer ?? defaultValue }

        // MARK: getters - floating

        public var floating: Float? {
            if case let .floating(val) = self { return val }
            if case let .integer(val) = self { return Float(val) }
            return nil
        }

        public func floating(or defaultValue: Float) -> Float { self.floating ?? defaultValue }

        // MARK: getters - dictionary

        public var dictionary: Config? {
            if case let .dictionary(val) = self { return val }
            return nil
        }

        // MARK: getters - array

        public var array: [Value]? {
            if case let .array(val) = self { return val }
            return nil
        }

        public func array(or defaultValue: [Value]) -> [Value] { self.array ?? defaultValue }

        // MARK: getters - token

        public var token: (UInt, String)? {
            if case let .token(id, token) = self { return (id, token) }
            if case let .array(arr) = self, arr.count == 2, let id = arr[1].integer, let token = arr[0].string {
                return (UInt(id), token)
            }
            return nil
        }

        public func token(or defaultValue: (UInt, String)) -> (UInt, String) { self.token ?? defaultValue }

        public subscript(dynamicMember member: String) -> Value? {
            if case let .dictionary(config) = self {
                return config[dynamicMember: member]
            }
            return nil
        }

        public subscript(index: Int) -> Value? {
            if case let .array(arr) = self, index >= 0, index < arr.count {
                return arr[index]
            }
            return nil
        }

        public func toJinjaCompatible() -> Any? {
            switch self {
            case let .array(val):
                return val.map { $0.toJinjaCompatible() }
            case let .dictionary(val):
                var result: [String: Any?] = [:]
                for (key, configValue) in val.properties {
                    result[key.value] = configValue.toJinjaCompatible()
                }
                return result
            case let .boolean(val):
                return val
            case let .floating(val):
                return val
            case let .integer(val):
                return val
            case let .string(val):
                return val
            case let .token(id, token):
                return [String(id): token] as [String: String]
            case .null:
                return nil
            }
        }
    }
}

// MARK: - Expressible by literal protocols

extension Config: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (Key, Value)...) {
        self.init(Dictionary(uniqueKeysWithValues: elements))
    }
}

extension Config.Key: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self.init(value) }
}

extension Config.Value: ExpressibleByNilLiteral {
    public init(nilLiteral _: ()) { self = .null }
}

extension Config.Value: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .boolean(value) }
}

extension Config.Value: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) { self = .integer(value) }
}

extension Config.Value: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Float) { self = .floating(value) }
}

extension Config.Value: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
}

extension Config.Value: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Config.Value...) { self = .array(elements) }
}

extension Config.Value: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (Config.Key, Config.Value)...) {
        self = .dictionary(Config(Dictionary(uniqueKeysWithValues: elements)))
    }
}

// MARK: - Codable

extension Config: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        var properties: [Key: Value] = [:]
        for key in container.allKeys {
            let value = try container.decode(Value.self, forKey: key)
            properties[Key(key.stringValue)] = value
        }
        self.properties = properties
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        for (key, value) in properties {
            try container.encode(value, forKey: CodingKeys(stringValue: key.value)!)
        }
    }

    private struct CodingKeys: CodingKey {
        var stringValue: String
        init?(stringValue: String) { self.stringValue = stringValue }
        var intValue: Int? { nil }
        init?(intValue _: Int) { nil }
    }
}

extension Config.Value: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self = .null
        } else if let intValue = try? container.decode(Int.self) {
            self = .integer(intValue)
        } else if let floatValue = try? container.decode(Float.self) {
            self = .floating(floatValue)
        } else if let boolValue = try? container.decode(Bool.self) {
            self = .boolean(boolValue)
        } else if let stringValue = try? container.decode(String.self) {
            self = .string(stringValue)
        } else if let arrayValue = try? container.decode([Config.Value].self) {
            if arrayValue.count == 2,
               let id = arrayValue[0].integer,
               let token = arrayValue[1].string
            {
                self = .token(UInt(id), token)
            } else {
                self = .array(arrayValue)
            }
        } else if let dictionaryValue = try? container.decode(Config.self) {
            self = .dictionary(dictionaryValue)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported Config.Value type")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case let .integer(val):
            try container.encode(val)
        case let .floating(val):
            try container.encode(val)
        case let .boolean(val):
            try container.encode(val)
        case let .string(val):
            try container.encode(val)
        case let .dictionary(dictionaryValue):
            try container.encode(dictionaryValue)
        case let .array(arrayValue):
            try container.encode(arrayValue)
        case let .token(id, token):
            try container.encode([.integer(Int(id)), .string(token)] as [Config.Value])
        }
    }
}
