//
//  BinaryDistinct.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 06.03.25.
//

import Foundation

/// A binary-distinct string type that preserves exact byte sequences without Unicode normalization.
///
/// BinaryDistinctString overcomes limitations of both String and NSString types, where String
/// performs Unicode normalization that can alter byte sequences, and NSString is not Sendable.
/// This type maintains exact binary representation while providing string-like operations.
///
/// Use this type when you need to preserve exact byte sequences in string data, such as
/// when working with tokenizer vocabularies or configuration keys that must match precisely.
///
/// For more information, see [Modifying and Comparing Strings](https://developer.apple.com/documentation/swift/string#Modifying-and-Comparing-Strings).
public struct BinaryDistinctString: Equatable, Hashable, Sendable, Comparable, CustomStringConvertible, ExpressibleByStringLiteral {
    /// The underlying UTF-16 code unit representation.
    public let value: [UInt16]

    /// The NSString representation of this binary-distinct string.
    public var nsString: NSString {
        let native = string
        // Well-formed UTF-16 (every real vocabulary) takes the native-string path. A lone surrogate
        // would be repaired to U+FFFD there, so it keeps the exact NSString it always produced.
        if native.utf16.elementsEqual(value) { return native as NSString }
        return String(utf16CodeUnits: value, count: value.count) as NSString
    }

    /// The Swift String representation of this binary-distinct string.
    public var string: String {
        // A native Swift String. Going through `String(utf16CodeUnits:count:) as NSString` makes a
        // bridged (foreign) string, and every hash and comparison of those takes the slow
        // UTF-16-via-CoreFoundation path. Lone surrogates decode to U+FFFD.
        Self.decodeUTF16(value)
    }

    /// `String(decoding: value, as: UTF16.self)`, transcoded to UTF-8 by hand.
    ///
    /// The standard library only fast-paths UTF-8 input; UTF-16 takes its generic slow path, which
    /// was the largest cost of loading a large BPE tokenizer (every vocabulary key and merge half
    /// is decoded once). The repair is the same: each unpaired surrogate becomes one U+FFFD.
    static func decodeUTF16(_ value: [UInt16]) -> String {
        value.withUnsafeBufferPointer { units in
            let count = units.count
            if count == 0 { return "" }
            // At most 3 UTF-8 bytes per UTF-16 code unit (a surrogate pair is 2 units, 4 bytes).
            return String(unsafeUninitializedCapacity: count * 3) { utf8 in
                var out = 0
                var i = 0
                while i < count {
                    let unit = units[i]
                    i += 1
                    if unit < 0x80 {
                        utf8[out] = UInt8(truncatingIfNeeded: unit)
                        out += 1
                        continue
                    }
                    var scalar = UInt32(unit)
                    if unit & 0xF800 == 0xD800 {
                        // A high surrogate followed by a low surrogate is one scalar; anything
                        // else (a lone high or a lone low surrogate) is U+FFFD.
                        if unit < 0xDC00, i < count, units[i] & 0xFC00 == 0xDC00 {
                            scalar = 0x10000 + ((UInt32(unit) & 0x3FF) << 10) + (UInt32(units[i]) & 0x3FF)
                            i += 1
                        } else {
                            scalar = 0xFFFD
                        }
                    }
                    if scalar < 0x800 {
                        utf8[out] = UInt8(truncatingIfNeeded: 0xC0 | (scalar >> 6))
                        utf8[out + 1] = UInt8(truncatingIfNeeded: 0x80 | (scalar & 0x3F))
                        out += 2
                    } else if scalar < 0x10000 {
                        utf8[out] = UInt8(truncatingIfNeeded: 0xE0 | (scalar >> 12))
                        utf8[out + 1] = UInt8(truncatingIfNeeded: 0x80 | ((scalar >> 6) & 0x3F))
                        utf8[out + 2] = UInt8(truncatingIfNeeded: 0x80 | (scalar & 0x3F))
                        out += 3
                    } else {
                        utf8[out] = UInt8(truncatingIfNeeded: 0xF0 | (scalar >> 18))
                        utf8[out + 1] = UInt8(truncatingIfNeeded: 0x80 | ((scalar >> 12) & 0x3F))
                        utf8[out + 2] = UInt8(truncatingIfNeeded: 0x80 | ((scalar >> 6) & 0x3F))
                        utf8[out + 3] = UInt8(truncatingIfNeeded: 0x80 | (scalar & 0x3F))
                        out += 4
                    }
                }
                return out
            }
        }
    }

    /// The character count of the string.
    public var count: Int {
        string.count
    }

    /// Satisfies ``CustomStringConvertible`` protocol.
    public var description: String {
        string
    }

    /// Creates a binary-distinct string from UTF-16 code units.
    ///
    /// - Parameter bytes: Array of UTF-16 code units representing the string
    public init(_ bytes: [UInt16]) {
        value = bytes
    }

    /// Creates a binary-distinct string from an NSString.
    ///
    /// - Parameter str: The NSString to convert
    public init(_ str: NSString) {
        value = Array((str as String).utf16)
    }

    /// Creates a binary-distinct string from a Swift String.
    ///
    /// - Parameter str: The String to convert
    public init(_ str: String) {
        value = Array(str.utf16)
    }

    /// Creates a binary-distinct string from a single character.
    ///
    /// - Parameter character: The BinaryDistinctCharacter to convert
    public init(_ character: BinaryDistinctCharacter) {
        value = character.bytes
    }

    /// Creates a binary-distinct string from an array of characters.
    ///
    /// - Parameter characters: Array of BinaryDistinctCharacter to concatenate
    public init(_ characters: [BinaryDistinctCharacter]) {
        var data: [UInt16] = []
        for character in characters {
            data.append(contentsOf: character.bytes)
        }
        value = data
    }

    /// Satisfies ``ExpressibleByStringLiteral`` protocol.
    public init(stringLiteral value: String) {
        self.init(value)
    }

    public static func == (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> Bool {
        lhs.value == rhs.value
    }

    public static func < (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> Bool {
        lhs.value.lexicographicallyPrecedes(rhs.value)
    }

    public static func + (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> BinaryDistinctString {
        BinaryDistinctString(lhs.value + rhs.value)
    }

    public func hasPrefix(_ prefix: BinaryDistinctString) -> Bool {
        guard prefix.value.count <= value.count else { return false }
        return value.starts(with: prefix.value)
    }

    public func hasSuffix(_ suffix: BinaryDistinctString) -> Bool {
        guard suffix.value.count <= value.count else { return false }
        return value.suffix(suffix.value.count) == suffix.value
    }

    public func lowercased() -> BinaryDistinctString {
        .init(string.lowercased())
    }

    public func replacingOccurrences(of: Self, with: Self) -> BinaryDistinctString {
        BinaryDistinctString(string.replacingOccurrences(of: of.string, with: with.string))
    }
}

public extension BinaryDistinctString {
    typealias Index = Int // Treat indices as integers

    var startIndex: Index { 0 }
    var endIndex: Index { count }

    func index(_ i: Index, offsetBy distance: Int) -> Index {
        let newIndex = i + distance
        guard newIndex >= 0, newIndex <= count else {
            fatalError("Index out of bounds")
        }
        return newIndex
    }

    func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
        let newIndex = i + distance
        return newIndex <= limit ? newIndex : nil
    }
}

extension BinaryDistinctString: Sequence {
    public func makeIterator() -> AnyIterator<BinaryDistinctCharacter> {
        var iterator = string.makeIterator() // Use native Swift String iterator

        return AnyIterator {
            guard let char = iterator.next() else { return nil }
            return BinaryDistinctCharacter(char)
        }
    }
}

public extension BinaryDistinctString {
    subscript(bounds: PartialRangeFrom<Int>) -> BinaryDistinctString {
        let validRange = bounds.lowerBound..<value.count // Convert to Range<Int>
        return self[validRange]
    }

    /// Returns a slice of the `BinaryDistinctString` while ensuring correct rune (grapheme cluster) boundaries.
    subscript(bounds: Range<Int>) -> BinaryDistinctString {
        guard bounds.lowerBound >= 0, bounds.upperBound <= count else {
            fatalError("Index out of bounds")
        }

        let utf8Bytes = value
        var byteIndices: [Int] = []

        // Decode UTF-8 manually to find rune start positions
        var currentByteIndex = 0
        for (index, scalar) in string.unicodeScalars.enumerated() {
            if index == bounds.lowerBound {
                byteIndices.append(currentByteIndex)
            }
            currentByteIndex += scalar.utf8.count
            if index == bounds.upperBound - 1 {
                byteIndices.append(currentByteIndex)
                break
            }
        }

        // Extract the byte range
        let startByteIndex = byteIndices.first ?? 0
        let endByteIndex = byteIndices.last ?? utf8Bytes.count

        let slicedBytes = Array(utf8Bytes[startByteIndex..<endByteIndex])
        return BinaryDistinctString(slicedBytes)
    }
}

public extension Dictionary where Key == BinaryDistinctString {
    /// Merges another `BinaryDistinctDictionary` into this one
    mutating func merge(_ other: [BinaryDistinctString: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        merge(other, uniquingKeysWith: strategy)
    }

    /// Merges a `[String: Value]` dictionary into this one
    mutating func merge(_ other: [String: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        let converted = Dictionary(uniqueKeysWithValues: other.map { (BinaryDistinctString($0.key), $0.value) })
        merge(converted, uniquingKeysWith: strategy)
    }

    /// Merges a `[NSString: Value]` dictionary into this one
    mutating func merge(_ other: [NSString: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        let converted = Dictionary(uniqueKeysWithValues: other.map { (BinaryDistinctString($0.key), $0.value) })
        merge(converted, uniquingKeysWith: strategy)
    }

    func merging(_ other: [String: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
        var newDict = self
        newDict.merge(other, strategy: strategy)
        return newDict
    }

    func merging(_ other: [BinaryDistinctString: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
        var newDict = self
        newDict.merge(other, strategy: strategy)
        return newDict
    }

    func merging(_ other: [NSString: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
        var newDict = self
        newDict.merge(other, strategy: strategy)
        return newDict
    }
}

public protocol StringConvertible: ExpressibleByStringLiteral {}

extension BinaryDistinctString: StringConvertible {}
extension String: StringConvertible {}
extension NSString: StringConvertible {}

public struct BinaryDistinctCharacter: Equatable, Hashable, CustomStringConvertible, ExpressibleByStringLiteral {
    let bytes: [UInt16]

    public init(_ character: Character) {
        bytes = Array(character.utf16)
    }

    public init(_ string: String) {
        bytes = Array(string.utf16)
    }

    public init(_ nsString: NSString) {
        let swiftString = nsString as String
        bytes = Array(swiftString.utf16)
    }

    public init(bytes: [UInt16]) {
        self.bytes = bytes
    }

    /// Satisfies ``ExpressibleByStringLiteral`` protocol.
    public init(stringLiteral value: String) {
        self.init(value)
    }

    var stringValue: String? {
        String(utf16CodeUnits: bytes, count: bytes.count)
    }

    public var description: String {
        if let str = stringValue {
            "BinaryDistinctCharacter('\(str)', bytes: \(bytes.map { String(format: "0x%02X", $0) }))"
        } else {
            "BinaryDistinctCharacter(invalid UTF-8, bytes: \(bytes.map { String(format: "0x%02X", $0) }))"
        }
    }

    public static func == (lhs: BinaryDistinctCharacter, rhs: BinaryDistinctCharacter) -> Bool {
        lhs.bytes == rhs.bytes
    }
}
