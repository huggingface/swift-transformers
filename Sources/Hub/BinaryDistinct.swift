//
//  BinaryDistinctString.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 06.03.25.
//

import Foundation

/// BinaryDistinctString helps to overcome limitations of both String and NSString types. Where the prior is performing unicode normalization and the following is not Sendable. For more reference [Modifying-and-Comparing-Strings](https://developer.apple.com/documentation/swift/string#Modifying-and-Comparing-Strings).
public struct BinaryDistinctString: Equatable, Hashable, Sendable, Comparable, CustomStringConvertible, ExpressibleByStringLiteral {
    public let value: [UInt16]

    public var nsString: NSString {
        return String(utf16CodeUnits: self.value, count: self.value.count) as NSString
    }

    public var string: String {
        return String(self.nsString)
    }

    public var count: Int {
        self.string.count
    }

    /// Satisfies ``CustomStringConvertible`` protocol.
    public var description: String {
        return self.string
    }

    public init(_ bytes: [UInt16]) {
        self.value = bytes
    }

    public init(_ str: NSString) {
        self.value = Array(str as String).flatMap { $0.utf16 }
    }

    public init(_ str: String) {
        self.init(str as NSString)
    }

    public init(_ character: BinaryDistinctCharacter) {
        self.value = character.bytes
    }

    public init(_ characters: [BinaryDistinctCharacter]) {
        var data: [UInt16] = []
        for character in characters {
            data.append(contentsOf: character.bytes)
        }
        self.value = data
    }

    /// Satisfies ``ExpressibleByStringLiteral`` protocol.
    public init(stringLiteral value: String) {
        self.init(value)
    }

    public static func == (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> Bool {
        return lhs.value == rhs.value
    }

    public static func < (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> Bool {
        return lhs.value.lexicographicallyPrecedes(rhs.value)
    }

    public static func + (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> BinaryDistinctString {
        return BinaryDistinctString(lhs.value + rhs.value)
    }

    public func hasPrefix(_ prefix: BinaryDistinctString) -> Bool {
        guard prefix.value.count <= self.value.count else { return false }
        return self.value.starts(with: prefix.value)
    }

    public func hasSuffix(_ suffix: BinaryDistinctString) -> Bool {
        guard suffix.value.count <= self.value.count else { return false }
        return self.value.suffix(suffix.value.count) == suffix.value
    }

    public func lowercased() -> BinaryDistinctString {
        .init(self.string.lowercased())
    }

    public func replacingOccurrences(of: Self, with: Self) -> BinaryDistinctString {
        return BinaryDistinctString(self.string.replacingOccurrences(of: of.string, with: with.string))
    }
}

extension BinaryDistinctString {
    public typealias Index = Int  // Treat indices as integers

    public var startIndex: Index { return 0 }
    public var endIndex: Index { return self.count }

    public func index(_ i: Index, offsetBy distance: Int) -> Index {
        let newIndex = i + distance
        guard newIndex >= 0, newIndex <= self.count else {
            fatalError("Index out of bounds")
        }
        return newIndex
    }

    public func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
        let newIndex = i + distance
        return newIndex <= limit ? newIndex : nil
    }
}

extension BinaryDistinctString: Sequence {
    public func makeIterator() -> AnyIterator<BinaryDistinctCharacter> {
        var iterator = self.string.makeIterator()  // Use native Swift String iterator

        return AnyIterator {
            guard let char = iterator.next() else { return nil }
            return BinaryDistinctCharacter(char)
        }
    }
}

extension BinaryDistinctString {
    public subscript(bounds: PartialRangeFrom<Int>) -> BinaryDistinctString {
        get {
            let validRange = bounds.lowerBound..<self.value.count  // Convert to Range<Int>
            return self[validRange]
        }
    }

    /// Returns a slice of the `BinaryDistinctString` while ensuring correct rune (grapheme cluster) boundaries.
    public subscript(bounds: Range<Int>) -> BinaryDistinctString {
        get {
            guard bounds.lowerBound >= 0, bounds.upperBound <= self.count else {
                fatalError("Index out of bounds")
            }

            let utf8Bytes = self.value
            var byteIndices: [Int] = []

            // Decode UTF-8 manually to find rune start positions
            var currentByteIndex = 0
            for (index, scalar) in self.string.unicodeScalars.enumerated() {
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
}

extension Dictionary where Key == BinaryDistinctString {
    /// Merges another `BinaryDistinctDictionary` into this one
    public mutating func merge(_ other: [BinaryDistinctString: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        self.merge(other, uniquingKeysWith: strategy)
    }

    /// Merges a `[String: Value]` dictionary into this one
    public mutating func merge(_ other: [String: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        let converted = Dictionary(uniqueKeysWithValues: other.map { (BinaryDistinctString($0.key), $0.value) })
        self.merge(converted, uniquingKeysWith: strategy)
    }

    /// Merges a `[NSString: Value]` dictionary into this one
    public mutating func merge(_ other: [NSString: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        let converted = Dictionary(uniqueKeysWithValues: other.map { (BinaryDistinctString($0.key), $0.value) })
        self.merge(converted, uniquingKeysWith: strategy)
    }

    public func merging(_ other: [String: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
        var newDict = self
        newDict.merge(other, strategy: strategy)
        return newDict
    }

    public func merging(_ other: [BinaryDistinctString: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
        var newDict = self
        newDict.merge(other, strategy: strategy)
        return newDict
    }

    public func merging(_ other: [NSString: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
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
        self.bytes = Array(character.utf16)
    }

    public init(_ string: String) {
        self.bytes = Array(string.utf16)
    }

    public init(_ nsString: NSString) {
        let swiftString = nsString as String
        self.bytes = Array(swiftString.utf16)
    }

    public init(bytes: [UInt16]) {
        self.bytes = bytes
    }

    /// Satisfies ``ExpressibleByStringLiteral`` protocol.
    public init(stringLiteral value: String) {
        self.init(value)
    }

    var stringValue: String? {
        String(utf16CodeUnits: self.bytes, count: self.bytes.count)
    }

    public var description: String {
        if let str = stringValue {
            return "BinaryDistinctCharacter('\(str)', bytes: \(bytes.map { String(format: "0x%02X", $0) }))"
        } else {
            return "BinaryDistinctCharacter(invalid UTF-8, bytes: \(bytes.map { String(format: "0x%02X", $0) }))"
        }
    }

    public static func == (lhs: BinaryDistinctCharacter, rhs: BinaryDistinctCharacter) -> Bool {
        lhs.bytes == rhs.bytes
    }
}
