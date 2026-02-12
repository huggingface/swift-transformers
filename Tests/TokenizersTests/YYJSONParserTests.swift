//
//  YYJSONParserTests.swift
//  swift-transformers
//
//  Tests for YYJSONParser functionality.
//

import Foundation
import Testing

@testable import Hub

@Suite
struct YYJSONParserTests {
    @Test
    func bomCharactersPreservedInStrings() throws {
        // JSON with BOM character inside a string value (like Gemma tokenizers)
        // See: https://github.com/huggingface/swift-transformers/issues/116
        let jsonWithBOMInString = "{\"token\": \"\u{feff}#\", \"normal\": \"#\"}"
        let data = Data(jsonWithBOMInString.utf8)

        let config = try YYJSONParser.parseToConfig(data)

        let tokenValue = config["token"].string()
        #expect(tokenValue == "\u{feff}#", "BOM character should be preserved in string values")
        #expect(tokenValue?.utf8.count == 4, "BOM (3 bytes) + # (1 byte) = 4 bytes")

        let normalValue = config["normal"].string()
        #expect(normalValue == "#", "Normal string should be unchanged")
    }

    @Test
    func parsesBasicTypes() throws {
        let json = "{\"string\": \"hello\", \"int\": 42, \"float\": 3.14, \"bool\": true, \"null\": null}"
        let data = Data(json.utf8)

        let config = try YYJSONParser.parseToConfig(data)

        #expect(config["string"].string() == "hello")
        #expect(config["int"].integer() == 42)
        #expect(config["float"].floating() == Float(3.14))
        #expect(config["bool"].boolean() == true)
        #expect(config["null"].isNull())
    }

    @Test
    func parsesNestedStructures() throws {
        let json = "{\"outer\": {\"inner\": {\"value\": 123}}, \"array\": [1, 2, 3]}"
        let data = Data(json.utf8)

        let config = try YYJSONParser.parseToConfig(data)

        #expect(config["outer"]["inner"]["value"].integer() == 123)

        let array = config["array"].array()
        #expect(array?.count == 3)
        #expect(array?[0].integer() == 1)
        #expect(array?[1].integer() == 2)
        #expect(array?[2].integer() == 3)
    }

    @Test
    func handlesEmptyStructures() throws {
        let json = "{\"emptyObject\": {}, \"emptyArray\": []}"
        let data = Data(json.utf8)

        let config = try YYJSONParser.parseToConfig(data)

        #expect(config["emptyObject"].dictionary()?.isEmpty == true)
        #expect(config["emptyArray"].array()?.isEmpty == true)
    }

    @Test
    func throwsOnInvalidJSON() {
        let invalidJSON = Data("not valid json".utf8)

        #expect(throws: YYJSONParser.ParseError.self) {
            try YYJSONParser.parseToConfig(invalidJSON)
        }
    }
}
