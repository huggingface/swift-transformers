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

    @Test
    func parsesInfinityAndNaN() throws {
        // YYJSON_READ_ALLOW_INF_AND_NAN enables Infinity, -Infinity, and NaN
        let json = #"{"limit": [0, Infinity], "neg": -Infinity, "nan_val": NaN}"#
        let config = try YYJSONParser.parseToConfig(Data(json.utf8))
        let limit = config["limit"].array()
        #expect(limit?[1].floating() == Float.infinity)
        #expect(config["neg"].floating() == -Float.infinity)
        #expect(config["nan_val"].floating()?.isNaN == true)
    }

    @Test
    func allowsTrailingCommaInObject() throws {
        let json = #"{"a": 1, "b": 2,}"#
        let config = try YYJSONParser.parseToConfig(Data(json.utf8))

        #expect(config["a"].integer() == 1)
        #expect(config["b"].integer() == 2)
    }

    @Test
    func allowsTrailingCommaInArray() throws {
        let json = #"{"items": [1, 2, 3,]}"#
        let config = try YYJSONParser.parseToConfig(Data(json.utf8))

        let items = config["items"].array()
        #expect(items?.count == 3)
        #expect(items?[0].integer() == 1)
        #expect(items?[2].integer() == 3)
    }

    @Test
    func allowsTrailingCommaInNestedStructures() throws {
        let json = #"{"outer": {"inner": [1, 2,], "key": "val",},}"#
        let config = try YYJSONParser.parseToConfig(Data(json.utf8))

        let inner = config["outer"]["inner"].array()
        #expect(inner?.count == 2)
        #expect(config["outer"]["key"].string() == "val")
    }

    @Test
    func allowsByteOrderMarkPrefix() throws {
        let bom = "\u{FEFF}"
        let json = bom + #"{"key": "value"}"#
        let config = try YYJSONParser.parseToConfig(Data(json.utf8))

        #expect(config["key"].string() == "value")
    }

    @Test
    func allowsBOMWithTrailingComma() throws {
        let bom = "\u{FEFF}"
        let json = bom + #"{"a": 1, "b": [2, 3,],}"#
        let config = try YYJSONParser.parseToConfig(Data(json.utf8))

        #expect(config["a"].integer() == 1)
        let array = config["b"].array()
        #expect(array?.count == 2)
    }
}

// MARK: - Foundation / YYJSONParser comparison tests
// Tests that both parsers produce identical Config output for standard JSON,
// and documents known behavioral differences (e.g., BOM in string values).

@Suite("YYJSONParser comparison with Foundation JSONSerialization")
struct YYJSONParserComparisonTests {

    /// Parses JSON data through JSONSerialization -> Config path
    private func parseWithFoundation(_ data: Data) throws -> Config {
        let parsed = try JSONSerialization.jsonObject(with: data, options: []) as! [NSString: Any]
        return Config(parsed)
    }

    @Test
    func basicTypesMatch() throws {
        let json = #"{"string": "hello", "int": 42, "float": 3.14, "bool": true, "null": null}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["string"] == yyjsonConfig["string"])
        #expect(foundationConfig["int"] == yyjsonConfig["int"])
        #expect(foundationConfig["float"] == yyjsonConfig["float"])
        #expect(foundationConfig["bool"] == yyjsonConfig["bool"])
        #expect(foundationConfig["null"].isNull())
        #expect(yyjsonConfig["null"].isNull())
    }

    @Test
    func nestedStructuresMatch() throws {
        let json = #"{"outer": {"inner": {"value": 123}}, "array": [1, 2, 3]}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["outer"]["inner"]["value"] == yyjsonConfig["outer"]["inner"]["value"])

        let foundationArray = foundationConfig["array"].array()
        let yyjsonArray = yyjsonConfig["array"].array()
        #expect(foundationArray?.count == yyjsonArray?.count)
        #expect(foundationArray == yyjsonArray)
    }

    @Test
    func emptyStructuresMatch() throws {
        let json = #"{"emptyObject": {}, "emptyArray": []}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["emptyObject"].dictionary()?.isEmpty == true)
        #expect(yyjsonConfig["emptyObject"].dictionary()?.isEmpty == true)
        #expect(foundationConfig["emptyArray"].array()?.isEmpty == true)
        #expect(yyjsonConfig["emptyArray"].array()?.isEmpty == true)
    }

    @Test
    func unicodeStringsMatch() throws {
        let json = #"{"emoji": "😀🎉", "cjk": "你好世界", "mixed": "café"}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["emoji"] == yyjsonConfig["emoji"])
        #expect(foundationConfig["cjk"] == yyjsonConfig["cjk"])
        #expect(foundationConfig["mixed"] == yyjsonConfig["mixed"])
    }

    @Test
    func escapedCharactersMatch() throws {
        let json = #"{"newline": "a\nb", "tab": "a\tb", "quote": "a\"b", "backslash": "a\\b", "slash": "a\/b"}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["newline"] == yyjsonConfig["newline"])
        #expect(foundationConfig["tab"] == yyjsonConfig["tab"])
        #expect(foundationConfig["quote"] == yyjsonConfig["quote"])
        #expect(foundationConfig["backslash"] == yyjsonConfig["backslash"])
        #expect(foundationConfig["slash"] == yyjsonConfig["slash"])
    }

    @Test
    func unicodeEscapeSequencesMatch() throws {
        let json = #"{"copyright": "\u00A9", "snowman": "\u2603", "surrogate": "\uD83D\uDE00"}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["copyright"] == yyjsonConfig["copyright"])
        #expect(foundationConfig["snowman"] == yyjsonConfig["snowman"])
        #expect(foundationConfig["surrogate"] == yyjsonConfig["surrogate"])
    }

    @Test
    func nulCharacterInStringMatchesFoundation() throws {
        // yyjson supports NUL (\u0000) inside strings via yyjson_get_len for the
        // real byte length. Foundation's JSONSerialization also preserves NUL.
        let json = #"{"with_nul": "before\u0000after", "normal": "hello"}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        let foundationValue = foundationConfig["with_nul"].string()
        let yyjsonValue = yyjsonConfig["with_nul"].string()

        #expect(foundationValue == "before\0after")
        #expect(yyjsonValue == foundationValue, "YYJSONParser should preserve NUL characters")

        #expect(foundationConfig["normal"] == yyjsonConfig["normal"])
    }

    @Test
    func bomPrefixHandledByBothParsers() throws {
        let bom = "\u{FEFF}"
        let json = bom + #"{"key": "value"}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["key"] == yyjsonConfig["key"])
        #expect(yyjsonConfig["key"].string() == "value")
    }

    @Test
    func deeplyNestedStructuresMatch() throws {
        let json = #"{"l1": {"l2": {"l3": {"l4": {"value": "deep"}}}}}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["l1"]["l2"]["l3"]["l4"]["value"] == yyjsonConfig["l1"]["l2"]["l3"]["l4"]["value"])
    }

    @Test
    func mixedArraysMatch() throws {
        let json = #"{"mixed": [1, "two", 3.0, true, null, {"nested": "obj"}, [4, 5]]}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        let foundationArray = foundationConfig["mixed"].array()
        let yyjsonArray = yyjsonConfig["mixed"].array()

        #expect(foundationArray?.count == yyjsonArray?.count)
        #expect(foundationArray?[0] == yyjsonArray?[0])
        #expect(foundationArray?[1] == yyjsonArray?[1])
        #expect(foundationArray?[3] == yyjsonArray?[3])
        #expect(foundationArray?[4].isNull() == true)
        #expect(yyjsonArray?[4].isNull() == true)
        #expect(foundationArray?[5]["nested"] == yyjsonArray?[5]["nested"])
    }

    @Test
    func largeIntegersMatch() throws {
        let json = #"{"big": 9007199254740992, "negative": -9007199254740992, "small": 0}"#
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(foundationConfig["big"] == yyjsonConfig["big"])
        #expect(foundationConfig["negative"] == yyjsonConfig["negative"])
        #expect(foundationConfig["small"] == yyjsonConfig["small"])
    }

    // MARK: Known differences

    @Test
    func bomInsideStringPreservedByYYJSON() throws {
        // Foundation strips BOM (U+FEFF) from string values,
        // yyjson preserves it. yyjson's behavior is correct per the spec --
        // BOM inside a string value is part of the content, not a document marker.
        // The previous Foundation path required a bomPreservingJsonObject workaround.
        let json = "{\"token\": \"\u{feff}#\", \"normal\": \"#\"}"
        let data = Data(json.utf8)

        let foundationConfig = try parseWithFoundation(data)
        let yyjsonConfig = try YYJSONParser.parseToConfig(data)

        #expect(yyjsonConfig["token"].string() == "\u{feff}#", "yyjson preserves BOM in string values")
        #expect(foundationConfig["token"].string() == "#", "Foundation strips BOM from string values")
        #expect(foundationConfig["normal"] == yyjsonConfig["normal"])
    }
}
