//
//  ConfigTests.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 13.03.25.
//

import Foundation
import Jinja
import XCTest

@testable import Hub

class ConfigGeneralTests: XCTestCase {
    func testHashable() throws {
        let testCases: [(Config.Value, Config.Value)] = [
            (.integer(1), .integer(2)),
            (.string("a"), .string("2")),
            (.boolean(true), .string("T")),
            (.boolean(true), .boolean(false)),
            (.floating(1.1), .floating(1.1000001)),
            (.token(1, "a"), .token(1, "b")),
            (.token(1, "a"), .token(2, "a")),
            (.dictionary(["1": .null]), .dictionary(["1": .integer(1)])),
            (.dictionary(["1": .integer(10)]), .dictionary(["2": .integer(10)])),
            (.array([.string("1"), .string("2")]), .array([.string("1"), .string("3")])),
            (.array([.integer(1), .integer(2)]), .array([.integer(2), .integer(1)])),
            (.array([.boolean(true), .boolean(false)]), .array([.boolean(true), .boolean(true)])),
        ]

        for (lhs, rhs) in testCases {
            var lhsh = Hasher()
            var rhsh = Hasher()

            lhs.hash(into: &lhsh)
            rhs.hash(into: &rhsh)

            XCTAssertNotEqual(lhsh.finalize(), rhsh.finalize())
        }
    }
}

class ConfigAsLiteralTests: XCTestCase {
    func testStringLiteral() throws {
        let val: Config.Value = "test"
        XCTAssertEqual(val, .string("test"))
    }

    func testIntegerLiteral() throws {
        let val: Config.Value = 678
        XCTAssertEqual(val, .integer(678))
    }

    func testBooleanLiteral() throws {
        let val: Config.Value = true
        XCTAssertEqual(val, .boolean(true))
    }

    func testFloatLiteral() throws {
        let val: Config.Value = 1.1
        XCTAssertEqual(val, .floating(1.1))
    }

    func testDictionaryLiteral() throws {
        let cfg: Config = ["key": .floating(1.1)]
        XCTAssertEqual(Double(cfg.properties["key"]?.floating ?? 0), 1.1, accuracy: 1e-6)
    }

    func testArrayLiteral() throws {
        let val: Config.Value = [.floating(1.1), .floating(1.2)]
        XCTAssertEqual(val.array?[0], .floating(1.1))
        XCTAssertEqual(val.array?[1], .floating(1.2))
    }
}

class ConfigAccessorsTests: XCTestCase {
    func testKeySubscript() throws {
        let cfg: Config = ["key": .floating(1.1)]

        XCTAssertEqual(cfg["key"], .floating(1.1))
        XCTAssertNil(cfg["non_existent"])
    }

    func testIndexSubscript() throws {
        let val: Config.Value = [.integer(1), .integer(2), .integer(3), .integer(4)]

        XCTAssertEqual(val[1], .integer(2))
        XCTAssertNil(val[99])
        XCTAssertNil(val[-1])
    }

    func testDynamicLookup() throws {
        let cfg: Config = ["model_type": "bert"]

        XCTAssertEqual(cfg["model_type"], "bert")
        XCTAssertEqual(cfg.modelType, "bert")
        XCTAssertEqual(cfg.model_type, "bert")
        XCTAssertNil(cfg.unknown_key)
    }

    func testArray() throws {
        let val: Config.Value = [.integer(1), .integer(2), .integer(3), .integer(4)]

        XCTAssertEqual(val.array, [.integer(1), .integer(2), .integer(3), .integer(4)])
        XCTAssertNil(val.dictionary)
    }

    func testArrayOfStrings() throws {
        let val: Config.Value = ["a", "b", "c"]
        let expected: [Config.Value] = [.string("a"), .string("b"), .string("c")]
        XCTAssertEqual(val.array, expected)

        let strings = val.array?.compactMap(\.string)
        XCTAssertEqual(strings, ["a", "b", "c"])

        let bds = val.array?.compactMap(\.string)
        XCTAssertEqual(bds, ["a", "b", "c"])
    }

    func testDictionary() throws {
        let cfg: Config = ["a": 1, "b": 2, "c": 3, "d": 4]

        XCTAssertEqual(cfg.properties.count, 4)
        XCTAssertEqual(cfg["a"], .integer(1))
        XCTAssertNil(cfg["a"]?.array)
    }

    func testDictionaryOfConfigs() throws {
        let cfg: Config = ["a": .dictionary(["child": 1]), "b": .dictionary(["child": 2])]
        XCTAssertEqual(cfg["a"]?.dictionary?["child"], .integer(1))
        XCTAssertEqual(cfg["b"]?.dictionary?["child"], .integer(2))
    }
}

class ConfigCodableTests: XCTestCase {
    func testCompleteHappyExample() throws {
        let cfg: Config = [
            "dict_of_floats": .dictionary(Config(["key1": .floating(1.1)])),
            "dict_of_ints": .dictionary(Config(["key2": .integer(100)])),
            "dict_of_strings": .dictionary(Config(["key3": .string("abc")])),
            "dict_of_bools": .dictionary(Config(["key4": .boolean(false)])),
            "dict_of_dicts": .dictionary(Config(["key5": .dictionary(Config(["key_inside": .integer(99)]))])),
            "dict_of_tokens": .dictionary(Config(["key6": .token(12, "dfe")])),
            "arr_empty": [],
            "arr_of_ints": [1, 2, 3],
            "arr_of_floats": [1.1, 1.2],
            "arr_of_strings": ["a", "b"],
            "arr_of_bools": [true, false],
            "arr_of_dicts": [.dictionary(Config(["key7": .floating(1.1)])), .dictionary(Config(["key8": .floating(1.2)]))],
            "arr_of_tokens": [.token(1, "a"), .token(2, "b")],
            "int": 678,
            "float": 1.1,
            "string": "test",
            "bool": true,
            "token": .token(1, "test"),
            "null": .null,
        ]

        let data = try JSONEncoder().encode(cfg)
        let got = try JSONDecoder().decode(Config.self, from: data)

        XCTAssertEqual(got, cfg)
        XCTAssertEqual(got["dict_of_floats"]?.dictionary?["key1"], .floating(1.1))
        XCTAssertEqual(got["dict_of_ints"]?.dictionary?["key2"], .integer(100))
        XCTAssertEqual(got["dict_of_strings"]?.dictionary?["key3"], .string("abc"))
        XCTAssertEqual(got["dict_of_bools"]?.dictionary?["key4"], .boolean(false))
        XCTAssertEqual(got["dict_of_dicts"]?.dictionary?["key5"]?.dictionary?["key_inside"], .integer(99))
        XCTAssertEqual(got["dict_of_tokens"]?.dictionary?["key6"]?.token?.0, 12)
        XCTAssertEqual(got["dict_of_tokens"]?.dictionary?["key6"]?.token?.1, "dfe")
        XCTAssertEqual(got["arr_empty"]?.array?.count, 0)
        XCTAssertEqual(got["arr_of_ints"], .array([.integer(1), .integer(2), .integer(3)]))
        XCTAssertEqual(got["arr_of_floats"], .array([.floating(1.1), .floating(1.2)]))
        XCTAssertEqual(got["arr_of_strings"], .array([.string("a"), .string("b")]))
        XCTAssertEqual(got["arr_of_bools"], .array([.boolean(true), .boolean(false)]))
        XCTAssertEqual(got["arr_of_dicts"]?[1]?.dictionary?["key8"], .floating(1.2))
        let token1 = got["arr_of_tokens"]?[1]?.token
        XCTAssertEqual(token1?.0, 2)
        XCTAssertEqual(token1?.1, "b")
        XCTAssertNil(got["arr_of_tokens"]?[2]?.token)
        XCTAssertEqual(got["int"], .integer(678))
        XCTAssertEqual(got["float"], .floating(1.1))
        XCTAssertEqual(got["string"], .string("test"))
        XCTAssertEqual(got["bool"], .boolean(true))
        let mainToken = got["token"]?.token
        XCTAssertEqual(mainToken?.0, 1)
        XCTAssertEqual(mainToken?.1, "test")
        XCTAssertEqual(got["null"], .null)
    }
}

class ConfigEquatableTests: XCTestCase {
    func testString() throws {
        let val: Config.Value = "a"

        XCTAssertEqual(val, "a")
        XCTAssertEqual(val.string, "a")
        XCTAssertEqual(val.string(or: "b"), "a")
        XCTAssertEqual(val.string, "a")
        XCTAssertEqual(val.string(or: "b"), "a")
    }

    func testInteger() throws {
        let val: Config.Value = 1
        XCTAssertEqual(val, 1)
        XCTAssertEqual(val.integer, 1)
        XCTAssertEqual(val.integer(or: 2), 1)
    }

    func testFloating() throws {
        let testCases: [(Config.Value, Float)] = [
            (1.1, 1.1),
            (1, 1.0),
        ]

        for (val, exp) in testCases {
            XCTAssertEqual(val.floating, exp)
            XCTAssertEqual(val.floating(or: 2.2), exp)
        }
    }

    func testBoolean() throws {
        let testCases: [(Config.Value, Bool)] = [
            (true, true),
            (1, true),
            ("T", true),
            ("t", true),
            ("TRUE", true),
            ("True", true),
            ("true", true),
            ("F", false),
            ("f", false),
            ("FALSE", false),
            ("False", false),
            ("false", false),
        ]

        for (val, exp) in testCases {
            XCTAssertEqual(val.boolean, exp)
            XCTAssertEqual(val.boolean(or: !exp), exp)
        }
    }

    func testToken() throws {
        let val = Config.Value.token(1, "a")
        let exp: (UInt, String) = (1, "a")

        XCTAssert(val.token! == exp)
        XCTAssert(val.token(or: (2, "b")) == exp)
    }

    func testDictionary() throws {
        let cfg: Config = ["a": 1]
        XCTAssertEqual(cfg["a"], .integer(1))
    }
}

class ConfigTextEncodingTests: XCTestCase {
    private func createFile(with content: String, encoding: String.Encoding, fileName: String) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent(fileName)
        guard let data = content.data(using: encoding) else {
            throw NSError(domain: "EncodingError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Could not encode string with \(encoding)"])
        }
        try data.write(to: fileURL)
        return fileURL
    }

    func testUtf16() throws {
        let json = """
            {
              "a": ["val_1", "val_2"],
              "b": 2,
              "c": [[10, "tkn_1"], [12, "tkn_2"], [4, "tkn_3"]],
              "d": false,
              "e": {
                "e_1": 1.1,
                "e_2": [1, 2, 3]
              },
              "f": null
            }
        """

        let urlUTF8 = try createFile(with: json, encoding: .utf8, fileName: "config_utf8.json")
        let urlUTF16LE = try createFile(with: json, encoding: .utf16LittleEndian, fileName: "config_utf16_le.json")
        let urlUTF16BE = try createFile(with: json, encoding: .utf16BigEndian, fileName: "config_utf16_be.json")

        let dataUTF8 = try Data(contentsOf: urlUTF8)
        let dataUTF16LE = try Data(contentsOf: urlUTF16LE)
        let dataUTF16BE = try Data(contentsOf: urlUTF16BE)

        XCTAssertNotEqual(dataUTF8.count, dataUTF16LE.count)
        XCTAssertNotEqual(dataUTF8.count, dataUTF16BE.count)

        let decoder = JSONDecoder()
        let configUTF8 = try decoder.decode(Config.self, from: dataUTF8)
        let configUTF16LE = try decoder.decode(Config.self, from: dataUTF16LE)
        let configUTF16BE = try decoder.decode(Config.self, from: dataUTF16BE)

        XCTAssertEqual(configUTF8, configUTF16LE)
        XCTAssertEqual(configUTF8, configUTF16BE)

        try FileManager.default.removeItem(at: urlUTF8)
        try FileManager.default.removeItem(at: urlUTF16LE)
        try FileManager.default.removeItem(at: urlUTF16BE)
    }

    func testUnicode() {
        // These are two different characters
        let json = "{\"vocab\": {\"à\": 1, \"à\": 2}}"
        let data = json.data(using: .utf8)!
        let dict = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
        let config = Config(dict)

        let vocab = config["vocab"]?.dictionary?.properties
        XCTAssertEqual(vocab?.count, 2)
    }
}

class ConfigTemplatingTests: XCTestCase {
    func testCompleteHappyExample() throws {
        let cfg = Config([
            "dict_of_floats": .dictionary(["key1": .floating(1.1)]),
            "dict_of_tokens": .dictionary(["key6": .token(12, "dfe")]),
            "arr_empty": .array([]),
            "arr_of_ints": .array([.integer(1), .integer(2), .integer(3)]),
            "arr_of_floats": .array([.floating(1.1), .floating(1.2)]),
            "arr_of_strings": .array([.string("tre"), .string("jeq")]),
            "arr_of_bools": .array([.boolean(true), .boolean(false)]),
            "arr_of_dicts": .array([.dictionary(["key7": .floating(1.1)]), .dictionary(["key8": .floating(1.2)])]),
            "arr_of_tokens": .array([.token(1, "ghz"), .token(2, "pkr")]),
            "int": .integer(678),
            "float": .floating(1.1),
            "string": .string("hha"),
            "bool": .boolean(true),
            "token": .token(1, "iop"),
            "null": .null,
        ])
        let template = """
        {{ config["dict_of_floats"]["key1"] }}
        {{ config["dict_of_tokens"]["key6"]["12"] }}
        {{ config["arr_of_ints"][0] }}
        {{ config["arr_of_ints"][1] }}
        {{ config["arr_of_ints"][2] }}
        {{ config["arr_of_floats"][0] }}
        {{ config["arr_of_floats"][1] }}
        {{ config["arr_of_strings"][0] }}
        {{ config["arr_of_strings"][1] }}
        {{ config["arr_of_bools"][0] }}
        {{ config["arr_of_bools"][1] }}
        {{ config["arr_of_dicts"][0]["key7"] }}
        {{ config["arr_of_dicts"][1]["key8"] }}
        {{ config["arr_of_tokens"][0]["1"] }}
        {{ config["arr_of_tokens"][1]["2"] }}
        {{ config["int"] }}
        {{ config["float"] }}
        {{ config["string"] }}
        {{ config["bool"] }}
        {{ config["token"]["1"] }}
        """
        let exp = """
        1.1
        dfe
        1
        2
        3
        1.1
        1.2
        tre
        jeq
        true
        false
        1.1
        1.2
        ghz
        pkr
        678
        1.1
        hha
        true
        iop
        """

        let got = try Template(template).render([
            "config": cfg.properties.mapValues { $0.toJinjaCompatible() },
        ])

        XCTAssertEqual(got, exp)
    }
}
