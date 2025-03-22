//
//  ConfigTests.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 13.03.25.
//

import Foundation
import Jinja
import Testing

@testable import Hub

@Suite struct ConfigGeneral {
    @Test(arguments: [
        (Config.Data.integer(1), Config.Data.integer(2)),
        (Config.Data.string("a"), Config.Data.string("2")),
        (Config.Data.boolean(true), Config.Data.string("T")),
        (Config.Data.boolean(true), Config.Data.boolean(false)),
        (Config.Data.floating(1.1), Config.Data.floating(1.1000001)),
        (Config.Data.token((1, "a")), Config.Data.token((1, "b"))),
        (Config.Data.token((1, "a")), Config.Data.token((2, "a"))),
        (Config.Data.dictionary(["1": Config()]), Config.Data.dictionary(["1": 1])),
        (Config.Data.dictionary(["1": 10]), Config.Data.dictionary(["2": 10])),
        (Config.Data.array(["1", "2"]), Config.Data.array(["1", "3"])),
        (Config.Data.array([1, 2]), Config.Data.array([2, 1])),
        (Config.Data.array([true, false]), Config.Data.array([true, true])),
    ])
    func hashable(lhs: Config.Data, rhs: Config.Data) async throws {
        var lhsh = Hasher()
        var rhsh = Hasher()

        lhs.hash(into: &lhsh)
        rhs.hash(into: &rhsh)

        #expect(lhsh.finalize() != rhsh.finalize())
    }
}

@Suite struct ConfigAsLiteral {
    @Test("Config can be represented as a string literal")
    func stringLiteral() async throws {
        let cfg: Config = "test"

        #expect(cfg == "test")
    }

    @Test("Config can be represented as a integer literal")
    func integerLiteral() async throws {
        let cfg: Config = 678

        #expect(cfg == 678)
    }

    @Test("Config can be represented as a boolean literal")
    func booleanLiteral() async throws {
        let cfg: Config = true

        #expect(cfg == true)
    }

    @Test("Config can be represented as a boolean literal")
    func floatLiteral() async throws {
        let cfg: Config = 1.1

        #expect(cfg == 1.1)
    }

    @Test("Config can be represented as a dictionary literal")
    func dictionaryLiteral() async throws {
        let cfg: Config = ["key": 1.1]

        #expect(cfg["key"].floating(or: 0) == 1.1)
    }

    @Test("Config can be represented as a dictionary literal")
    func arrayLiteral() async throws {
        let cfg: Config = [1.1, 1.2]

        #expect(cfg[0] == 1.1)
        #expect(cfg[1] == 1.2)
    }
}

@Suite struct ConfigAccessors {
    @Test("Config can be accessed via key subscript")
    func keySubscript() async throws {
        let cfg: Config = ["key": 1.1]

        #expect(cfg["key"] == 1.1)
        #expect(cfg["non_existent"].isNull())
        #expect(cfg[1].isNull())
    }

    @Test("Config can be accessed via index subscript")
    func indexSubscript() async throws {
        let cfg: Config = [1, 2, 3, 4]

        #expect(cfg[1] == 2)
        #expect(cfg[99].isNull())
        #expect(cfg[-1].isNull())
    }

    @Test("Config can be converted to an array")
    func array() async throws {
        let cfg: Config = [1, 2, 3, 4]

        #expect(cfg.array() == [1, 2, 3, 4])
        #expect(cfg.get() == [1, 2, 3, 4])
        #expect(cfg.get(or: []) == [1, 2, 3, 4])
        #expect(cfg["fake_key"].isNull())
        #expect(cfg.dictionary() == nil)
        #expect(cfg.dictionary(or: ["a": 1]) == ["a": 1])
    }

    @Test("Config can be converted to an array of strings")
    func arrayOfStrings() async throws {
        let cfg: Config = ["a", "b", "c"]

        #expect(cfg.array() == ["a", "b", "c"])
        #expect(cfg.get() == ["a", "b", "c"])
        #expect(cfg.get() == [BinaryDistinctString("a"), BinaryDistinctString("b"), BinaryDistinctString("c")])
        #expect(cfg.get(or: []) == [BinaryDistinctString("a"), BinaryDistinctString("b"), BinaryDistinctString("c")])
        #expect(cfg.get(or: []) == ["a", "b", "c"])
        #expect(cfg.dictionary() == nil)
        #expect(cfg.dictionary(or: ["a": 1]) == ["a": 1])
    }

    @Test("Config can be converted to an array of strings")
    func arrayOfConfigs() async throws {
        let cfg: Config = [Config("a"), Config("b")]

        #expect(cfg.array() == ["a", "b"])
        #expect(cfg.get() == ["a", "b"])
        #expect(cfg.get() == [BinaryDistinctString("a"), BinaryDistinctString("b")])
        #expect(cfg.get(or: []) == [BinaryDistinctString("a"), BinaryDistinctString("b")])
        #expect(cfg.get(or: []) == ["a", "b"])
        #expect(cfg.dictionary() == nil)
        #expect(cfg.dictionary(or: ["a": 1]) == ["a": 1])
    }

    @Test("Config can be converted to a dictionary of ints")
    func dictionary() async throws {
        let cfg: Config = ["a": 1, "b": 2, "c": 3, "d": 4]

        #expect(cfg.dictionary() == ["a": 1, "b": 2, "c": 3, "d": 4])
        #expect(cfg.get() == ["a": 1, "b": 2, "c": 3, "d": 4])
        #expect(cfg.get(or: [:]) == ["a": 1, "b": 2, "c": 3, "d": 4])
        #expect(cfg[666].isNull())
        #expect(cfg.array() == nil)
        #expect(cfg.array(or: ["a"]) == ["a"])
    }
    @Test("Config can be converted to a dictionary of configs")
    func dictionaryOfConfigs() async throws {
        let cfg: Config = ["a": .init([1, 2]), "b": .init([3, 4])]
        let exp = [BinaryDistinctString("a"): Config([1, 2]), BinaryDistinctString("b"): Config([3, 4])]

        #expect(cfg.dictionary() == exp)
        #expect(cfg.get() == exp)
        #expect(cfg.get(or: [:]) == exp)
        #expect(cfg[666].isNull())
        #expect(cfg.array() == nil)
        #expect(cfg.array(or: ["a"]) == ["a"])
    }
}

@Suite struct ConfigCodable {
    @Test("Config can be serialized and deserialized")
    func completeHappyExample() async throws {
        let cfg: Config = [
            "dict_of_floats": ["key1": 1.1],
            "dict_of_ints": ["key2": 100],
            "dict_of_strings": ["key3": "abc"],
            "dict_of_bools": ["key4": false],
            "dict_of_dicts": ["key5": ["key_inside": 99]],
            "dict_of_tokens": ["key6": .init((12, "dfe"))],
            "arr_empty": [],
            "arr_of_ints": [1, 2, 3],
            "arr_of_floats": [1.1, 1.2],
            "arr_of_strings": ["a", "b"],
            "arr_of_bools": [true, false],
            "arr_of_dicts": [["key7": 1.1], ["key8": 1.2]],
            "arr_of_tokens": [.init((1, "a")), .init((2, "b"))],
            "int": 678,
            "float": 1.1,
            "string": "test",
            "bool": true,
            "token": .init((1, "test")),
            "null": Config(),
        ]

        let data = try JSONEncoder().encode(cfg)

        let got = try JSONDecoder().decode(Config.self, from: data)

        #expect(got == cfg)
        #expect(got["dict_of_floats"]["key1"] == 1.1)
        #expect(got["dict_of_ints"]["key2"] == 100)
        #expect(got["dict_of_strings"]["key3"] == "abc")
        #expect(got["dict_of_bools"]["key4"] == false)
        #expect(got["dict_of_dicts"]["key5"]["key_inside"] == 99)
        #expect(got["dict_of_tokens"]["key6"].token()?.0 == 12)
        #expect(got["dict_of_tokens"]["key6"].token()?.1 == "dfe")
        #expect(got["arr_empty"].array()?.count == 0)
        #expect(got["arr_of_ints"] == [1, 2, 3])
        #expect(got["arr_of_floats"] == [1.1, 1.2])
        #expect(got["arr_of_strings"] == ["a", "b"])
        #expect(got["arr_of_bools"] == [true, false])
        #expect(got["arr_of_dicts"][1]["key8"] == 1.2)
        #expect(got["arr_of_tokens"][1].token(or: (0, "")) == (2, "b"))
        #expect(got["arr_of_tokens"][2].token() == nil)
        #expect(got["int"] == 678)
        #expect(got["float"] == 1.1)
        #expect(got["string"] == "test")
        #expect(got["bool"] == true)
        #expect(got["token"].token(or: (0, "")) == (1, "test"))
        #expect(got["null"].isNull())
    }
}

@Suite struct ConfigEquatable {
    @Test func string() async throws {
        let cfg = Config("a")

        #expect(cfg == "a")
        #expect(cfg.get() == "a")
        #expect(cfg.get(or: "b") == "a")
        #expect(cfg.string() == "a")
        #expect(cfg.string(or: "b") == "a")
        #expect(cfg.get() == BinaryDistinctString("a"))
        #expect(cfg.get(or: "b") == BinaryDistinctString("a"))
        #expect(cfg.binaryDistinctString() == "a")
        #expect(cfg.binaryDistinctString(or: "b") == "a")
    }

    @Test func integer() async throws {
        let cfg = Config(1)

        #expect(cfg == 1)
        #expect(cfg.get() == 1)
        #expect(cfg.get(or: 2) == 1)
        #expect(cfg.integer() == 1)
        #expect(cfg.integer(or: 2) == 1)
    }

    @Test(arguments: [
        (Config(1.1), 1.1 as Float),
        (Config(1), 1.0 as Float),
    ])
    func floating(cfg: Config, exp: Float) async throws {
        #expect(cfg == .init(exp))
        #expect(cfg.get() == exp)
        #expect(cfg.get(or: 2.2) == exp)
        #expect(cfg.floating() == exp)
        #expect(cfg.floating(or: 2.2) == exp)
    }

    @Test(arguments: [
        (Config(true), true),
        (Config(1), true),
        (Config("T"), true),
        (Config("t"), true),
        (Config("TRUE"), true),
        (Config("True"), true),
        (Config("true"), true),
        (Config("F"), false),
        (Config("f"), false),
        (Config("FALSE"), false),
        (Config("False"), false),
        (Config("false"), false),
    ])
    func boolean(cfg: Config, exp: Bool) async throws {
        #expect(cfg.get() == exp)
        #expect(cfg.get(or: !exp) == exp)
        #expect(cfg.boolean() == exp)
        #expect(cfg.boolean(or: !exp) == exp)
    }

    @Test func token() async throws {
        let cfg = Config((1, "a"))
        let exp: (UInt, String) = (1, "a")

        #expect(cfg == .init((1, "a")))
        #expect(cfg.get()! == exp)
        #expect(cfg.get(or: (2, "b")) == exp)
        #expect(cfg.token()! == exp)
        #expect(cfg.token(or: (2, "b")) == exp)
    }

    @Test(arguments: [
        (Config(["a": 1]), 1),
        (Config(["a": 2] as [NSString: Any]), 2),
        (Config(["a": 3] as [NSString: Config]), 3),
        (Config([BinaryDistinctString("a"): 4] as [BinaryDistinctString: Config]), 4),
        (Config(["a": Config(5)]), 5),
        (Config(["a": 6]), 6),
        (Config((BinaryDistinctString("a"), 7)), 7),
    ])
    func dictionary(cfg: Config, exp: Int) async throws {
        #expect(cfg["a"] == Config(exp))
        #expect(cfg.get(or: [:])["a"] == Config(exp))
    }
}

@Suite struct ConfigTextEncoding {
    private func createFile(with content: String, encoding: String.Encoding, fileName: String) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent(fileName)
        guard let data = content.data(using: encoding) else {
            throw NSError(domain: "EncodingError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Could not encode string with \(encoding)"])
        }
        try data.write(to: fileURL)
        return fileURL
    }

    @Test func utf16() async throws {
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

        #expect(dataUTF8.count != dataUTF16LE.count)
        #expect(dataUTF8.count != dataUTF16BE.count)

        let decoder = JSONDecoder()
        let configUTF8 = try decoder.decode(Config.self, from: dataUTF8)
        let configUTF16LE = try decoder.decode(Config.self, from: dataUTF16LE)
        let configUTF16BE = try decoder.decode(Config.self, from: dataUTF16BE)

        #expect(configUTF8 == configUTF16LE)
        #expect(configUTF8 == configUTF16BE)

        try FileManager.default.removeItem(at: urlUTF8)
        try FileManager.default.removeItem(at: urlUTF16LE)
        try FileManager.default.removeItem(at: urlUTF16BE)
    }

    @Test func unicode() {
        // These are two different characters
        let json = "{\"vocab\": {\"à\": 1, \"à\": 2}}"
        let data = json.data(using: .utf8)
        let dict = try! JSONSerialization.jsonObject(with: data!, options: []) as! [NSString: Any]
        let config = Config(dict)

        let vocab = config["vocab"].dictionary(or: [:])

        #expect(vocab.count == 2)
    }
}

@Suite struct ConfigTemplating {
    @Test func completeHappyExample() async throws {
        let cfg = Config([
            "dict_of_floats": ["key1": 1.1],
            "dict_of_tokens": ["key6": .init((12, "dfe"))],
            "arr_empty": [],
            "arr_of_ints": [1, 2, 3],
            "arr_of_floats": [1.1, 1.2],
            "arr_of_strings": ["tre", "jeq"],
            "arr_of_bools": [true, false],
            "arr_of_dicts": [["key7": 1.1], ["key8": 1.2]],
            "arr_of_tokens": [.init((1, "ghz")), .init((2, "pkr"))],
            "int": 678,
            "float": 1.1,
            "string": "hha",
            "bool": true,
            "token": .init((1, "iop")),
            "null": Config(),
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
            "config": cfg.toJinjaCompatible()
        ])

        #expect(got == exp)
    }
}
