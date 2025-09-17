import Foundation
import Testing

@testable import Jinja

@Suite("Value Tests")
struct ValueTests {
    @Test("Initialization from Any")
    func initFromAny() throws {
        #expect(try Value(any: nil) == Value.null)
        #expect(try Value(any: "hello") == Value.string("hello"))
        #expect(try Value(any: 42) == Value.int(42))
        #expect(try Value(any: 3.14) == Value.double(3.14))
        #expect(try Value(any: Float(2.5)) == Value.double(2.5))
        #expect(try Value(any: true) == Value.boolean(true))

        let arrayValue = try Value(any: [1, "test", nil])
        if case let .array(values) = arrayValue {
            #expect(values.count == 3)
            #expect(values[0] == Value.int(1))
            #expect(values[1] == Value.string("test"))
            #expect(values[2] == Value.null)
        } else {
            Issue.record("Expected array value")
        }

        let dictValue = try Value(any: ["key": "value", "num": 42])
        if case let .object(dict) = dictValue {
            #expect(dict["key"] == Value.string("value"))
            #expect(dict["num"] == Value.int(42))
        } else {
            Issue.record("Expected object value")
        }

        #expect(throws: JinjaError.self) {
            _ = try Value(any: NSObject())
        }
    }

    @Test("Literal conformances")
    func literals() {
        let stringValue: Value = "test"
        #expect(stringValue == Value.string("test"))

        let intValue: Value = 42
        #expect(intValue == Value.int(42))

        let doubleValue: Value = 3.14
        #expect(doubleValue == Value.double(3.14))

        let boolValue: Value = true
        #expect(boolValue == Value.boolean(true))

        let arrayValue: Value = [1, 2, 3]
        if case let .array(values) = arrayValue {
            #expect(values.count == 3)
            #expect(values[0] == Value.int(1))
            #expect(values[1] == Value.int(2))
            #expect(values[2] == Value.int(3))
        } else {
            Issue.record("Expected array value")
        }

        let dictValue: Value = ["a": 1, "b": 2]
        if case let .object(dict) = dictValue {
            #expect(dict["a"] == Value.int(1))
            #expect(dict["b"] == Value.int(2))
        } else {
            Issue.record("Expected object value")
        }

        let nilValue: Value = nil
        #expect(nilValue == Value.null)
    }

    @Test("CustomStringConvertible conformance")
    func description() {
        #expect(Value.string("test").description == "test")
        #expect(Value.int(42).description == "42")
        #expect(Value.double(3.14).description == "3.14")
        #expect(Value.boolean(true).description == "true")
        #expect(Value.boolean(false).description == "false")
        #expect(Value.null.description == "")
        #expect(Value.undefined.description == "")
        #expect(Value.array([Value.int(1), Value.int(2)]).description == "[1, 2]")
        #expect(Value.object(["a": Value.int(1)]).description == "{a: 1}")
    }

    @Test("isTruthy behavior")
    func isTruthy() {
        #expect(Value.null.isTruthy == false)
        #expect(Value.undefined.isTruthy == false)
        #expect(Value.boolean(true).isTruthy == true)
        #expect(Value.boolean(false).isTruthy == false)
        #expect(Value.string("").isTruthy == false)
        #expect(Value.string("hello").isTruthy == true)
        #expect(Value.double(0.0).isTruthy == false)
        #expect(Value.double(1.0).isTruthy == true)
        #expect(Value.int(0).isTruthy == false)
        #expect(Value.int(1).isTruthy == true)
        #expect(Value.array([]).isTruthy == false)
        #expect(Value.array([Value.int(1)]).isTruthy == true)
        #expect(Value.object([:]).isTruthy == false)
        #expect(Value.object(["key": Value.string("value")]).isTruthy == true)
    }

    @Test("Encodable conformance")
    func encodable() throws {
        let encoder = JSONEncoder()

        // Test primitive values
        let stringData = try encoder.encode(Value.string("hello"))
        let stringJSON = String(data: stringData, encoding: .utf8)!
        #expect(stringJSON == "\"hello\"")

        let intData = try encoder.encode(Value.int(42))
        let intJSON = String(data: intData, encoding: .utf8)!
        #expect(intJSON == "42")

        let numberData = try encoder.encode(Value.double(3.14))
        let numberJSON = String(data: numberData, encoding: .utf8)!
        #expect(numberJSON == "3.14")

        let boolData = try encoder.encode(Value.boolean(true))
        let boolJSON = String(data: boolData, encoding: .utf8)!
        #expect(boolJSON == "true")

        let nullData = try encoder.encode(Value.null)
        let nullJSON = String(data: nullData, encoding: .utf8)!
        #expect(nullJSON == "null")

        let undefinedData = try encoder.encode(Value.undefined)
        let undefinedJSON = String(data: undefinedData, encoding: .utf8)!
        #expect(undefinedJSON == "null")

        // Test array encoding
        let arrayValue = Value.array([Value.int(1), Value.string("test"), Value.boolean(false)])
        let arrayData = try encoder.encode(arrayValue)
        let arrayJSON = String(data: arrayData, encoding: .utf8)!
        #expect(arrayJSON == "[1,\"test\",false]")

        // Test object encoding
        var objectDict = OrderedDictionary<String, Value>()
        objectDict["name"] = Value.string("John")
        objectDict["age"] = Value.int(30)
        objectDict["active"] = Value.boolean(true)
        let objectValue = Value.object(objectDict)
        let objectData = try encoder.encode(objectValue)
        let objectJSON = String(data: objectData, encoding: .utf8)!
        #expect(objectJSON.contains("\"name\":\"John\""))
        #expect(objectJSON.contains("\"age\":30"))
        #expect(objectJSON.contains("\"active\":true"))

        // Test nested structures
        let nestedArray = Value.array([
            Value.string("item1"),
            Value.object(["nested": Value.int(42)]),
            Value.array([Value.boolean(true), Value.null]),
        ])
        let nestedArrayData = try encoder.encode(nestedArray)
        let nestedArrayJSON = String(data: nestedArrayData, encoding: .utf8)!
        #expect(nestedArrayJSON.contains("\"item1\""))
        #expect(nestedArrayJSON.contains("\"nested\":42"))
        #expect(nestedArrayJSON.contains("true"))
        #expect(nestedArrayJSON.contains("null"))

        // Test function encoding should throw
        let functionValue = Value.function { _, _, _ in Value.null }
        #expect(throws: EncodingError.self) {
            _ = try encoder.encode(functionValue)
        }
    }

    @Test("Decodable conformance")
    func decodable() throws {
        let decoder = JSONDecoder()

        // Test primitive values
        let stringData = "\"hello\"".data(using: .utf8)!
        let stringValue = try decoder.decode(Value.self, from: stringData)
        #expect(stringValue == Value.string("hello"))

        let intData = "42".data(using: .utf8)!
        let intValue = try decoder.decode(Value.self, from: intData)
        #expect(intValue == Value.int(42))

        let numberData = "3.14".data(using: .utf8)!
        let numberValue = try decoder.decode(Value.self, from: numberData)
        #expect(numberValue == Value.double(3.14))

        let boolData = "true".data(using: .utf8)!
        let boolValue = try decoder.decode(Value.self, from: boolData)
        #expect(boolValue == Value.boolean(true))

        let nullData = "null".data(using: .utf8)!
        let nullValue = try decoder.decode(Value.self, from: nullData)
        #expect(nullValue == Value.null)

        // Test array decoding
        let arrayData = "[1,\"test\",false]".data(using: .utf8)!
        let arrayValue = try decoder.decode(Value.self, from: arrayData)
        if case let .array(values) = arrayValue {
            #expect(values.count == 3)
            #expect(values[0] == Value.int(1))
            #expect(values[1] == Value.string("test"))
            #expect(values[2] == Value.boolean(false))
        } else {
            Issue.record("Expected array value")
        }

        // Test object decoding
        let objectData = "{\"name\":\"John\",\"age\":30,\"active\":true}".data(using: .utf8)!
        let objectValue = try decoder.decode(Value.self, from: objectData)
        if case let .object(dict) = objectValue {
            #expect(dict["name"] == Value.string("John"))
            #expect(dict["age"] == Value.int(30))
            #expect(dict["active"] == Value.boolean(true))
        } else {
            Issue.record("Expected object value")
        }

        // Test nested structures
        let nestedData = "[\"item1\",{\"nested\":42},[true,null]]".data(using: .utf8)!
        let nestedValue = try decoder.decode(Value.self, from: nestedData)
        if case let .array(values) = nestedValue {
            #expect(values.count == 3)
            #expect(values[0] == Value.string("item1"))

            if case let .object(nestedDict) = values[1] {
                #expect(nestedDict["nested"] == Value.int(42))
            } else {
                Issue.record("Expected nested object")
            }

            if case let .array(nestedArray) = values[2] {
                #expect(nestedArray.count == 2)
                #expect(nestedArray[0] == Value.boolean(true))
                #expect(nestedArray[1] == Value.null)
            } else {
                Issue.record("Expected nested array")
            }
        } else {
            Issue.record("Expected array value")
        }
    }

    @Test("Round-trip encoding/decoding")
    func roundTrip() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        let testValues: [Value] = [
            .string("hello"),
            .int(42),
            .double(3.14),
            .boolean(true),
            .boolean(false),
            .null,
            .array([Value.int(1), Value.string("test"), Value.boolean(false)]),
            .object(["key1": Value.string("value1"), "key2": Value.int(123)]),
        ]

        for originalValue in testValues {
            let data = try encoder.encode(originalValue)
            let decodedValue = try decoder.decode(Value.self, from: data)
            #expect(decodedValue == originalValue, "Round-trip failed for \(originalValue)")
        }
    }

    @Test("Complex nested structures")
    func complexNestedStructures() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Create a complex nested structure
        var complexDict = OrderedDictionary<String, Value>()
        complexDict["users"] = Value.array([
            Value.object([
                "id": Value.int(1),
                "name": Value.string("Alice"),
                "active": Value.boolean(true),
                "scores": Value.array([Value.double(95.5), Value.double(87.2), Value.double(92.0)]),
            ]),
            Value.object([
                "id": Value.int(2),
                "name": Value.string("Bob"),
                "active": Value.boolean(false),
                "scores": Value.array([Value.double(78.1), Value.double(81.5)]),
            ]),
        ])
        complexDict["metadata"] = Value.object([
            "total": Value.int(2),
            "lastUpdated": Value.string("2024-01-01T00:00:00Z"),
            "version": Value.double(1.1),
        ])
        complexDict["settings"] = Value.null

        let complexValue = Value.object(complexDict)

        // Encode and decode
        let data = try encoder.encode(complexValue)
        let decodedValue = try decoder.decode(Value.self, from: data)

        // Verify specific nested values
        if case let .object(decodedDict) = decodedValue {
            if case let .array(users) = decodedDict["users"] {
                #expect(users.count == 2)

                if case let .object(user1) = users[0] {
                    #expect(user1["name"] == Value.string("Alice"))
                    #expect(user1["active"] == Value.boolean(true))

                    if case let .array(scores) = user1["scores"] {
                        #expect(scores.count == 3)
                        #expect(scores[0] == Value.double(95.5))
                    } else {
                        Issue.record("Expected scores array")
                    }
                } else {
                    Issue.record("Expected user1 object")
                }
            } else {
                Issue.record("Expected users array")
            }

            if case let .object(metadata) = decodedDict["metadata"] {
                #expect(metadata["total"] == Value.int(2))
                #expect(metadata["version"] == Value.double(1.1))
            } else {
                Issue.record("Expected metadata object")
            }

            #expect(decodedDict["settings"] == Value.null)
        } else {
            Issue.record("Expected root object")
        }
    }

    @Test("Edge cases and error handling")
    func edgeCases() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Test empty arrays and objects
        let emptyArray = Value.array([])
        let emptyArrayData = try encoder.encode(emptyArray)
        let decodedEmptyArray = try decoder.decode(Value.self, from: emptyArrayData)
        #expect(decodedEmptyArray == emptyArray)

        let emptyObject = Value.object([:])
        let emptyObjectData = try encoder.encode(emptyObject)
        let decodedEmptyObject = try decoder.decode(Value.self, from: emptyObjectData)
        #expect(decodedEmptyObject == emptyObject)

        // Test arrays with mixed types
        let mixedArray = Value.array([
            Value.string("text"),
            Value.int(42),
            Value.double(3.14),
            Value.boolean(true),
            Value.null,
            Value.array([Value.int(1), Value.int(2)]),
            Value.object(["nested": Value.string("value")]),
        ])
        let mixedArrayData = try encoder.encode(mixedArray)
        let decodedMixedArray = try decoder.decode(Value.self, from: mixedArrayData)
        #expect(decodedMixedArray == mixedArray)

        // Test objects with various key types (all should be strings in JSON)
        var objectWithVariousKeys = OrderedDictionary<String, Value>()
        objectWithVariousKeys["stringKey"] = Value.string("stringValue")
        objectWithVariousKeys["numberKey"] = Value.double(123.45)
        objectWithVariousKeys["booleanKey"] = Value.boolean(false)
        objectWithVariousKeys["nullKey"] = Value.null
        objectWithVariousKeys["arrayKey"] = Value.array([Value.int(1), Value.int(2)])
        objectWithVariousKeys["objectKey"] = Value.object(["nested": Value.string("nestedValue")])
    }

    @Test("JSON string escaping and unescaping")
    func jsonStringEscaping() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Test strings with special characters
        let specialString = Value.string("Hello \"World\" with\nnewlines\tand\ttabs")
        let specialStringData = try encoder.encode(specialString)
        let specialStringJSON = String(data: specialStringData, encoding: .utf8)!
        #expect(specialStringJSON.contains("\\\""))
        #expect(specialStringJSON.contains("\\n"))
        #expect(specialStringJSON.contains("\\t"))

        let decodedSpecialString = try decoder.decode(Value.self, from: specialStringData)
        #expect(decodedSpecialString == specialString)

        // Test strings with Unicode characters
        let unicodeString = Value.string("Hello 世界 🌍")
        let unicodeStringData = try encoder.encode(unicodeString)
        let decodedUnicodeString = try decoder.decode(Value.self, from: unicodeStringData)
        #expect(decodedUnicodeString == unicodeString)

        // Test empty string
        let emptyString = Value.string("")
        let emptyStringData = try encoder.encode(emptyString)
        let emptyStringJSON = String(data: emptyStringData, encoding: .utf8)!
        #expect(emptyStringJSON == "\"\"")

        let decodedEmptyString = try decoder.decode(Value.self, from: emptyStringData)
        #expect(decodedEmptyString == emptyString)
    }

    @Test("Error handling for invalid JSON")
    func invalidJsonHandling() throws {
        let decoder = JSONDecoder()

        // Test invalid JSON syntax
        let invalidJSONData = "{invalid json}".data(using: .utf8)!
        #expect(throws: DecodingError.self) {
            _ = try decoder.decode(Value.self, from: invalidJSONData)
        }

        // Test incomplete JSON
        let incompleteJSONData = "{\"key\":".data(using: .utf8)!
        #expect(throws: DecodingError.self) {
            _ = try decoder.decode(Value.self, from: incompleteJSONData)
        }

        // Test unsupported JSON types (like undefined in JSON)
        // Note: JSON doesn't have undefined, only null, so this should work
        let nullData = "null".data(using: .utf8)!
        let nullValue = try decoder.decode(Value.self, from: nullData)
        #expect(nullValue == Value.null)
    }
}
