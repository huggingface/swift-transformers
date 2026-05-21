//
//  ResponseParserTests.swift
//  swift-transformers
//

import Foundation
import Testing

@testable import Tokenizers

@Suite("Response Parser Tests")
struct ResponseParserTests {
    // MARK: - Spec helpers

    private func basicTemplate() throws -> ResponseTemplate {
        try ResponseTemplate(spec: [
            "defaults": ["role": "assistant"],
            "fields": [
                "content": [
                    "open": "<response>",
                    "close": "</response>",
                    "content": "text",
                ] as [String: Any]
            ] as [String: Any],
        ])
    }

    private func cohereLikeTemplate() throws -> ResponseTemplate {
        try ResponseTemplate(spec: [
            "defaults": ["role": "assistant"],
            "fields": [
                "content": [
                    "open": "<|START_RESPONSE|>",
                    "close": "<|END_RESPONSE|>",
                    "content": "text",
                ] as [String: Any],
                "thinking": [
                    "open": "<|START_THINKING|>",
                    "close": "<|END_THINKING|>",
                    "content": "text",
                ] as [String: Any],
            ] as [String: Any],
        ])
    }

    // MARK: - Happy path

    @Test("Explicit open and close yields a text field")
    func explicitOpenClose() throws {
        let template = try basicTemplate()
        let parsed = try ResponseParser.parse("<response>hello world</response>", template: template)
        #expect(parsed["role"] == .string("assistant"))
        #expect(parsed["content"] == .string("hello world"))
    }

    @Test("Implicit field captures leftover text")
    func implicitField() throws {
        let template = try ResponseTemplate(spec: [
            "defaults": ["role": "assistant"],
            "fields": [
                "content": ["content": "text"] as [String: Any],
                "thinking": [
                    "open": "<think>",
                    "close": "</think>",
                    "content": "text",
                ] as [String: Any],
            ] as [String: Any],
        ])
        let parsed = try ResponseParser.parse("<think>reasoning</think>actual answer", template: template)
        #expect(parsed["thinking"] == .string("reasoning"))
        #expect(parsed["content"] == .string("actual answer"))
    }

    @Test("Multiple regions emit events in order")
    func multipleRegionsEvents() throws {
        let template = try cohereLikeTemplate()
        var parser = try ResponseParser(template: template)
        var events = try parser.feed("<|START_THINKING|>plan<|END_THINKING|><|START_RESPONSE|>done<|END_RESPONSE|>")
        let result = try parser.finalize()
        events.append(contentsOf: result.events)
        let names = events.map { event -> String in
            switch event {
            case let .regionOpen(field): return "open:\(field)"
            case let .regionClose(field, _): return "close:\(field)"
            case let .regionChunk(field, _, _): return "chunk:\(field)"
            }
        }
        #expect(names == ["open:thinking", "chunk:thinking", "close:thinking", "open:content", "chunk:content", "close:content"])
        #expect(result.message["thinking"] == .string("plan"))
        #expect(result.message["content"] == .string("done"))
    }

    // MARK: - Content parsers

    @Test("Int content parser")
    func intContent() throws {
        let template = try ResponseTemplate(spec: [
            "defaults": ["role": "assistant"],
            "fields": [
                "count": [
                    "open": "<n>",
                    "close": "</n>",
                    "content": "int",
                ] as [String: Any]
            ] as [String: Any],
        ])
        let parsed = try ResponseParser.parse("<n>42</n>", template: template)
        #expect(parsed["count"] == .int(42))
    }

    @Test("Bool content parser")
    func boolContent() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "flag": ["open": "<f>", "close": "</f>", "content": "bool"] as [String: Any]
            ] as [String: Any]
        ])
        #expect(try ResponseParser.parse("<f>true</f>", template: template)["flag"] == .bool(true))
        #expect(try ResponseParser.parse("<f>false</f>", template: template)["flag"] == .bool(false))
    }

    @Test("JSON content parser")
    func jsonContent() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "payload": ["open": "<j>", "close": "</j>", "content": "json"] as [String: Any]
            ] as [String: Any]
        ])
        let parsed = try ResponseParser.parse(#"<j>{"a": 1, "b": [true, "x"]}</j>"#, template: template)
        guard case let .object(obj) = parsed["payload"] else {
            Issue.record("expected object")
            return
        }
        #expect(obj["a"] == .int(1))
        if case let .array(arr) = obj["b"] {
            #expect(arr.count == 2)
            #expect(arr[0] == .bool(true))
            #expect(arr[1] == .string("x"))
        } else {
            Issue.record("expected array under 'b'")
        }
    }

    @Test("kv-lines parser")
    func kvLines() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "metadata": ["open": "<meta>", "close": "</meta>", "content": "kv-lines"] as [String: Any]
            ] as [String: Any]
        ])
        let parsed = try ResponseParser.parse("<meta>name: alice\nage: 30</meta>", template: template)
        if case let .object(obj) = parsed["metadata"] {
            #expect(obj["name"] == .string("alice"))
            #expect(obj["age"] == .string("30"))
        } else {
            Issue.record("expected object")
        }
    }

    @Test("xml-inline with named groups")
    func xmlInline() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "tags": [
                    "open": "<tags>",
                    "close": "</tags>",
                    "content": "xml-inline",
                    "content_args": [
                        "tag_pattern": #"<(?<key>\w+)=(?<value>[^>]+)>"#
                    ] as [String: Any],
                ] as [String: Any]
            ] as [String: Any]
        ])
        let parsed = try ResponseParser.parse("<tags><name=foo><age=10></tags>", template: template)
        if case let .object(obj) = parsed["tags"] {
            #expect(obj["name"] == .string("foo"))
            #expect(obj["age"] == .string("10"))
        } else {
            Issue.record("expected object")
        }
    }

    // MARK: - Transform template

    @Test("Declarative transform resolves captures and content placeholders")
    func transformDeclarative() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "tool": [
                    "open_pattern": #"<tool name=(?<name>\w+)>"#,
                    "close": "</tool>",
                    "content": "text",
                    "transform": [
                        "name": "{name}",
                        "args": "{content}",
                    ] as [String: Any],
                ] as [String: Any]
            ] as [String: Any]
        ])
        let parsed = try ResponseParser.parse(
            "<tool name=search>weather in paris</tool>",
            template: template
        )
        if case let .object(obj) = parsed["tool"] {
            #expect(obj["name"] == .string("search"))
            #expect(obj["args"] == .string("weather in paris"))
        } else {
            Issue.record("expected object")
        }
    }

    @Test("Mixing a placeholder with literal text is rejected at load")
    func mixedPlaceholderRejected() {
        #expect(throws: ResponseParserError.self) {
            _ = try ResponseTemplate(spec: [
                "fields": [
                    "x": [
                        "open_pattern": #"<x v=(?<v>\w+)>"#,
                        "close": "</x>",
                        "transform": ["msg": "hello {v}"] as [String: Any],
                    ] as [String: Any]
                ] as [String: Any]
            ])
        }
    }

    @Test("transform_each applies template per list element with keys in scope")
    func transformEach() throws {
        // Mirrors the Cohere-style template: parsed content is a list of dicts,
        // each containing `tool_name` + `parameters` keys that get unpacked
        // into the transform scope.
        let template = try ResponseTemplate(spec: [
            "fields": [
                "tool_calls": [
                    "open": "<calls>",
                    "close": "</calls>",
                    "content": "json",
                    "transform_each": true,
                    "transform": [
                        "type": "function",
                        "function": [
                            "name": "{tool_name}",
                            "arguments": "{parameters}",
                        ] as [String: Any],
                    ] as [String: Any],
                ] as [String: Any]
            ] as [String: Any]
        ])
        let parsed = try ResponseParser.parse(
            #"<calls>[{"tool_name": "search", "parameters": {"q": "weather"}}, {"tool_name": "get_time", "parameters": {}}]</calls>"#,
            template: template
        )
        guard case let .array(calls) = parsed["tool_calls"] else {
            Issue.record("expected tool_calls array")
            return
        }
        #expect(calls.count == 2)
        if case let .object(first) = calls.first, case let .object(fn) = first["function"] {
            #expect(fn["name"] == .string("search"))
            #expect(fn["arguments"] == .object(["q": .string("weather")]))
        } else {
            Issue.record("malformed first tool call")
        }
    }

    // MARK: - Literal anchors

    @Test("Literal-list open and close accept any alternative")
    func literalListAlternation() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "x": [
                    "open": ["<a>", "<bb>"],
                    "close": ["</a>", "</bb>"],
                    "content": "text",
                ] as [String: Any]
            ] as [String: Any]
        ])
        for (open, close) in [("<a>", "</a>"), ("<bb>", "</bb>"), ("<a>", "</bb>")] {
            let parsed = try ResponseParser.parse("\(open)hi\(close)", template: template)
            #expect(parsed["x"] == .string("hi"))
        }
    }

    @Test("EOS literal terminates an implicit field at end of stream")
    func eosLiteral() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "content": ["close": "eos", "content": "text"] as [String: Any]
            ] as [String: Any]
        ])
        let parsed = try ResponseParser.parse("just text", template: template)
        #expect(parsed["content"] == .string("just text"))
    }

    // MARK: - Streaming behavior

    @Test("Streaming feed emits chunks progressively")
    func streamingChunks() throws {
        let template = try basicTemplate()
        var parser = try ResponseParser(template: template)
        // Split mid-content; the parser should hold back enough bytes that the close tag is still detectable.
        let chunks = ["<response>hel", "lo wor", "ld</response>"]
        var allEvents: [ResponseEvent] = []
        for chunk in chunks { allEvents.append(contentsOf: try parser.feed(chunk)) }
        let result = try parser.finalize()
        allEvents.append(contentsOf: result.events)
        #expect(result.message["content"] == .string("hello world"))
        // Reconstruct text from chunk events.
        let reconstructed = allEvents.reduce(into: "") { acc, event in
            if case let .regionChunk(_, text, _) = event { acc += text }
        }
        #expect(reconstructed == "hello world")
    }

    @Test("Literal-list with non-prefix-overlap streams without hold")
    func literalListStreamsWithoutHold() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "content": [
                    "close": ["<turn|>", "<|tool_response>", "<eos>"],
                    "content": "text",
                ] as [String: Any]
            ] as [String: Any]
        ])
        var parser = try ResponseParser(template: template)
        let plain = String(repeating: "x", count: 32)
        let events = try parser.feed(plain)
        let flushed = events.reduce(into: "") { acc, event in
            if case let .regionChunk(_, text, _) = event { acc += text }
        }
        // Plain text has zero prefix-overlap with any literal: nothing is held back.
        #expect(flushed == plain)
        _ = try parser.finalize()
    }

    // MARK: - Defaults / required / repeats

    @Test("Defaults survive when field is absent")
    func defaultsSurvive() throws {
        let template = try basicTemplate()
        let parsed = try ResponseParser.parse("nothing matching here", template: template)
        #expect(parsed == ["role": .string("assistant")])
    }

    @Test("optional: false missing field throws")
    func optionalFalseMissing() throws {
        let template = try ResponseTemplate(spec: [
            "defaults": ["role": "assistant"],
            "fields": [
                "content": [
                    "open": "<r>",
                    "close": "</r>",
                    "content": "text",
                    "optional": false,
                ] as [String: Any]
            ] as [String: Any],
        ])
        #expect(throws: ResponseParserError.self) {
            _ = try ResponseParser.parse("nothing here", template: template)
        }
    }

    @Test("repeats: true collects into an array")
    func repeatsCollects() throws {
        let template = try ResponseTemplate(spec: [
            "fields": [
                "item": [
                    "open": "<i>",
                    "close": "</i>",
                    "content": "text",
                    "repeats": true,
                ] as [String: Any]
            ] as [String: Any]
        ])
        let parsed = try ResponseParser.parse("<i>a</i><i>b</i><i>c</i>", template: template)
        #expect(parsed["item"] == .array([.string("a"), .string("b"), .string("c")]))
    }

    // MARK: - Prefix and start_anchor

    @Test("start_anchor truncates to last anchor occurrence")
    func startAnchorTruncates() throws {
        let template = try ResponseTemplate(spec: [
            "defaults": ["role": "assistant"],
            "start_anchor": "<|im_start|>assistant\n",
            "fields": [
                "thinking": [
                    "open": "<think>",
                    "close": "</think>",
                    "content": "text",
                ] as [String: Any]
            ] as [String: Any],
        ])
        let prefix = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>"
        var parser = try ResponseParser(template: template, prefix: prefix)
        let opens = parser.initialEvents.compactMap { event -> String? in
            if case let .regionOpen(field) = event { return field }
            return nil
        }
        #expect(opens == ["thinking"])
        _ = try parser.feed("reasoning</think>")
        let result = try parser.finalize()
        #expect(result.message["thinking"] == .string("reasoning"))
    }

    // MARK: - Validation errors

    @Test("Two implicit fields rejected")
    func twoImplicitsRejected() {
        #expect(throws: ResponseParserError.self) {
            _ = try ResponseTemplate(spec: [
                "fields": [
                    "a": ["content": "text"] as [String: Any],
                    "b": ["content": "text"] as [String: Any],
                ] as [String: Any]
            ])
        }
    }

    @Test("Named groups without transform rejected")
    func namedGroupsWithoutTransformRejected() {
        #expect(throws: ResponseParserError.self) {
            _ = try ResponseTemplate(spec: [
                "fields": [
                    "tool": [
                        "open_pattern": #"<tool name=(?<name>\w+)>"#,
                        "close": "</tool>",
                        "content": "text",
                    ] as [String: Any]
                ] as [String: Any]
            ])
        }
    }

    @Test("Unknown content parser rejected")
    func unknownContentRejected() {
        #expect(throws: ResponseParserError.self) {
            _ = try ResponseTemplate(spec: [
                "fields": [
                    "x": ["open": "[", "close": "]", "content": "not-a-real-parser"] as [String: Any]
                ] as [String: Any]
            ])
        }
    }

    // MARK: - Async stream helper

    @Test("Async stream helper yields events and final message")
    func asyncStreamHelper() async throws {
        let template = try cohereLikeTemplate()
        let chunks = [
            "<|START_THINKING|>plan",
            "ning",
            "<|END_THINKING|>",
            "<|START_RESPONSE|>done<|END_RESPONSE|>",
        ]
        let source = AsyncStream<String> { continuation in
            for chunk in chunks { continuation.yield(chunk) }
            continuation.finish()
        }

        let (events, messageTask) = ResponseParser.stream(from: source, template: template)
        var collected: [ResponseEvent] = []
        for try await event in events { collected.append(event) }
        let message = try await messageTask.value

        #expect(message["thinking"] == .string("planning"))
        #expect(message["content"] == .string("done"))

        let kinds = collected.map { event -> String in
            switch event {
            case let .regionOpen(f): return "open:\(f)"
            case let .regionClose(f, _): return "close:\(f)"
            case .regionChunk: return "chunk"
            }
        }
        #expect(kinds.first == "open:thinking")
        #expect(kinds.contains("close:thinking"))
        #expect(kinds.contains("open:content"))
        #expect(kinds.last == "close:content")
    }
}
