//
//  ResponseParserE2ETests.swift
//  swift-transformers
//
//  End-to-end tests that hit the Hugging Face Hub. Gated on the
//  RUN_HUB_E2E_TESTS=1 environment variable so default `swift test`
//  runs stay offline.
//

import Foundation
import Testing

@testable import Tokenizers

@Suite("Response Parser E2E Tests")
struct ResponseParserE2ETests {
    private func loadTokenizer(_ name: String, revision: String) async throws -> PreTrainedTokenizer {
        let tokenizer = try await AutoTokenizer.from(pretrained: name, revision: revision)
        guard let pretrained = tokenizer as? PreTrainedTokenizer else {
            throw TokenizerError.unsupportedTokenizer("expected PreTrainedTokenizer, got \(type(of: tokenizer))")
        }
        return pretrained
    }

    @Test("SmolLM3 thinking+content parses without a transform")
    func smolLM3ThinkingAndContent() async throws {
        let tokenizer = try await loadTokenizer("pcuenq/SmolLM3-3B", revision: "refs/pr/1")
        #expect(tokenizer.hasResponseTemplate)

        // Thinking + final-answer response; no tool call, so no transform path is exercised.
        let output = """
            <think>
            The user wants a friendly greeting. Keep it short and welcoming.
            </think>

            Hello! How can I help you today?<|im_end|>
            """
        let parsed = try tokenizer.parseResponse(output)

        #expect(parsed["role"] == .string("assistant"))
        if case let .string(thinking) = parsed["thinking"] {
            #expect(thinking.contains("friendly greeting"))
        } else {
            Issue.record("expected 'thinking' string, got \(String(describing: parsed["thinking"]))")
        }
        if case let .string(content) = parsed["content"] {
            #expect(content.contains("How can I help"))
        } else {
            Issue.record("expected 'content' string, got \(String(describing: parsed["content"]))")
        }
    }

    @Test("SmolLM3 tool-call parses")
    func smolLM3ToolCall() async throws {
        let tokenizer = try await loadTokenizer("pcuenq/SmolLM3-3B", revision: "refs/pr/1")
        #expect(tokenizer.hasResponseTemplate)

        let output = """
            <think>
            The user wants the weather. I should call the weather tool.
            </think>

            <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
            """

        let parsed = try tokenizer.parseResponse(output)

        #expect(parsed["role"] == .string("assistant"))
        if case let .string(thinking) = parsed["thinking"] {
            #expect(thinking.contains("weather tool"))
        } else {
            Issue.record("expected 'thinking' string, got \(String(describing: parsed["thinking"]))")
        }

        guard case let .array(toolCalls) = parsed["tool_calls"] else {
            Issue.record("expected 'tool_calls' array, got \(String(describing: parsed["tool_calls"]))")
            return
        }
        #expect(toolCalls.count == 1)

        guard case let .object(call) = toolCalls.first,
            case .string("function") = call["type"],
            case let .object(function) = call["function"]
        else {
            Issue.record("malformed tool call: \(String(describing: toolCalls.first))")
            return
        }
        #expect(function["name"] == .string("get_weather"))
        if case let .object(args) = function["arguments"] {
            #expect(args["city"] == .string("Paris"))
        } else {
            Issue.record("expected arguments object, got \(String(describing: function["arguments"]))")
        }
    }

    @Test("Gemma 4 tool-call parses")
    func gemmaToolCallWithTransform() async throws {
        let tokenizer = try await loadTokenizer("pcuenq/gemma-4-E2B-it", revision: "refs/pr/1")
        #expect(tokenizer.hasResponseTemplate)

        // Synthetic model output matching Gemma 4 chat template
        let output = """
            <|channel>thought
            The user is asking for the current temperature in Paris.
            <channel|>\
            <|tool_call>call:get_current_temperature\
            {detail_level:0,location:<|\"|>Paris, France<|\"|>,unit:<|\"|>celsius<|\"|>}\
            <tool_call|>
            """

        let parsed = try tokenizer.parseResponse(output)

        #expect(parsed["role"] == .string("assistant"))
        if case let .string(thinking) = parsed["thinking"] {
            #expect(thinking.contains("Paris"))
        } else {
            Issue.record("expected 'thinking' string, got \(String(describing: parsed["thinking"]))")
        }

        guard case let .array(toolCalls) = parsed["tool_calls"] else {
            Issue.record("expected 'tool_calls' array, got \(String(describing: parsed["tool_calls"]))")
            return
        }
        #expect(toolCalls.count == 1)

        guard case let .object(call) = toolCalls.first,
            case .string("function") = call["type"],
            case let .object(function) = call["function"]
        else {
            Issue.record("malformed tool call: \(String(describing: toolCalls.first))")
            return
        }
        #expect(function["name"] == .string("get_current_temperature"))

        guard case let .object(args) = function["arguments"] else {
            Issue.record("expected arguments object, got \(String(describing: function["arguments"]))")
            return
        }
        #expect(args["location"] == .string("Paris, France"))
        #expect(args["unit"] == .string("celsius"))
        #expect(args["detail_level"] == .int(0))
    }
}
