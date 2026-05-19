//
//  Tokenizer+ResponseParser.swift
//  swift-transformers
//
//  Wires the response parser into `PreTrainedTokenizer`. Mirrors the
//  `apply_chat_template` / `parse_response` symmetry from the Python side:
//  tokenizers carry a `response_template` in their config, and callers can
//  parse model output back into a structured message dict — either all at
//  once or as a live stream.
//

import Foundation
import Hub

public extension PreTrainedTokenizer {
    /// One-shot parse of a fully-buffered response.
    ///
    /// - Parameters:
    ///   - text: Model output text.
    ///   - prefix: The chat prompt the model is continuing. The parser needs
    ///     this because chat templates often prefill part of the assistant
    ///     message (e.g. a `<think>` opening tag), and parsing the prefill
    ///     is what teaches the parser the current region. Pass the prompt
    ///     string returned by `applyChatTemplate(...)` decoded back to text,
    ///     or the equivalent verbatim.
    ///   - transform: Optional resolver for `transform` expressions in the
    ///     template (typically a jmespath evaluator).
    /// - Returns: Parsed message dict.
    /// - Throws: `ResponseParserError.invalidSpec` if no template is configured,
    ///   or any error from parsing.
    func parseResponse(
        _ text: String,
        prefix: String? = nil,
        transform: ResponseTransform? = nil
    ) throws -> ParsedMessage {
        guard let template = try responseTemplate() else {
            throw ResponseParserError.invalidSpec("Tokenizer has no response_template configured")
        }
        return try ResponseParser.parse(text, template: template, prefix: prefix, transform: transform)
    }

    /// One-shot parse with the `prefix` provided as token IDs (e.g. the output
    /// of `applyChatTemplate(...)`). Tokens are decoded back to text before
    /// being fed to the parser.
    func parseResponse(
        _ text: String,
        prefix: [Int],
        transform: ResponseTransform? = nil
    ) throws -> ParsedMessage {
        let prefixText = decode(tokens: prefix)
        return try parseResponse(text, prefix: prefixText, transform: transform)
    }

    /// Construct a streaming `ResponseParser` configured against this
    /// tokenizer's `response_template`. The returned parser is move-only;
    /// see `ResponseParser` for usage.
    func responseParser(
        prefix: String? = nil,
        transform: ResponseTransform? = nil
    ) throws -> ResponseParser {
        guard let template = try responseTemplate() else {
            throw ResponseParserError.invalidSpec("Tokenizer has no response_template configured")
        }
        return try ResponseParser(template: template, prefix: prefix, transform: transform)
    }

    /// Streaming parser whose `prefix` is supplied as token IDs.
    func responseParser(
        prefix: [Int],
        transform: ResponseTransform? = nil
    ) throws -> ResponseParser {
        let prefixText = decode(tokens: prefix)
        return try responseParser(prefix: prefixText, transform: transform)
    }
}
