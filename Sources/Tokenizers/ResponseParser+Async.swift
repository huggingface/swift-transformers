//
//  ResponseParser+Async.swift
//  swift-transformers
//
//  Async-stream helper layered over the sync `ResponseParser` core. Drives
//  the parser from an upstream `AsyncSequence<String>` and exposes:
//    - an `AsyncThrowingStream<ResponseEvent, Error>` of live events
//    - a `Task<ParsedMessage, Error>` for the final parsed message
//

import Foundation

public extension ResponseParser {
    /// Drive the parser from an upstream chunk stream.
    ///
    /// Iterate `events` to render live; await `message` for the final parsed
    /// message dict. If parsing throws, both surfaces fail with the same error
    /// (the stream finishes with `throwing` and the task throws).
    static func stream<Source: AsyncSequence & Sendable>(
        from source: Source,
        template: ResponseTemplate,
        prefix: String? = nil
    ) -> (events: AsyncThrowingStream<ResponseEvent, Error>, message: Task<ParsedMessage, Error>)
    where Source.Element == String {
        let (stream, continuation) = AsyncThrowingStream<ResponseEvent, Error>.makeStream()
        let task = Task { () throws -> ParsedMessage in
            do {
                var parser = try ResponseParser(template: template, prefix: prefix)
                for event in parser.initialEvents { continuation.yield(event) }
                for try await chunk in source {
                    for event in try parser.feed(chunk) { continuation.yield(event) }
                }
                let result = try parser.finalize()
                for event in result.events { continuation.yield(event) }
                continuation.finish()
                return result.message
            } catch {
                continuation.finish(throwing: error)
                throw error
            }
        }
        return (stream, task)
    }
}
