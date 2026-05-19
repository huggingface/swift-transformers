//
//  ResponseParser.swift
//  swift-transformers
//
//  Streaming response parser. Buffers text and emits `ResponseEvent`s as
//  declared regions open and close. The core is a small state machine driven
//  by a `ResponseTemplate`; see the Python `transformers.utils.chat_parsing`
//  reference implementation for the same algorithm.
//

import Foundation

public enum ResponseEvent: Sendable, Equatable {
    case regionOpen(field: String)
    case regionChunk(field: String, text: String, dirty: Bool)
    case regionClose(field: String, value: ParsedValue)
}

/// Stateful streaming parser. Move-only: once `finalize()` is called, the
/// instance is consumed and cannot be reused. This makes "finalize then
/// continue feeding" a compile-time error rather than a runtime one.
public struct ResponseParser: ~Copyable {
    /// Events produced while consuming the `prefix` passed at init. Replay
    /// these into your renderer before feeding any model output, so any
    /// regions opened by the chat template prefill are reflected on screen.
    public private(set) var initialEvents: [ResponseEvent]

    private let template: ResponseTemplate
    private let transform: ResponseTransform?
    private let implicitName: String?

    private var buffer: NSMutableString
    private var pos: Int
    private var output: ParsedMessage
    /// Current region, or `nil` for the null sink. Tracks `implicitName` whenever
    /// no explicit region is open.
    private var current: String?
    private var captures: [String: String]
    private var body: NSMutableString
    private var opened: Bool

    public init(
        template: ResponseTemplate,
        prefix: String? = nil,
        transform: ResponseTransform? = nil
    ) throws {
        self.template = template
        self.transform = transform
        self.implicitName = template.implicit
        self.buffer = NSMutableString()
        self.pos = 0
        self.output = template.defaults
        self.current = template.implicit
        self.captures = [:]
        self.body = NSMutableString()
        self.opened = false
        self.initialEvents = []

        if let prefix, !prefix.isEmpty {
            let truncated = template.truncatePastLastAnchor(prefix)
            if !truncated.isEmpty {
                buffer.append(truncated)
                var events: [ResponseEvent] = []
                try process(events: &events, eos: false)
                initialEvents = events
            }
        }
    }

    /// One-shot parse of a fully-buffered response. Streaming consumers
    /// should construct a `ResponseParser` directly.
    public static func parse(
        _ text: String,
        template: ResponseTemplate,
        prefix: String? = nil,
        transform: ResponseTransform? = nil
    ) throws -> ParsedMessage {
        var parser = try ResponseParser(template: template, prefix: prefix, transform: transform)
        _ = try parser.feed(text)
        return try parser.finalize().message
    }

    public mutating func feed(_ text: String) throws -> [ResponseEvent] {
        if !text.isEmpty { buffer.append(text) }
        var events: [ResponseEvent] = []
        try process(events: &events, eos: false)
        return events
    }

    public consuming func finalize() throws -> (message: ParsedMessage, events: [ResponseEvent]) {
        var events: [ResponseEvent] = []
        try process(events: &events, eos: true)
        var missing: [String] = []
        for (name, field) in template.fields where !field.optional && output[name] == nil {
            missing.append(name)
        }
        if !missing.isEmpty {
            throw ResponseParserError.missingRequiredFields(missing)
        }
        let defaults = template.defaults
        var filtered: ParsedMessage = [:]
        for (k, v) in output {
            if defaults[k] != nil || !v.isEmpty {
                filtered[k] = v
            }
        }
        return (filtered, events)
    }

    // MARK: - Algorithm

    private mutating func process(events: inout [ResponseEvent], eos: Bool) throws {
        while true {
            let watch = watchlist()
            var best = bestMatch(watch: watch)
            // Mid-stream, a match that reaches the end of the buffer is
            // ambiguous: zero-width alternations and pattern delimiters may
            // still extend with more input. Non-zero-width literal matches
            // can never extend, so we always commit those.
            if let b = best, !eos, Self.shouldDefer(field: b.field, match: b.match, bufLen: buffer.length, isOpen: b.kind == .open) {
                best = nil
            }

            if let b = best {
                let mStart = b.match.range.location
                let mEnd = b.match.range.location + b.match.range.length
                if mStart > pos {
                    let between = buffer.substring(with: NSRange(location: pos, length: mStart - pos))
                    accumulate(text: between, events: &events)
                }
                pos = mEnd
                switch b.kind {
                case .open:
                    try closeCurrent(events: &events)
                    openExplicit(field: b.field, match: b.match, events: &events)
                case .close:
                    let hadContent = opened
                    try closeCurrent(events: &events)
                    // A zero-width close on an already-empty region would just
                    // re-fire forever — bail out so we make progress.
                    if !hadContent && mStart == mEnd { return }
                }
                continue
            }

            if eos {
                if pos < buffer.length {
                    let rest = buffer.substring(with: NSRange(location: pos, length: buffer.length - pos))
                    accumulate(text: rest, events: &events)
                    pos = buffer.length
                }
                try closeCurrent(events: &events)
                return
            }
            if watch.isEmpty {
                // Nothing to wait for — flush everything buffered into the current region.
                if pos < buffer.length {
                    let rest = buffer.substring(with: NSRange(location: pos, length: buffer.length - pos))
                    accumulate(text: rest, events: &events)
                    pos = buffer.length
                }
                return
            }
            let hold = maxHold(watch: watch)
            let safeEnd = buffer.length - hold
            if safeEnd > pos {
                let safe = buffer.substring(with: NSRange(location: pos, length: safeEnd - pos))
                accumulate(text: safe, events: &events)
                pos = safeEnd
            }
            return
        }
    }

    private enum WatchKind { case open, close }
    private struct Watch { let kind: WatchKind; let field: ResponseField }
    private struct Best { let kind: WatchKind; let field: ResponseField; let match: NSTextCheckingResult }

    private func watchlist() -> [Watch] {
        if let c = current, c != implicitName {
            guard let f = template.fields[c], f.close != nil else { return [] }
            return [Watch(kind: .close, field: f)]
        }
        var out: [Watch] = []
        for f in template.fields.values where f.open != nil {
            out.append(Watch(kind: .open, field: f))
        }
        if let implicit = implicitName, let f = template.fields[implicit], f.close != nil {
            out.append(Watch(kind: .close, field: f))
        }
        return out
    }

    private func bestMatch(watch: [Watch]) -> Best? {
        var best: Best?
        var bestStart = Int.max
        var bestNegLen = Int.max
        var bestKindRank = Int.max
        var bestName = ""
        let searchRange = NSRange(location: pos, length: buffer.length - pos)
        for w in watch {
            let anchor: ResponseAnchor? = (w.kind == .open) ? w.field.open : w.field.close
            guard let anchor else { continue }
            guard let m = anchor.regex.firstMatch(in: buffer as String, options: [], range: searchRange) else { continue }
            let start = m.range.location
            let length = m.range.length
            let kindRank = (w.kind == .open) ? 0 : 1
            let name = w.field.name
            // Tuple comparison: earliest start, then longest length, then opens before closes, then name.
            let negLen = -length
            if start < bestStart
                || (start == bestStart && negLen < bestNegLen)
                || (start == bestStart && negLen == bestNegLen && kindRank < bestKindRank)
                || (start == bestStart && negLen == bestNegLen && kindRank == bestKindRank && name < bestName)
            {
                best = Best(kind: w.kind, field: w.field, match: m)
                bestStart = start
                bestNegLen = negLen
                bestKindRank = kindRank
                bestName = name
            }
        }
        return best
    }

    private static func shouldDefer(field: ResponseField, match: NSTextCheckingResult, bufLen: Int, isOpen: Bool) -> Bool {
        let mEnd = match.range.location + match.range.length
        if mEnd != bufLen { return false }
        let anchor: ResponseAnchor? = isOpen ? field.open : field.close
        guard let anchor else { return false }
        return anchor.literals == nil || anchor.literalCanExtend || match.range.length == 0
    }

    private func maxHold(watch: [Watch]) -> Int {
        var hold = 0
        for w in watch {
            let anchor: ResponseAnchor? = (w.kind == .open) ? w.field.open : w.field.close
            guard let anchor else { continue }
            hold = max(hold, patternHold(literals: anchor.literals))
        }
        return hold
    }

    private func patternHold(literals: [String]?) -> Int {
        let avail = buffer.length - pos
        if avail <= 0 { return 0 }
        guard let literals else { return min(avail, 64) }
        var best = 0
        for literal in literals {
            let litNS = literal as NSString
            let maxK = min(litNS.length - 1, avail)
            if maxK <= best { continue }
            var k = maxK
            while k > best {
                let tail = buffer.substring(with: NSRange(location: buffer.length - k, length: k))
                let prefix = litNS.substring(to: k)
                if tail == prefix {
                    best = k
                    break
                }
                k -= 1
            }
        }
        return best
    }

    private mutating func accumulate(text: String, events: inout [ResponseEvent]) {
        if text.isEmpty || current == nil { return }
        guard let fld = template.fields[current!] else { return }
        if !opened {
            events.append(.regionOpen(field: fld.name))
            opened = true
        }
        body.append(text)
        let dirty = !streamableParsers.contains(fld.content.kind)
        events.append(.regionChunk(field: fld.name, text: text, dirty: dirty))
    }

    private mutating func openExplicit(field: ResponseField, match: NSTextCheckingResult, events: inout [ResponseEvent]) {
        current = field.name
        captures = [:]
        if let anchor = field.open {
            for name in anchor.namedGroups {
                let r = match.range(withName: name)
                if r.location != NSNotFound {
                    captures[name] = buffer.substring(with: r)
                }
            }
        }
        body.setString("")
        opened = true
        events.append(.regionOpen(field: field.name))
    }

    private mutating func closeCurrent(events: inout [ResponseEvent]) throws {
        guard let c = current, opened, let fld = template.fields[c] else {
            resetToImplicit()
            return
        }
        let bodyString = body as String
        let value = try processField(body: bodyString, field: fld, captures: captures, transform: transform)
        if fld.repeats {
            switch output[c] {
            case .array(var existing):
                existing.append(value)
                output[c] = .array(existing)
            default:
                output[c] = .array([value])
            }
        } else {
            output[c] = value
        }
        events.append(.regionClose(field: c, value: value))
        resetToImplicit()
    }

    private mutating func resetToImplicit() {
        current = implicitName
        captures = [:]
        body.setString("")
        opened = false
    }

}
