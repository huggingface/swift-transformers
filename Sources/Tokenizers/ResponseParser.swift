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

/// One-shot convenience for a fully-buffered response. Streaming consumers
/// should use `ResponseParser` directly.
public func parseResponse(
    _ text: String,
    template: ResponseTemplate,
    prefix: String? = nil,
    transform: ResponseTransform? = nil
) throws -> ParsedMessage {
    var parser = try ResponseParser(template: template, prefix: prefix, transform: transform)
    _ = try parser.feed(text)
    let result = try parser.finalize()
    return result.message
}

/// Stateful streaming parser. Move-only: once `finalize()` is called, the
/// instance is consumed and cannot be reused. This makes "finalize then
/// continue feeding" a compile-time error rather than a runtime one.
public struct ResponseParser: ~Copyable {
    /// Events produced while consuming the `prefix` passed at init. Replay
    /// these into your renderer before feeding any model output, so any
    /// regions opened by the chat template prefill are reflected on screen.
    public let initialEvents: [ResponseEvent]

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

        var initial: [ResponseEvent] = []
        if let prefix, !prefix.isEmpty {
            let truncated = template.truncatePastLastAnchor(prefix)
            if !truncated.isEmpty {
                self.buffer.append(truncated)
                try Self.process(
                    buffer: self.buffer, posRef: &self.pos,
                    output: &self.output, current: &self.current,
                    captures: &self.captures, body: self.body, opened: &self.opened,
                    template: template, transform: transform, events: &initial, eos: false
                )
            }
        }
        self.initialEvents = initial
    }

    public mutating func feed(_ text: String) throws -> [ResponseEvent] {
        if !text.isEmpty { buffer.append(text) }
        var events: [ResponseEvent] = []
        try Self.process(
            buffer: buffer, posRef: &pos,
            output: &output, current: &current,
            captures: &captures, body: body, opened: &opened,
            template: template, transform: transform, events: &events, eos: false
        )
        return events
    }

    public consuming func finalize() throws -> (message: ParsedMessage, events: [ResponseEvent]) {
        var events: [ResponseEvent] = []
        try Self.process(
            buffer: buffer, posRef: &pos,
            output: &output, current: &current,
            captures: &captures, body: body, opened: &opened,
            template: template, transform: transform, events: &events, eos: true
        )
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
            if defaults[k] != nil || !Self.isEmpty(v) {
                filtered[k] = v
            }
        }
        return (filtered, events)
    }

    // MARK: - Algorithm

    /// The whole algorithm operates over inout state and a shared `buffer`/`body`
    /// pair; we keep it `static` so each `mutating`/`consuming` entry point can
    /// hand off the inout pieces without taking `self` apart for a non-Copyable
    /// dance.
    private static func process(
        buffer: NSMutableString,
        posRef: inout Int,
        output: inout ParsedMessage,
        current: inout String?,
        captures: inout [String: String],
        body: NSMutableString,
        opened: inout Bool,
        template: ResponseTemplate,
        transform: ResponseTransform?,
        events: inout [ResponseEvent],
        eos: Bool
    ) throws {
        while true {
            let watch = watchlist(template: template, current: current)
            var best = bestMatch(watch: watch, buffer: buffer, pos: posRef)
            // Mid-stream, a match that reaches the end of the buffer is
            // ambiguous: zero-width alternations and pattern delimiters may
            // still extend with more input. Non-zero-width literal matches
            // can never extend, so we always commit those.
            if let b = best, !eos, shouldDefer(field: b.field, match: b.match, bufLen: buffer.length, isOpen: b.kind == .open) {
                best = nil
            }

            if let b = best {
                let mStart = b.match.range.location
                let mEnd = b.match.range.location + b.match.range.length
                if mStart > posRef {
                    let between = buffer.substring(with: NSRange(location: posRef, length: mStart - posRef))
                    accumulate(events: &events, text: between, current: current, opened: &opened, body: body, template: template)
                }
                posRef = mEnd
                switch b.kind {
                case .open:
                    try closeCurrent(events: &events, output: &output, current: &current, captures: &captures, body: body, opened: &opened, template: template, transform: transform)
                    openExplicit(events: &events, field: b.field, match: b.match, buffer: buffer, current: &current, captures: &captures, body: body, opened: &opened)
                case .close:
                    let hadContent = opened
                    try closeCurrent(events: &events, output: &output, current: &current, captures: &captures, body: body, opened: &opened, template: template, transform: transform)
                    // A zero-width close on an already-empty region would just
                    // re-fire forever — bail out so we make progress.
                    if !hadContent && mStart == mEnd { return }
                }
                continue
            }

            if eos {
                if posRef < buffer.length {
                    let rest = buffer.substring(with: NSRange(location: posRef, length: buffer.length - posRef))
                    accumulate(events: &events, text: rest, current: current, opened: &opened, body: body, template: template)
                    posRef = buffer.length
                }
                try closeCurrent(events: &events, output: &output, current: &current, captures: &captures, body: body, opened: &opened, template: template, transform: transform)
                return
            }
            if watch.isEmpty {
                // Nothing to wait for — flush everything buffered into the current region.
                if posRef < buffer.length {
                    let rest = buffer.substring(with: NSRange(location: posRef, length: buffer.length - posRef))
                    accumulate(events: &events, text: rest, current: current, opened: &opened, body: body, template: template)
                    posRef = buffer.length
                }
                return
            }
            let hold = maxHold(watch: watch, buffer: buffer, pos: posRef)
            let safeEnd = buffer.length - hold
            if safeEnd > posRef {
                let safe = buffer.substring(with: NSRange(location: posRef, length: safeEnd - posRef))
                accumulate(events: &events, text: safe, current: current, opened: &opened, body: body, template: template)
                posRef = safeEnd
            }
            return
        }
    }

    private enum WatchKind { case open, close }
    private struct Watch { let kind: WatchKind; let field: ResponseField }
    private struct Best { let kind: WatchKind; let field: ResponseField; let match: NSTextCheckingResult }

    private static func watchlist(template: ResponseTemplate, current: String?) -> [Watch] {
        if let c = current, c != template.implicit {
            guard let f = template.fields[c], f.close != nil else { return [] }
            return [Watch(kind: .close, field: f)]
        }
        var out: [Watch] = []
        for f in template.fields.values where f.open != nil {
            out.append(Watch(kind: .open, field: f))
        }
        if let implicit = template.implicit, let f = template.fields[implicit], f.close != nil {
            out.append(Watch(kind: .close, field: f))
        }
        return out
    }

    private static func bestMatch(watch: [Watch], buffer: NSMutableString, pos: Int) -> Best? {
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

    private static func maxHold(watch: [Watch], buffer: NSMutableString, pos: Int) -> Int {
        var hold = 0
        for w in watch {
            let anchor: ResponseAnchor? = (w.kind == .open) ? w.field.open : w.field.close
            guard let anchor else { continue }
            hold = max(hold, patternHold(buffer: buffer, pos: pos, literals: anchor.literals))
        }
        return hold
    }

    private static func patternHold(buffer: NSMutableString, pos: Int, literals: [String]?) -> Int {
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

    private static func accumulate(
        events: inout [ResponseEvent],
        text: String,
        current: String?,
        opened: inout Bool,
        body: NSMutableString,
        template: ResponseTemplate
    ) {
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

    private static func openExplicit(
        events: inout [ResponseEvent],
        field: ResponseField,
        match: NSTextCheckingResult,
        buffer: NSMutableString,
        current: inout String?,
        captures: inout [String: String],
        body: NSMutableString,
        opened: inout Bool
    ) {
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

    private static func closeCurrent(
        events: inout [ResponseEvent],
        output: inout ParsedMessage,
        current: inout String?,
        captures: inout [String: String],
        body: NSMutableString,
        opened: inout Bool,
        template: ResponseTemplate,
        transform: ResponseTransform?
    ) throws {
        guard let c = current, opened, let fld = template.fields[c] else {
            resetToImplicit(current: &current, captures: &captures, body: body, opened: &opened, implicit: template.implicit)
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
        resetToImplicit(current: &current, captures: &captures, body: body, opened: &opened, implicit: template.implicit)
    }

    private static func resetToImplicit(
        current: inout String?,
        captures: inout [String: String],
        body: NSMutableString,
        opened: inout Bool,
        implicit: String?
    ) {
        current = implicit
        captures = [:]
        body.setString("")
        opened = false
    }

    private static func isEmpty(_ value: ParsedValue) -> Bool {
        switch value {
        case .null: return true
        case .string(let s): return s.isEmpty
        case .array(let arr): return arr.isEmpty
        case .object(let dict): return dict.isEmpty
        default: return false
        }
    }
}
