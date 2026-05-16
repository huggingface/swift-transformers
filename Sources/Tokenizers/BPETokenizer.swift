//
//  BPETokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import Hub

/// A pair of byte/token strings used in Byte-Pair Encoding (BPE) merge operations.
struct BytePair: Hashable, Sendable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }

    init(tuple: [String]) {
        a = tuple[0]
        b = tuple[1]
    }

    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        lhs.a == rhs.a && lhs.b == rhs.b
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(a)
        hasher.combine(b)
    }
}

/// Minimal binary min-heap. `push` and `pop` are O(log n).
/// Used by `BPETokenizer.bpe(token:)` for the priority-queue BPE merge loop,
/// as the reference implementation does, see https://github.com/huggingface/tokenizers/blob/b58227c7f1ccf8b73ee2268354336da56d91e492/tokenizers/src/models/bpe/word.rs
private struct MinHeap<Element: Comparable> {
    private var storage: [Element] = []

    var isEmpty: Bool { storage.isEmpty }

    mutating func reserveCapacity(_ n: Int) {
        storage.reserveCapacity(n)
    }

    mutating func push(_ element: Element) {
        storage.append(element)
        var i = storage.count - 1
        while i > 0 {
            let parent = (i - 1) / 2
            if storage[parent] <= storage[i] { break }
            storage.swapAt(parent, i)
            i = parent
        }
    }

    mutating func pop() -> Element? {
        guard !storage.isEmpty else { return nil }
        let top = storage[0]
        let last = storage.removeLast()
        if storage.isEmpty { return top }
        storage[0] = last
        var i = 0
        let n = storage.count
        while true {
            let l = 2 * i + 1
            let r = 2 * i + 2
            var smallest = i
            if l < n, storage[l] < storage[smallest] { smallest = l }
            if r < n, storage[r] < storage[smallest] { smallest = r }
            if smallest == i { break }
            storage.swapAt(i, smallest)
            i = smallest
        }
        return top
    }
}

/// Heap entry for the BPE merge priority queue. Lower `rank` wins; ties break
/// on leftmost `left` index, matching `huggingface/tokenizers` semantics.
private struct BPEMergeCandidate: Comparable {
    let rank: Int
    let left: Int

    static func < (lhs: Self, rhs: Self) -> Bool {
        if lhs.rank != rhs.rank { return lhs.rank < rhs.rank }
        return lhs.left < rhs.left
    }
}

/// A Byte-Pair Encoding (BPE) tokenizer implementation.
///
/// BPE tokenizers learn to merge the most frequently occurring pairs of characters
/// or character sequences. This implementation supports various BPE-based models
/// including GPT-2, RoBERTa, and other transformer models.
class BPETokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    let bpeRanks: [BytePair: Int]
    private let tokensToIds: [NSString: Int]
    private let idsToTokens: [Int: NSString]

    /// The total number of tokens in the vocabulary.
    var vocabCount: Int { tokensToIds.count }

    /// The beginning-of-sequence token string, if defined.
    let bosToken: String?

    /// The numeric ID of the beginning-of-sequence token, if defined.
    let bosTokenId: Int?

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// The unknown token string used for out-of-vocabulary words.
    let unknownToken: String?

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// Whether consecutive unknown tokens should be fused together.
    let fuseUnknownTokens: Bool

    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config else { return nil }

        if let merges = config.array() {
            return merges.reduce(into: [[String]]()) { result, element in
                if let val: [String] = element.get() { // New format (pushed with tokenizers >= 0.20.0): each merge is a list of 2 items
                    result.append(val)
                }
                if let val: String = element.get() { // legacy
                    result.append(val.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) })
                }
            }
        }

        return nil
    }

    /// Initializes a BPE tokenizer from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and merges
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Throws: `TokenizerError` if required configuration is missing
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let merges = Self.mergesFromConfig(tokenizerData.model.merges) else { fatalError("BPETokenizer requires merges") }
        guard let vocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }
        var bpeRanks: [BytePair: Int] = [:]
        for (i, merge) in merges.enumerated() {
            let bp = BytePair(tuple: merge)
            bpeRanks[bp] = i
        }
        self.bpeRanks = bpeRanks

        let addedTokens = addedTokens.reduce(into: [BinaryDistinctString: Config]()) { result, element in
            result[BinaryDistinctString(element.key)] = .init(element.value)
        }
        tokensToIds = vocab.merging(addedTokens) { $1 }.reduce(into: [NSString: Int]()) { result, element in
            result[element.key.nsString] = element.value.integer()
        }

        idsToTokens = tokensToIds.reduce(into: [Int: NSString]()) { result, element in
            result[element.value] = element.key
        }

        // Populate tokens
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID, or the unknown token ID if not found
    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token as NSString] ?? unknownTokenId
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    func convertIdToToken(_ id: Int) -> String? {
        idsToTokens[id] as String?
    }

    /// Cached `<0x%02X>` byte fallback strings, indexed by byte value.
    private static let hexaEncoderTable: [String] = (0..<256).map { String(format: "<0x%02X>", $0) }

    func byteEncode(text: String) -> [String] {
        var result: [String] = []
        enumerateRegexTokens(in: text, with: byteLevelPreTokenizeRegex) { token in
            var encoded = ""
            encoded.reserveCapacity(token.utf8.count)
            for byte in token.utf8 {
                encoded.append(byteEncoderTable[Int(byte)])
            }
            result.append(encoded)
        }
        return result
    }

    func hexaEncode(text: String) -> [String] {
        var result: [String] = []
        enumerateRegexTokens(in: text, with: byteLevelPreTokenizeRegex) { token in
            for byte in token.utf8 {
                result.append(Self.hexaEncoderTable[Int(byte)])
            }
        }
        return result
    }

    /// Applies the BPE merge sequence to `token` and returns the resulting pieces.
    ///
    /// Equivalent to the canonical greedy "lowest-rank merge first" BPE algorithm
    /// (e.g. `tiktoken`, `huggingface/tokenizers`). Maintains the symbols as an
    /// in-place linked list and tracks candidate merges in a min-heap keyed by
    /// rank, so the work per merge step is O(log N) heap ops + O(1) list surgery.
    ///
    /// Returns an array of pieces rather than a space-joined string: a downstream
    /// `split(separator: " ")` would be unsafe when a piece begins with a Unicode
    /// non-spacing mark, because the mark forms a single grapheme cluster with the
    /// preceding space and the split silently swallows the boundary.
    func bpe(token: String) -> [String] {
        var symbols = token.unicodeScalars.map { String($0) }
        if symbols.count <= 1 {
            return symbols.isEmpty ? [] : [token]
        }

        let initialCount = symbols.count
        var prevIndex = Array(repeating: -1, count: initialCount)
        var nextIndex = Array(repeating: -1, count: initialCount)
        var alive = Array(repeating: true, count: initialCount)
        for i in 0..<initialCount {
            prevIndex[i] = i - 1
            nextIndex[i] = (i == initialCount - 1) ? -1 : i + 1
        }

        // Min-heap of merge candidates. We never remove entries on merge;
        // instead, when an entry is popped we re-check that the pair at
        // `(left, next[left])` is still the same and still has the recorded
        // rank — stale entries are simply skipped (lazy deletion).
        var heap = MinHeap<BPEMergeCandidate>()
        heap.reserveCapacity(initialCount)

        // Enqueue the candidate merge at position `left -> next[left]`, if any.
        func enqueue(left: Int) {
            let right = nextIndex[left]
            guard right != -1, alive[left], alive[right] else { return }
            if let rank = bpeRanks[BytePair(symbols[left], symbols[right])] {
                heap.push(BPEMergeCandidate(rank: rank, left: left))
            }
        }

        for i in 0..<(initialCount - 1) {
            enqueue(left: i)
        }

        while let top = heap.pop() {
            let i = top.left
            guard alive[i] else { continue }
            let j = nextIndex[i]
            guard j != -1, alive[j] else { continue }
            // Validate the entry is not stale: the pair at (i, j) must still
            // have exactly the rank we recorded when we enqueued it.
            guard let actualRank = bpeRanks[BytePair(symbols[i], symbols[j])], actualRank == top.rank else {
                continue
            }

            // Absorb symbol j into symbol i.
            symbols[i] = symbols[i] + symbols[j]
            let k = nextIndex[j]
            nextIndex[i] = k
            if k != -1 { prevIndex[k] = i }
            alive[j] = false

            // The merge invalidates the pairs (prev, i) and (j, k); the new
            // candidates to consider are (prev, i) (now spanning the merged
            // text on the right) and (i, k) (now spanning the merged text on
            // the left). Stale entries left over for the old pairs in the
            // heap will be discarded when popped.
            if prevIndex[i] != -1 {
                enqueue(left: prevIndex[i])
            }
            enqueue(left: i)
        }

        // Walk the surviving symbols from the head (index 0 is never absorbed:
        // merges always move text from `next` into `current`, so position 0
        // remains alive throughout).
        var pieces: [String] = []
        pieces.reserveCapacity(initialCount)
        var cursor = 0
        while cursor != -1 {
            pieces.append(symbols[cursor])
            cursor = nextIndex[cursor]
        }
        return pieces
    }

    /// Tokenizes input text using the BPE algorithm.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of BPE token strings
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let bpeTokens = bpe(token: text)
        for token in bpeTokens {
            if convertTokenToId(token) != unknownTokenId {
                tokens.append(token)
            } else {
                // TODO: if config.byte_fallback is False, append the unknown token instead
                tokens.append(contentsOf: hexaEncode(text: token))
            }
        }
        return tokens
    }
}
