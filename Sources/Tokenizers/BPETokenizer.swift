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

    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { token -> String in
            return Array(token.utf8).compactMap { byteEncoder[$0] }.joined()
        }
    }

    func hexaEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.flatMap { token -> [String] in
            return Array(token.utf8).map { String(format: "<0x%02X>", $0) }
        }
    }

    /// Applies the BPE merge sequence to `token` and returns the merged pieces
    /// joined by a single space.
    ///
    /// Equivalent to the canonical greedy "lowest-rank merge first" BPE algorithm
    /// (e.g. `tiktoken`, `huggingface/tokenizers`). The previous implementation
    /// rebuilt the entire pair set and scanned the full word on every iteration,
    /// which is O(N² · M). This version maintains the symbols as an in-place
    /// linked list and tracks candidate merges in a min-heap keyed by rank, so
    /// the work per merge step is O(log N) heap ops + O(1) list surgery.
    func bpe(token: String) -> String {
        if token.count <= 1 {
            return token
        }

        // Initial symbols: one entry per Character of `token`. We keep these as
        // a doubly linked list embedded in parallel arrays of indices.
        var symbols = Array(token).map { String($0) }
        let initialCount = symbols.count
        var prevIndex = Array(repeating: -1, count: initialCount)
        var nextIndex = Array(repeating: -1, count: initialCount)
        var alive = Array(repeating: true, count: initialCount)
        for i in 0..<initialCount {
            prevIndex[i] = i - 1
            nextIndex[i] = (i == initialCount - 1) ? -1 : i + 1
        }

        // Min-heap of `(rank, leftSymbolIndex)`. We never remove entries on
        // merge; instead, when an entry is popped we re-check that the pair at
        // `(left, next[left])` is still the same and still has the recorded
        // rank — stale entries are simply skipped (lazy deletion).
        var heap: [(rank: Int, left: Int)] = []
        heap.reserveCapacity(initialCount)

        func heapPush(_ entry: (rank: Int, left: Int)) {
            heap.append(entry)
            var i = heap.count - 1
            while i > 0 {
                let parent = (i - 1) / 2
                let p = heap[parent]
                let c = heap[i]
                if p.rank < c.rank || (p.rank == c.rank && p.left <= c.left) { break }
                heap[parent] = c
                heap[i] = p
                i = parent
            }
        }

        func heapPop() -> (rank: Int, left: Int)? {
            guard !heap.isEmpty else { return nil }
            let top = heap[0]
            let last = heap.removeLast()
            if heap.isEmpty { return top }
            heap[0] = last
            var i = 0
            let count = heap.count
            while true {
                let l = 2 * i + 1
                let r = 2 * i + 2
                var smallest = i
                if l < count {
                    let cur = heap[smallest]
                    let cand = heap[l]
                    if cand.rank < cur.rank || (cand.rank == cur.rank && cand.left < cur.left) {
                        smallest = l
                    }
                }
                if r < count {
                    let cur = heap[smallest]
                    let cand = heap[r]
                    if cand.rank < cur.rank || (cand.rank == cur.rank && cand.left < cur.left) {
                        smallest = r
                    }
                }
                if smallest == i { break }
                heap.swapAt(i, smallest)
                i = smallest
            }
            return top
        }

        // Enqueue the candidate merge at position `left -> next[left]`, if any.
        @inline(__always) func enqueue(left: Int) {
            let right = nextIndex[left]
            guard right != -1, alive[left], alive[right] else { return }
            if let rank = bpeRanks[BytePair(symbols[left], symbols[right])] {
                heapPush((rank, left))
            }
        }

        for i in 0..<(initialCount - 1) {
            enqueue(left: i)
        }

        while let top = heapPop() {
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
        return pieces.joined(separator: " ")
    }

    /// Tokenizes input text using the BPE algorithm.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of BPE token strings
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let bpeTokens = bpe(token: text).split(separator: " ").map { String($0) }
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
