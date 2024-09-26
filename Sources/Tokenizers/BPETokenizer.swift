//
//  GPT2Tokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import Hub

struct BytePair: Hashable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }
    init(tuple: [String]) {
        self.a = tuple[0]
        self.b = tuple[1]
    }
    
    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        return lhs.a == rhs.a && lhs.b == rhs.b
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(a)
        hasher.combine(b)
    }
}


class BPETokenizer: PreTrainedTokenizerModel {
    let bpeRanks: Dictionary<BytePair, Int>
    private let tokensToIds: [NSString: Int]
    private let idsToTokens: [Int: NSString]

    var vocabCount: Int { tokensToIds.count }

    public let bosToken: String?
    public let bosTokenId: Int?
    public let eosToken: String?
    public let eosTokenId: Int?
    public let unknownToken: String?
    public let unknownTokenId: Int?

    public let fuseUnknownTokens: Bool

    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config = config else { return nil }

        // New format (pushed with tokenizers >= 0.20.0): each merge is a list of 2 items
        if let merges = config.value as? [[String]] { return merges }

        // Legacy: each merge is a string
        guard let merges = config.value as? [String] else { return nil }
        return merges.map { mergeString in
            mergeString.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) }
        }
    }

    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws {
        guard let merges = Self.mergesFromConfig(tokenizerData.model?.merges) else { fatalError("BPETokenizer requires merges") }
        guard let vocab = tokenizerData.model?.vocab?.dictionary as? [NSString: Int] else {
            throw TokenizerError.missingVocab
        }
        var bpeRanks: Dictionary<BytePair, Int> = [:]
        for (i, merge) in merges.enumerated() {
            let bp = BytePair(tuple: merge)
            bpeRanks[bp] = i
        }
        self.bpeRanks = bpeRanks
        
        self.tokensToIds = vocab.merging(addedTokens as [NSString : Int]) { $1 }
        self.idsToTokens = Utils.invert(self.tokensToIds)
        
        // Populate tokens
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            self.unknownTokenId = self.tokensToIds[unknownToken as NSString]
        } else {
            self.unknownToken = nil
            self.unknownTokenId = nil
        }
        
        eosToken = tokenizerConfig.eosToken?.stringValue
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = tokenizerConfig.bosToken?.stringValue
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk?.boolValue ?? false
    }

    func convertTokenToId(_ token: String) -> Int? {
        return tokensToIds[token as NSString] ?? self.unknownTokenId
    }
    
    func convertIdToToken(_ id: Int) -> String? {
        return idsToTokens[id] as String?
    }

    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { (token) -> String in
            return Array(token.utf8).compactMap { byteEncoder[$0] }.joined()
        }
    }
    
    func hexaEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.flatMap { (token) -> [String] in
            return Array(token.utf8).map { String(format: "<0x%02X>", $0) }
        }
    }
    
    private func getPairs(word: [String]) -> Set<BytePair> {
        var s = Set<BytePair>()
        for i in 0..<word.count-1 {
            let bp = BytePair(
                word[i],
                word[i+1]
            )
            s.insert(bp)
        }
        return s
    }
    
    func bpe(token: String) -> String {
        if token.count <= 1 {
            return token
        }
        
        var word = Array(token).map { String($0) }
        var pairs = Array(getPairs(word: word))
        
        while true {
            let bigrams = pairs.filter { (bp) -> Bool in bpeRanks[bp] != nil }
            if bigrams.count == 0 {
                break
            }
            let bigram = bigrams.min { (bp1, bp2) -> Bool in
                return bpeRanks[bp1]! < bpeRanks[bp2]!
            }!
            let first = bigram.a
            let second = bigram.b
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if let j = word[i..<word.count].firstIndex(of: first) {
                    newWord.append(contentsOf: word[i..<j])
                    i = j
                } else {
                    newWord.append(contentsOf: word[i..<word.count])
                    break
                }
                
                if word[i] == first && i < word.count - 1 && word[i+1] == second {
                    newWord.append(first+second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 {
                break
            } else {
                pairs = Array(getPairs(word: word))
            }
        }
        return word.joined(separator: " ")
    }

    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let bpeTokens = self.bpe(token: text).split(separator: " ").map { String($0) }
        for token in bpeTokens {
            if convertTokenToId(token) != unknownTokenId {
                tokens.append(token)
            } else {
                // TODO: if config.byte_fallback is False, append the unknown token instead
                tokens.append(contentsOf: self.hexaEncode(text: token))
            }
        }
        return tokens
    }
}
