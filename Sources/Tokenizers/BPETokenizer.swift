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


class BPETokenizer: Tokenizer {
    let bpeRanks: Dictionary<BytePair, Int>
    private let tokensToIds: [String: Int]
    private let idsToTokens: [Int: String]
    private let addedTokens: Set<String>
    private let specialTokens: [String: Int]
    
    public let unknownToken: String
    public let unknownTokenId: Int
    
    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: Decoder?
    
    private let cleanUpTokenizationSpaces: Bool

    required init(tokenizerConfig: Config, tokenizerData: Config) throws {
        guard let vocab = tokenizerData.model?.vocab?.dictionary as? [String: Int] else {
            throw TokenizerError.missingVocab
        }
        let merges = tokenizerData.model?.merges?.value as? [String]
        
        var addedTokens: [String : Int] = [:]
        var specialTokens: [String : Int] = [:]
        for addedToken in tokenizerData.addedTokens?.arrayValue ?? [] {
            guard let id = addedToken.id?.intValue else { continue /* malformed: token with no id */ }
            guard let content = addedToken.content?.stringValue else { continue /* malformed: token with no content */ }
            addedTokens[content] = id
            
            if addedToken.special?.boolValue ?? false {
                specialTokens[content] = id
            }
        }
        // TODO: specialTokens are stored but never used
        self.specialTokens = specialTokens
        self.addedTokens = Set(addedTokens.keys)

        self.preTokenizer = PreTokenizerFactory.fromConfig(config: tokenizerData.preTokenizer)
        self.normalizer = NormalizerFactory.fromConfig(config: tokenizerData.normalizer)
        self.postProcessor = PostProcessorFactory.fromConfig(config: tokenizerData.postProcessor)
        self.decoder = DecoderFactory.fromConfig(config: tokenizerData.decoder)
        self.cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces?.boolValue ?? true

        guard let merges = merges else { fatalError("BPETokenizer requires merges") }
        var bpeRanks: Dictionary<BytePair, Int> = [:]
        for (i, item) in merges.enumerated() {
            let tuple = item.split(separator: " ").map { String($0) }
            let bp = BytePair(tuple: tuple)
            bpeRanks[bp] = i
        }
        self.bpeRanks = bpeRanks
        
        self.tokensToIds = vocab.merging(addedTokens) { $1 }
        self.idsToTokens = Utils.invert(self.tokensToIds)
        
        // Populate unknown token
        guard let unknownToken = tokenizerConfig.unkToken?.content?.stringValue else {
            throw TokenizerError.missingItem("Missing unk_token in tokenizer configuration")
        }
        guard let unknownTokenId = self.tokensToIds[unknownToken] else {
            throw TokenizerError.missingItem("No mapping in vocab for unk_token '\(unknownToken)'")
        }
        self.unknownToken = unknownToken
        self.unknownTokenId = unknownTokenId
    }
    
    func convertTokenToId(_ token: String) -> Int {
        return tokensToIds[token] ?? self.unknownTokenId
    }
    
    func convertIdToToken(_ id: Int) -> String {
        return idsToTokens[id] ?? self.unknownToken
    }

    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { (token) -> String in
            return Array(token.utf8).map { byteEncoder[$0]! }.joined()
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
    
    func preTokenize(_ text: String) -> [String] {
        guard let preTokenizer = preTokenizer else { return [text] }
        return preTokenizer(text: text)
    }
    
    func normalize(_ text: String) -> String {
        guard let normalizer = normalizer else { return text }
        return normalizer(text: text)
    }
    
    func postProcess(_ tokens: [String]) -> [String] {
        guard let postProcessor = postProcessor else { return tokens }
        return postProcessor(tokens: tokens)
    }
    
    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }
    
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let sectionTokens = preTokenize(normalize(text))
        let bpeTokens = sectionTokens.flatMap { token in
            self.bpe(token: token).split(separator: " ").map { String($0) }
        }
        for token in bpeTokens {
            if let _ = tokensToIds[token] {
                tokens.append(token)
            } else {
                // TODO: if config.byte_fallback is False, append the unknown token instead
                tokens.append(contentsOf: self.hexaEncode(text: token))
            }
        }        
        return tokens
    }
    
    /// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }

        return text.replacingOccurrences(of: " .", with: ".")
            .replacingOccurrences(of: " ?", with: "?")
            .replacingOccurrences(of: " !", with: "!")
            .replacingOccurrences(of: " ,", with: ",")
            .replacingOccurrences(of: " ' ", with: "'")
            .replacingOccurrences(of: " n't", with: "n't")
            .replacingOccurrences(of: " 'm", with: "'m")
            .replacingOccurrences(of: " 's", with: "'s")
            .replacingOccurrences(of: " 've", with: "'ve")
            .replacingOccurrences(of: " 're", with: "'re")
    }

    /// Main entry point
    func encode(text: String) -> [Int] {
        return postProcess(tokenize(text: text)).map { tokensToIds[$0]! }
    }
    
    /// Decode
    func decode(tokens: [Int]) -> String {
        // IDs to tokens
        let tokenStrings = tokens.map { idsToTokens[$0]! }
        let decoded = decodeTokens(tokenStrings)
        // At this point we should have a single String
        return cleanUp(text: decoded.joined(separator: ""))
    }
}

class GPT2Tokenizer    : BPETokenizer {}
class FalconTokenizer  : BPETokenizer {}
class LlamaTokenizer   : BPETokenizer {}
class CodeGenTokenizer : BPETokenizer {}
class WhisperTokenizer : BPETokenizer {}
