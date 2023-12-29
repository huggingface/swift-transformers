//
//  BertTokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import Hub

class BertTokenizer {
    private let basicTokenizer = BasicTokenizer()
    private let wordpieceTokenizer: WordpieceTokenizer
    private let maxLen = 512
    
    private let vocab: [String: Int]
    private let ids_to_tokens: [Int: String]
    
    init(vocab: [String: Int], merges: [String]?) {
        self.vocab = vocab
        self.ids_to_tokens = Utils.invert(vocab)
        self.wordpieceTokenizer = WordpieceTokenizer(vocab: self.vocab)
    }
    
    required convenience init(tokenizerConfig: Config, tokenizerData: Config) throws {
        guard let vocab = tokenizerData.model?.vocab?.dictionary as? [String: Int] else {
            throw TokenizerError.missingVocab
        }
        let merges = tokenizerData.model?.merges?.value as? [String]
        self.init(vocab: vocab, merges: merges)
    }
    
    
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        for token in basicTokenizer.tokenize(text: text) {
            for subToken in wordpieceTokenizer.tokenize(word: token) {
                tokens.append(subToken)
            }
        }
        return tokens
    }
    
    private func convertTokensToIds(tokens: [String]) throws -> [Int] {
        if tokens.count > maxLen {
            throw TokenizerError.tooLong(
                """
                Token indices sequence length is longer than the specified maximum
                sequence length for this BERT model (\(tokens.count) > \(maxLen). Running this
                sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
                """
            )
        }
        return tokens.map { vocab[$0]! }
    }
    
    /// Main entry point
    func tokenizeToIds(text: String) -> [Int] {
        return try! convertTokensToIds(tokens: tokenize(text: text))
    }
    
    func tokenToId(token: String) -> Int {
        return vocab[token]!
    }
    
    /// Un-tokenization: get tokens from tokenIds
    func unTokenize(tokens: [Int]) -> [String] {
        return tokens.map { ids_to_tokens[$0]! }
    }
    
    /// Un-tokenization:
    func convertWordpieceToBasicTokenList(_ wordpieceTokenList: [String]) -> String {
        var tokenList: [String] = []
        var individualToken: String = ""
        
        for token in wordpieceTokenList {
            if token.starts(with: "##") {
                individualToken += String(token.suffix(token.count - 2))
            } else {
                if individualToken.count > 0 {
                    tokenList.append(individualToken)
                }
                
                individualToken = token
            }
        }
        
        tokenList.append(individualToken)
        
        return tokenList.joined(separator: " ")
    }
}


extension BertTokenizer: Tokenizer {
    var unknownToken: String? { wordpieceTokenizer.unkToken }
    var unknownTokenId: Int? { vocab[unknownToken!] }
    
    func encode(text: String) -> [Int] { tokenizeToIds(text: text) }
    
    func decode(tokens: [Int]) -> String {
        let tokens = unTokenize(tokens: tokens)
        return convertWordpieceToBasicTokenList(tokens)
    }
    
    func convertTokenToId(_ token: String) -> Int? {
        return vocab[token] ?? unknownTokenId
    }
    
    func convertIdToToken(_ id: Int) -> String? {
        return ids_to_tokens[id]
    }
}


class BasicTokenizer {
    let neverSplit = [
        "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"
    ]
    
    func tokenize(text: String) -> [String] {
        let splitTokens = text.folding(options: .diacriticInsensitive, locale: nil)
            .components(separatedBy: NSCharacterSet.whitespaces)
        let tokens = splitTokens.flatMap({ (token: String) -> [String] in
            if neverSplit.contains(token) {
                return [token]
            }
            var toks: [String] = []
            var currentTok = ""
            for c in token.lowercased() {
                if c.isLetter || c.isNumber || c == "°" {
                    currentTok += String(c)
                } else if currentTok.count > 0 {
                    toks.append(currentTok)
                    toks.append(String(c))
                    currentTok = ""
                } else {
                    toks.append(String(c))
                }
            }
            if currentTok.count > 0 {
                toks.append(currentTok)
            }
            return toks
        })
        return tokens
    }
}


class WordpieceTokenizer {
    let unkToken = "[UNK]"
    private let maxInputCharsPerWord = 100
    private let vocab: [String: Int]
    
    init(vocab: [String: Int]) {
        self.vocab = vocab
    }
    
    /// `word`: A single token.
    /// Warning: this differs from the `pytorch-transformers` implementation.
    /// This should have already been passed through `BasicTokenizer`.
    func tokenize(word: String) -> [String] {
        if word.count > maxInputCharsPerWord {
            return [unkToken]
        }
        var outputTokens: [String] = []
        var isBad = false
        var start = 0
        var subTokens: [String] = []
        while start < word.count {
            var end = word.count
            var cur_substr: String? = nil
            while start < end {
                var substr = Utils.substr(word, start..<end)!
                if start > 0 {
                    substr = "##\(substr)"
                }
                if vocab[substr] != nil {
                    cur_substr = substr
                    break
                }
                end -= 1
            }
            if cur_substr == nil {
                isBad = true
                break
            }
            subTokens.append(cur_substr!)
            start = end
        }
        if isBad {
            outputTokens.append(unkToken)
        } else {
            outputTokens.append(contentsOf: subTokens)
        }
        return outputTokens
    }
}
