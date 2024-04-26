//
//  UnigramTokenizer.swift
//
//
//  Created by Pedro Cuenca on 20240131.
//  Copyright © 2024 Hugging Face. All rights reserved.
//

import Hub

class UnigramTokenizer: PreTrainedTokenizerModel {
    struct SentencePieceToken {
        var token: String
        var score: Float
    }
    let vocab: [SentencePieceToken]
    
    let unknownPiece: SentencePieceToken
    var unknownTokenScore: Float { unknownPiece.score }
    
    public let unknownTokenId: Int?
    public var unknownToken: String? { unknownPiece.token }
    
    let minScore: Float
    let tokensToIds: [LiteralString: Int]
    
    let bosToken: String? = " "
    let bosTokenId: Int?
    let eosToken: String?
    let eosTokenId: Int?
    
    private let trie: Trie<Character>
    
    struct LiteralString: Hashable {
        let value: String

        static func ==(lhs: LiteralString, rhs: LiteralString) -> Bool {
            return lhs.value.compare(rhs.value, options: .literal) == .orderedSame
        }

        func hash(into hasher: inout Hasher) {
            hasher.combine(value)
        }
    }
        
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws {
        guard let configVocab = tokenizerData.model?.vocab?.value as? [[Any]] else {
            throw TokenizerError.missingVocab
        }
        
        vocab = try configVocab.map { piece in
            guard let token = piece.first as? String else { throw TokenizerError.malformedVocab }
            // Immediately mapping to `Float` values will result in exception,
            // when precision loss is detected. So let's convert to `Double` first.
            guard let scoreDouble = piece.last as? Double else { throw TokenizerError.malformedVocab }
            let score = Float(scoreDouble) // Convert Double to Float
            return SentencePieceToken(token: token, score: score)
        }
        
        minScore = vocab.reduce(999) { partial, token in
            min(partial, token.score)
        }
        
        guard let unknownTokenId = tokenizerData.model?.unkId?.intValue else { throw TokenizerError.malformedVocab }
        self.unknownTokenId = unknownTokenId
        self.unknownPiece = SentencePieceToken(token: vocab[unknownTokenId].token, score: minScore - 10)
        
        // Using `Dictionary(uniqueKeysWithValues:)` is the default approach for constructing the mapping.
        // It, however, will use the normal `compare` function of strings and will result in collisions for different 
        // UTF8 strings. Instead, we should use the `a.compare(b, options: .literal) == .orderedSame`.
        tokensToIds = Dictionary(uniqueKeysWithValues: vocab.map { $0.token }.enumerated().map { (LiteralString(value: $1), $0) })
        bosTokenId = tokensToIds[LiteralString(value: bosToken!)]      // May be nil
        
        eosToken = tokenizerConfig.eosToken?.stringValue
        eosTokenId = eosToken == nil ? nil : tokensToIds[LiteralString(value: eosToken!)]
        
        trie = Trie()
        trie.append(contentsOf: vocab.map { $0.token })
                
        // TODO: set fuse_unk to true
    }
    
    func convertTokenToId(_ token: String) -> Int? {
        return tokensToIds[LiteralString(value: token)] ?? self.unknownTokenId
    }
    
    func convertIdToToken(_ id: Int) -> String? {
        return vocab[id].token
    }
        
    func tokenize(text: String) -> [String] {
        var lattice = TokenLattice(sentence: text, bosTokenId: bosTokenId ?? 0, eosTokenId: eosTokenId ?? 0)
        
        // Populate nodes
        let sentence = lattice.sentence
        var beginPos = 0
        while beginPos < sentence.count {
            let mblen = 1
            var hasSingleNode = false
            
            let beginIndex = sentence.index(sentence.startIndex, offsetBy: beginPos)
            for token in trie.commonPrefixSearchIterator(sentence[beginIndex...]).map({ String($0) }) {
                guard let tokenId = tokensToIds[LiteralString(value: token)] else { fatalError("Token not in vocab: \(token)") }
                let tokenScore = vocab[tokenId].score
                lattice.insert(startOffset: beginPos, length: token.count, score: tokenScore, tokenId: tokenId)
                if !hasSingleNode && token.count == mblen {
                    hasSingleNode = true
                }
            }
            if !hasSingleNode {
                lattice.insert(startOffset: beginPos, length: mblen, score: unknownTokenScore, tokenId: unknownTokenId ?? 0)
            }
            beginPos += mblen
        }

        return lattice.tokens
    }
}
