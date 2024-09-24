// WordPieceTokenizer.swift

import Foundation
import Hub

class WordPieceTokenizer: PreTrainedTokenizerModel {
    struct Token {
        var token: String
        var score: Float
    }
    
    let vocab: [Token]
    let unknownPiece: Token
    let unknownTokenId: Int?
    var unknownTokenScore: Float { unknownPiece.score }
    public var unknownToken: String? { unknownPiece.token }
    
    let minScore: Float
    let tokensToIds: [NSString: Int]
    
    let bosToken: String? = " "
    let bosTokenId: Int?
    let eosToken: String?
    let eosTokenId: Int?
    
    let fuseUnknownTokens: Bool = true
    private let maxInputCharsPerWord: Int = 100
    private let prefix: String = "##"
    
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws {
        guard let configVocab = tokenizerData.model?.vocab?.value as? [[Any]] else {
            throw TokenizerError.missingVocab
        }
        
        vocab = try configVocab.map { piece in
            guard let token = piece.first as? String,
                  let scoreValue = piece.last else {
                throw TokenizerError.malformedVocab
            }

            let score: Float
            if let floatScore = scoreValue as? Float {
                score = floatScore
            } else if let numberScore = scoreValue as? NSNumber {
                score = numberScore.floatValue
            } else {
                throw TokenizerError.malformedVocab
            }
            
            return Token(token: token, score: score)
        }
        
        minScore = vocab.reduce(999) { partial, token in
            min(partial, token.score)
        }
        
        guard let unknownTokenId = tokenizerData.model?.unkId?.intValue else { throw TokenizerError.malformedVocab }
        self.unknownTokenId = unknownTokenId
        self.unknownPiece = Token(token: vocab[unknownTokenId].token, score: minScore - 10)
        
        tokensToIds = Dictionary(uniqueKeysWithValues: vocab.map { $0.token as NSString }.enumerated().map { ($1, $0) })
        bosTokenId = tokensToIds[bosToken! as NSString]      // May be nil
        
        eosToken = tokenizerConfig.eosToken?.stringValue
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]
    }
    
    func convertTokenToId(_ token: String) -> Int? {
        return tokensToIds[token as NSString] ?? self.unknownTokenId
    }
    
    func convertIdToToken(_ id: Int) -> String? {
        return vocab[id].token
    }
    
    func tokenize(text: String) -> [String] {
        var outputTokens: [String] = []
        let words = text.split(separator: " ")
        
        for word in words {
            var chars = Array(word)
            if chars.count > maxInputCharsPerWord {
                outputTokens.append(unknownPiece.token)
                continue
            }
            
            var isBad = false
            var start = 0
            var subTokens: [String] = []
            
            while start < chars.count {
                var end = chars.count
                var curSubstr: String? = nil
                
                while start < end {
                    var substr = String(chars[start..<end])
                    if start > 0 {
                        substr = prefix + substr
                    }
                    
                    if let _ = tokensToIds[substr as NSString] {
                        curSubstr = substr
                        break
                    }
                    end -= 1
                }
                
                if curSubstr == nil {
                    isBad = true
                    break
                }
                
                subTokens.append(curSubstr!)
                start = end
            }
            
            if isBad {
                outputTokens.append(unknownPiece.token)
            } else {
                outputTokens.append(contentsOf: subTokens)
            }
        }
        
        return outputTokens
    }
}
