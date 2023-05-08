//
//  Generation.swift
//  
//
//  Created by Pedro Cuenca on 7/5/23.
//

import Tokenizers

public enum GenerationMode {
    case contrastiveSearch
    case greedy
    case sample
    case beam
    case groupBeam
    case unsupported
}

// TODO: maybe concretize these to Core ML types. Maybe make generic when we need to.
public typealias InputTokens = [Int]
public typealias GenerationOutput = [Int]
public typealias NextTokenModel = (InputTokens) -> Int

// TODO: callbacks (for streaming)
public protocol Generation {
    func greedySearch(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel) -> GenerationOutput
    
    func generate(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel, tokenizer: Tokenizer) -> String
}

public extension Generation {
    func greedySearch(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel) -> GenerationOutput {
        return [0]
    }
    
    func generate(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel, tokenizer: Tokenizer) -> String {
        let output: GenerationOutput
        
        switch config.generationMode {
        case .greedy:
            output = greedySearch(config: config, tokens: tokens, model: model)
        default:
            fatalError("Generation mode \(config.generationMode) not implemented yet")
        }
        
        return tokenizer.decode(tokens: output)
    }
}
