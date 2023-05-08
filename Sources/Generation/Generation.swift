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
    
    func generate(config: GenerationConfig, prompt: String, model: NextTokenModel, tokenizer: Tokenizer) -> String
}

public extension Generation {
    func greedySearch(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel) -> GenerationOutput {
        // Iterate until we find the eos token or reach the max length
        // TODO: additional stopping criteria
        var outputTokens = tokens
        while outputTokens.count < config.maxLength {
            let nextToken = model(outputTokens)
            if nextToken == config.eosTokenId {
                print("jurlrlrlrlr")
                break
            }
            outputTokens.append(nextToken)
        }
        return outputTokens
    }
    
    func generate(config: GenerationConfig, prompt: String, model: NextTokenModel, tokenizer: Tokenizer) -> String {
        let tokens = tokenizer.encode(text: prompt)
        var generationConfig = config
        generationConfig.maxLength = config.maxNewTokens + tokens.count

        let output: GenerationOutput
        switch generationConfig.generationMode {
        case .greedy:
            output = greedySearch(config: generationConfig, tokens: tokens, model: model)
        default:
            fatalError("Generation mode \(generationConfig.generationMode) not implemented yet")
        }
        
        return tokenizer.decode(tokens: output)
    }
}
