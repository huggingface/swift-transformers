//
//  Generation.swift
//  
//
//  Created by Pedro Cuenca on 7/5/23.
//

import Tokenizers
import CoreML
import TensorUtils

public enum GenerationMode {
    case contrastiveSearch
    case greedy
    case sample
    case beam
    case groupBeam
    case unsupported
}

public typealias InputTokens = [Int]
public typealias GenerationOutput = [Int]

/// A callable (a model, usually), that predicts the next token after a given sequence
public typealias NextTokenModel = (InputTokens, GenerationConfig) -> any MLShapedArrayProtocol

public typealias PredictionTokensCallback = (GenerationOutput) -> Void
public typealias PredictionStringCallback = (String) -> Void

// TODO: callbacks (for streaming)
public protocol Generation {
    func greedySearch(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel, callback: PredictionTokensCallback?) async -> GenerationOutput
    
    func generate(config: GenerationConfig, prompt: String, model: NextTokenModel, tokenizer: Tokenizer, callback: PredictionStringCallback?) async -> String
}

public extension Generation {
    func greedySearch(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel, callback: PredictionTokensCallback? = nil) async -> GenerationOutput {
        // Iterate until we find the eos token or reach the max length
        // TODO: additional stopping criteria
        var outputTokens = tokens
        while outputTokens.count < config.maxLength {
            let logits = model(outputTokens, config)
            let (nextToken, _) = Math.argmax(logits)
            if nextToken == config.eosTokenId { break }
            outputTokens.append(nextToken)
            callback?(outputTokens)
        }
        return outputTokens
    }
    
    /// https://github.com/huggingface/transformers/blob/42017d82baa083da2bee3055fdac80c81ee97b8a/src/transformers/generation/utils.py#L1552
    func sample(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel, callback: PredictionTokensCallback? = nil) async -> GenerationOutput {
        // Iterate until we find the eos token or reach the max length
        // TODO: additional stopping criteria
        var outputTokens = tokens
        while outputTokens.count < config.maxLength {
            let outputs = model(outputTokens, config)
            
            /// `floats` can be much faster than `scalars` for a vector with stride 1, as it uses `memcpy` in that case
            var logits = (outputs as? MLShapedArraySlice<Float>)?.floats ?? outputs.scalars as! [Float]
                                    
            let nextToken: Int
            if config.temperature > 0 && config.temperature != 1 {
                logits = logits.map { $0 / Float(config.temperature) }
            }
            if config.topK > 0 {
                let topK = Math.topK(arr: logits, k: config.topK)
                nextToken = Math.sample(indexes: topK.indexes, probs: topK.probs)
            } else if config.topP < 1.0 {
                let topP = Math.topP(arr: logits, p: Float(config.topP))
                nextToken = Math.sample(indexes: topP.indexes, probs: topP.probs)
            } else {
                fatalError("not implemented yet")
            }
            
            if nextToken == config.eosTokenId { break }
            outputTokens.append(nextToken)
            callback?(outputTokens)
        }
        return outputTokens
    }

    func generate(config: GenerationConfig, prompt: String, model: NextTokenModel, tokenizer: Tokenizer, callback: PredictionStringCallback? = nil) async -> String {
        let tokens = tokenizer.encode(text: prompt)
        var generationConfig = config
        generationConfig.maxLength = config.maxNewTokens + tokens.count

        let output: GenerationOutput
        switch generationConfig.generationMode {
        case .greedy:
            output = await greedySearch(config: generationConfig, tokens: tokens, model: model) { tokens in
                callback?(tokenizer.decode(tokens: tokens))
            }
        case .sample:
            output = await sample(config: generationConfig, tokens: tokens, model: model) { tokens in
                callback?(tokenizer.decode(tokens: tokens))
            }
        default:
            fatalError("Generation mode \(generationConfig.generationMode) not implemented yet")
        }
        
        return tokenizer.decode(tokens: output)
    }
}
