//
//  Generation.swift
//
//
//  Created by Pedro Cuenca on 7/5/23.
//

import CoreML
import Tokenizers

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
@available(macOS 15.0, iOS 18.0, *)
public typealias NextTokenModel = (MLTensor, GenerationConfig) async -> MLTensor

public typealias PredictionTokensCallback = (GenerationOutput) -> Void
public typealias PredictionStringCallback = (String) -> Void

@available(macOS 15.0, iOS 18.0, *)
public protocol Generation {
    func generate(
        config: GenerationConfig,
        tokens: InputTokens,
        model: NextTokenModel,
        callback: PredictionTokensCallback?
    ) async -> GenerationOutput
}

@available(macOS 15.0, iOS 18.0, *)
extension Generation {
    public func generate(
        config: GenerationConfig,
        tokens: InputTokens,
        model: NextTokenModel,
        callback: PredictionTokensCallback? = nil
    ) async -> GenerationOutput {
        let tokens = tokens.map { Int32($0) }
        var outputTokens = MLTensor(tokens).expandingShape(at: 0)
        while outputTokens.shape[1] < config.maxLength {
            let nextTokenScores = await model(outputTokens, config)
            let nextToken = switch config.generationMode {
            case .greedy:
                selectNextTokenUsingGreedyDecoding(from: nextTokenScores)
            case .sample:
                selectNextTokenUsingTopKSampling(
                    from: nextTokenScores,
                    temperature: config.temperature,
                    topK: config.topK
                )
            default:
                fatalError("Generation mode \(config.generationMode) not implemented yet")
            }

            if let nextTokenId = await tensorToGenerationOutput(nextToken).first, nextTokenId == config.eosTokenId {
                break
            }

            outputTokens = MLTensor(concatenating: [outputTokens, nextToken], alongAxis: -1)
            if let callback {
                let outputTokenIDs = await tensorToGenerationOutput(outputTokens)
                callback(outputTokenIDs)
            }
        }
        return await tensorToGenerationOutput(outputTokens)
    }

    private func tensorToGenerationOutput(_ tensor: MLTensor) async -> GenerationOutput {
        await tensor.shapedArray(of: Int32.self).scalars.map { Int($0) }
    }
}

@available(macOS 15.0, iOS 18.0, *)
public extension Generation {
    func generate(
        config: GenerationConfig,
        prompt: String,
        model: NextTokenModel,
        tokenizer: Tokenizer,
        callback: PredictionStringCallback? = nil
    ) async -> String {
        let tokens = tokenizer.encode(text: prompt)
        var generationConfig = config
        generationConfig.maxLength = config.maxNewTokens + tokens.count
        generationConfig.eosTokenId = tokenizer.eosTokenId
        generationConfig.bosTokenId = tokenizer.bosTokenId
        let output = await generate(
            config: generationConfig,
            tokens: tokens,
            model: model
        ) { tokens in
            callback?(tokenizer.decode(tokens: tokens))
        }

        return tokenizer.decode(tokens: output)
    }
}
