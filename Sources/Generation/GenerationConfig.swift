//
//  GenerationConfig.swift
//
//
//  Created by Pedro Cuenca on 7/5/23.
//

import Foundation

/// Configuration parameters for text generation algorithms.
///
/// Contains all the parameters needed to control various aspects of text generation,
/// including sampling parameters, beam search settings, and special token IDs.
///
/// - Note: Based on https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
public struct GenerationConfig {
    /// Maximum total length of the generated sequence (input + output tokens).
    public var maxLength = 20

    /// Maximum number of new tokens to generate.
    public var maxNewTokens: Int

    /// Whether to use sampling instead of deterministic decoding.
    public var doSample = false

    /// Number of beams for beam search (1 for greedy decoding).
    public var numBeams = 1

    /// Number of beam groups for group beam search.
    public var numBeamGroups = 1

    /// Penalty parameter for contrastive search.
    public var penaltyAlpha: Double?

    /// Temperature for sampling (higher values increase randomness).
    public var temperature = 1.0

    /// Number of top tokens to consider for top-k sampling.
    public var topK = 50

    /// Cumulative probability threshold for top-p sampling.
    public var topP = 1.0

    /// Penalty for token repetition (1.0 means no penalty).
    public var repetitionPenalty = 1.0

    /// Token ID used for padding sequences.
    public var padTokenId: Int?

    /// Token ID for beginning of sequence.
    public var bosTokenId: Int?

    /// Token ID for end of sequence.
    public var eosTokenId: Int?

    /// Creates a new generation configuration.
    ///
    /// - Parameters:
    ///   - maxLength: Maximum total sequence length
    ///   - maxNewTokens: Maximum new tokens to generate
    ///   - doSample: Enable sampling instead of greedy decoding
    ///   - numBeams: Number of beams for beam search
    ///   - numBeamGroups: Number of beam groups for group beam search
    ///   - penaltyAlpha: Penalty parameter for contrastive search
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling parameter
    ///   - topP: Top-p sampling parameter
    ///   - repetitionPenalty: Repetition penalty factor
    public init(maxLength: Int = 20, maxNewTokens: Int, doSample: Bool = false, numBeams: Int = 1, numBeamGroups: Int = 1, penaltyAlpha: Double? = nil, temperature: Double = 1.0, topK: Int = 50, topP: Double = 1.0, repetitionPenalty: Double = 1.0) {
        self.maxLength = maxLength
        self.maxNewTokens = maxNewTokens
        self.doSample = doSample
        self.numBeams = numBeams
        self.numBeamGroups = numBeamGroups
        self.penaltyAlpha = penaltyAlpha
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
    }
}

public extension GenerationConfig {
    /// Determines the appropriate generation mode based on configuration parameters.
    ///
    /// Analyzes the combination of sampling settings, beam parameters, and penalty values
    /// to automatically select the most appropriate generation algorithm.
    ///
    /// - Returns: The determined generation mode
    var generationMode: GenerationMode {
        // Exclude this case from the pattern matching below
        if topK > 1, !doSample, penaltyAlpha != nil, penaltyAlpha! > 0 {
            return .contrastiveSearch
        }

        switch (numBeams, numBeamGroups, doSample) {
        case (1, 1, false): return .greedy
        case (1, 1, true): return .sample
        case (2..., 1, false): return .beam
        case (2..., 2..., _): return .groupBeam
        default: return .unsupported
        }
    }
}

extension GenerationConfig: Decodable {}
