//
//  GenerationConfig.swift
//  
//
//  Created by Pedro Cuenca on 7/5/23.
//

import Foundation

/// Essentials taken from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
public struct GenerationConfig {
    var maxLength = 20
    var maxNewTokens: Int
    var doSample = false
    var numBeams = 1
    var numBeamGroups = 1
    var penaltyAlpha: Double? = nil
    var temperature = 1.0
    var topK = 50
    var topP = 1.0
    var repetitionPenalty = 1.0
    
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
    var generationMode: GenerationMode {
        // Exclude this case from the pattern matching below
        if topK > 1 && !doSample && penaltyAlpha != nil && penaltyAlpha! > 0 {
            return .contrastiveSearch
        }
        
        switch (numBeams, numBeamGroups, doSample) {
        case (1, 1, false)   : return .greedy
        case (1, 1, true)    : return .sample
        case (2..., 1, false): return .beam
        case (2..., 2..., _) : return .groupBeam
        default              : return .unsupported
        }
    }
}
