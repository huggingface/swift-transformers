import Hub
import Tokenizers
import CoreML
import Accelerate


public protocol Embedding {}

public struct AutoEmbedding {} // Otherwise AutoModel

extension AutoEmbedding {
    public static func from(pretrained model: String, hubApi: HubApi = .shared) async throws -> Embedding {
        return try await BGEM3Model(repoName: model, hubApi: hubApi)
    }
}

class BERTEmbedding: Embedding { // Otherwise BERTModel
    private let wordEmbedding: BNNS.EmbeddingLayer
    private let positionEmbedding: BNNS.EmbeddingLayer
    private let tokenTypeEmbedding: BNNS.EmbeddingLayer
    private let normalization: BNNS.NormalizationLayer
    private let dropout: BNNS.DropoutLayer

    private let positionEmbeddingType = "absolute"

    init(repoName: String) { fatalError() }

    public func callAsFunction(inputIds: MLMultiArray? = nil,
                               tokenTypeIDs: MLMultiArray? = nil,
                               positionIDs: MLMultiArray? = nil,
                               inputEmbeds: MLMultiArray? = nil,
                               pastKeyValuesLength: Int = 0) -> MLMultiArray {
        fatalError()
    }
}

class BGEM3Model: Embedding {

    struct Output {
        let lastHidddenState: MLMultiArray // batchSize, sequenceLength, hiddenSize
        let hiddenStates: MLMultiArray?
        let attentions: MLMultiArray?
        
        let loss: MLMultiArray?
        let scores: MLMultiArray?
        let pReps: MLMultiArray?
        let qReps: MLMultiArray?
    }

    let withSparse      = false
    let withDense       = true
    let withColbert     = false

    let shouldNormalize = false
//    let poolingMethod               = "cls"
//    let negativesCrossDevice        = false
//    let temperature                 = 1.0
//    let enableSubBatch              = true
//    let unifiedFinetuning           = true
//    let useSelfDistill              = false
//    let colbertDim: Int?            = nil
//    let selfDistillStartStep: Int?  = nil

    private let tokenizer: Tokenizer
    private let denseLayer: BNNS.FullyConnectedLayer
    private let sparseLayer: BNNS.FullyConnectedLayer
    private let colbertLayer: BNNS.FullyConnectedLayer

    init(repoName: String, hubApi: HubApi) async throws {
        let config = LanguageModelConfigurationFromHub(modelName: repoName)
        self.tokenizer = try await AutoTokenizer.from(pretrained: repoName, hubApi: hubApi)

        let hiddenSize = try await config.modelConfig.hiddenSize?.intValue ?? 384
        let colbertDim: Int? = nil
        let denseInput = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(hiddenSize, stride: 2))
        let denseOutput = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(colbertDim ?? hiddenSize, stride: 2))
        let denseWeights = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(hiddenSize, stride: 2))
        self.denseLayer = BNNS.FullyConnectedLayer(input: denseInput, output: denseOutput, weights: denseWeights, bias: nil, activation: .identity)!
        
        let sparseInput = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(hiddenSize, stride: 2))
        let sparseOutput = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(1, stride: 2))
        let sparseWeights = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(hiddenSize, stride: 2))
        self.sparseLayer = BNNS.FullyConnectedLayer(input: sparseInput, output: sparseOutput, weights: sparseWeights, bias: nil, activation: .identity)!
        
        let colbertInput = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(hiddenSize, stride: 2))
        let colbertOutput = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(1, stride: 2))
        let colbertWeights = BNNSNDArrayDescriptor(dataType: .float16, shape: .vector(hiddenSize, stride: 2))
        self.colbertLayer = BNNS.FullyConnectedLayer(input: colbertInput, output: colbertOutput, weights: colbertWeights, bias: nil, activation: .identity)!
    }

    public func callAsFunction(_ textInput: (indices: MLMultiArray, attentionMask: MLMultiArray)) -> Output {
        fatalError()
    }

    private func forward(textInput: (indices: MLMultiArray, attentionMask: MLMultiArray)) -> [String: MLMultiArray] {
        let lastHiddenState = self(textInput).lastHidddenState

        var output = [String: MLMultiArray]()
        if withDense {
            output["dense"] = self.dense(hiddenState: lastHiddenState, mask: textInput.attentionMask)
        }
        if withSparse {
            output["sparse"] = self.sparse(hiddenState: lastHiddenState, mask: textInput.attentionMask)
        }
        if withColbert {
            output["colbert"] = self.colbert(hiddenState: lastHiddenState, mask: textInput.attentionMask)
        }

        if shouldNormalize {
            if withDense {
                // TODO: Normalize output["dense"] =
                fatalError()
            }
            if withColbert {
                // TODO: Normalize output["colbert"] =
                fatalError()
            }
        }

        return output
    }
    
    private func dense(hiddenState: MLMultiArray, mask: MLMultiArray) -> MLMultiArray {
        assert(hiddenState.shape.count == 2)
        var data = [Float]()
        data.reserveCapacity(hiddenState.count)

        for index in 0..<hiddenState.count {
            data.append(hiddenState[index].floatValue)
        }
        
        return try! MLMultiArray(data)
    }
    
    private func sparse(hiddenState: MLMultiArray, mask: MLMultiArray) -> MLMultiArray { 
        fatalError()
    }

    private func colbert(hiddenState: MLMultiArray, mask: MLMultiArray) -> MLMultiArray { 
        fatalError()
    }
}
