import Hub
import Tokenizers
import CoreML
import Accelerate


class BERTEmbedding {

    typealias Weights = [String: MLMultiArray]
    
    var shape: [NSNumber] {[
        NSNumber(value: maxPositionEmbeddings),
        NSNumber(value: hiddenSize),
    ]}

    private let weights: Weights

    private let positionEmbeddingType: String
    private let hiddenSize: Int
    private let vocabSize: Int
    private let maxPositionEmbeddings: Int
    private let typeVocabSize: Int
    private let padTokenID: Int
    private let normalizationEpsilon: Float
    private let dropoutRate: Float = 1e-1
    private let hiddenActivation: BNNS.ActivationFunction = .geluApproximation2(alpha: 1e-1, beta: 1e-1)

    private var allocations: [BNNSNDArrayDescriptor] = []

    private lazy var wordEmbedding: BNNS.EmbeddingLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Int64.self, shape: .vector(maxPositionEmbeddings))
        allocations.append(input)
        let dictData: [Float32] = weights["bert.embeddings.word_embeddings.weight"]!.toArray()
        let dict = BNNSNDArrayDescriptor.allocate(initializingFrom: dictData, shape: .matrixColumnMajor(hiddenSize, vocabSize))
        allocations.append(dict)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(output)
        
        return BNNS.EmbeddingLayer(input: input, output: output, dictionary: dict, paddingIndex: 0, maximumNorm: 0, normType: .l2, scalesGradientByFrequency: false)!
    }()
    
    private lazy var positionEmbedding: BNNS.EmbeddingLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Int64.self, shape: .vector(maxPositionEmbeddings))
        allocations.append(input)
        let dictData: [Float32] = weights["bert.embeddings.position_embeddings.weight"]!.toArray()
        let dict = BNNSNDArrayDescriptor.allocate(initializingFrom: dictData, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(dict)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(output)

        return BNNS.EmbeddingLayer(input: input, output: output, dictionary: dict, paddingIndex: -1, maximumNorm: 0, normType: .l2, scalesGradientByFrequency: true)!
    }()
    
    private lazy var tokenTypeEmbedding: BNNS.EmbeddingLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Int64.self, shape: .vector(maxPositionEmbeddings))
        allocations.append(input)
        let dictData: [Float32] = weights["bert.embeddings.token_type_embeddings.weight"]!.toArray()
        let dict = BNNSNDArrayDescriptor.allocate(initializingFrom: dictData, shape: .matrixColumnMajor(hiddenSize, typeVocabSize))
        allocations.append(dict)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(output)
        
        return BNNS.EmbeddingLayer(input: input, output: output, dictionary: dict, paddingIndex: -1, maximumNorm: 0, normType: .l2, scalesGradientByFrequency: true)!
    }()
    
    private lazy var normalization: BNNS.NormalizationLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixRowMajor(maxPositionEmbeddings, hiddenSize))
        allocations.append(input)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixRowMajor(maxPositionEmbeddings, hiddenSize))
        allocations.append(output)

        let betaWA: MLMultiArray! = weights["bert.embeddings.LayerNorm.beta"] ?? weights["bert.embeddings.LayerNorm.bias"]
        let beta = BNNSNDArrayDescriptor.allocate(initializingFrom: betaWA.toArray() as [Float32], shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(beta)

        let gammaWA: MLMultiArray! = weights["bert.embeddings.LayerNorm.gamma"] ?? weights["bert.embeddings.LayerNorm.weight"]
        let gamma = BNNSNDArrayDescriptor.allocate(initializingFrom: gammaWA.toArray() as [Float32], shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(gamma)

        return BNNS.NormalizationLayer(type: .batch(movingMean: nil, movingVariance: nil), input: input, output: output, beta: beta, gamma: gamma, epsilon: normalizationEpsilon, activation: hiddenActivation)!
    }()

    private lazy var dropout: BNNS.DropoutLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(input)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(output)

        return BNNS.DropoutLayer(input: input, output: output, rate: dropoutRate, seed: 0, control: 0)!
    }()

    deinit {
        allocations.forEach({ $0.deallocate() })
    }

    init(config: Config, weights: Weights = [:]) {
        assert(config.model_type!.stringValue == "bert")
        for key in [
            "bert.embeddings.word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight",
            "bert.embeddings.token_type_embeddings.weight",
        ] { assert(weights.keys.contains(where: { $0 == key })) }
        assert(weights.keys.contains(where: { $0 == "bert.embeddings.LayerNorm.beta" || $0 == "bert.embeddings.LayerNorm.bias" }))
        assert(weights.keys.contains(where: { $0 == "bert.embeddings.LayerNorm.gamma" || $0 == "bert.embeddings.LayerNorm.weight" }))
        assert(config.hidden_act!.stringValue == "gelu")
        assert("absolute" == config.position_embedding_type!.stringValue!)
        self.positionEmbeddingType = config.position_embedding_type!.stringValue!
        self.hiddenSize = config.hidden_size!.intValue!
        self.vocabSize = config.vocab_size!.intValue!
        self.maxPositionEmbeddings = config.max_position_embeddings!.intValue!
        self.typeVocabSize = config.type_vocab_size!.intValue!
        self.padTokenID = config.pad_token_id!.intValue!
        self.normalizationEpsilon = Float(config.layer_norm_eps!.doubleValue!)
        self.weights = weights
   }

    public func callAsFunction(inputIDs: [Int64],
                               tokenTypeIDs: [Int64]? = nil,
                               positionIDs: [Int64]? = nil) -> MLMultiArray {
        let inputLength = inputIDs.count
        let inputIDs: [Int64] = inputIDs.padded(length: maxPositionEmbeddings)
        let wordInput = BNNSNDArrayDescriptor.allocate(initializingFrom: inputIDs, shape: .vector(inputIDs.count))
        let wordOutput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, inputIDs.count))
        defer {
            wordInput.deallocate()
            wordOutput.deallocate()
        }
        try! wordEmbedding.apply(batchSize: 1, input: wordInput, output: wordOutput)

        let positionIDs = positionIDs ?? Array<Int64>(stride(from: 0, through: Int64(inputLength - 1), by: 1))
        let positionInput = BNNSNDArrayDescriptor.allocate(initializingFrom: positionIDs.padded(length: maxPositionEmbeddings), shape: .vector(maxPositionEmbeddings))
        let positionOutput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        defer {
            positionInput.deallocate()
            positionOutput.deallocate()
        }
        try! self.positionEmbedding.apply(batchSize: 1, input: positionInput, output: positionOutput)

        let tokenTypeIDs: [Int64] = tokenTypeIDs ?? Array(repeating: 0, count: maxPositionEmbeddings)
        let typeInput = BNNSNDArrayDescriptor.allocate(initializingFrom: tokenTypeIDs, shape: .vector(maxPositionEmbeddings))
        let typeOutput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        defer {
            typeInput.deallocate()
            typeOutput.deallocate()
        }
        try! self.tokenTypeEmbedding.apply(batchSize: 1, input: typeInput, output: typeOutput)

        let multiWord = try! wordOutput.makeMultiArray(of: Float32.self, shape: shape)
        let multiPosition = try! positionOutput.makeMultiArray(of: Float32.self, shape: shape)
        let multiType = try! typeOutput.makeMultiArray(of: Float32.self, shape: shape)

        return multiWord + multiPosition + multiType
    }
}
