import CoreML
import Tokenizers
import XCTest

@testable import Generation

@available(macOS 15.0, iOS 18.0, *)
final class GenerationIntegrationTests: XCTestCase {

    // MARK: - Mock Model for Testing

    /// Mock language model that returns predictable logits for testing
    class MockLanguageModel {
        var callCount = 0
        var logitsHistory: [MLTensor] = []

        func predictNextToken(_ inputTokens: MLTensor, _ config: GenerationConfig) async -> MLTensor {
            callCount += 1

            // Return different logits based on the sequence length
            let seqLength = inputTokens.shape[1]

            // Simulate a vocabulary of 10 tokens
            // Create logits that favor certain tokens based on context
            let vocabSize = 10
            var logits = [Float](repeating: 0.0, count: vocabSize)

            switch seqLength {
            case 1:
                // First generation: favor token 5
                logits = [1.0, 1.5, 2.0, 2.5, 3.0, 10.0, 3.0, 2.5, 2.0, 1.5]
            case 2:
                // Second generation: favor token 3
                logits = [1.0, 2.0, 3.0, 8.0, 3.0, 2.0, 2.0, 1.5, 1.0, 0.5]
            case 3:
                // Third generation: create a more uniform distribution
                logits = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 3.5, 3.0, 2.5, 2.0]
            default:
                // Default: slightly favor middle tokens
                logits = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
            }

            let tensor = MLTensor(shape: [1, vocabSize], scalars: logits, scalarType: Float.self)
            logitsHistory.append(tensor)
            return tensor
        }
    }

    // MARK: - Integration Tests

    func testGreedyGenerationWithoutProcessors() async throws {
        let model = MockLanguageModel()

        var config = GenerationConfig(maxNewTokens: 3)
        config.doSample = false // Greedy mode
        config.eosTokenId = -1 // Disable early stopping
        config.maxLength = 10

        let generation = TestGeneration()
        let startTokens = [0] // Start with token 0

        let output = await generation.generate(
            config: config,
            tokens: startTokens,
            model: model.predictNextToken
        )

        // Greedy should always pick the highest logit
        // Token 0 -> Token 5 (logit 10.0) -> Token 3 (logit 8.0) -> Token 5 (logit 4.5)
        XCTAssertEqual(output.count, 4, "Should generate 3 new tokens + initial token")
        XCTAssertEqual(output[0], 0, "First token should be the start token")
        XCTAssertEqual(output[1], 5, "Second token should be 5 (highest logit)")
        XCTAssertEqual(output[2], 3, "Third token should be 3 (highest logit)")
        XCTAssertEqual(output[3], 5, "Fourth token should be 5 (highest logit)")

        XCTAssertEqual(model.callCount, 3, "Model should be called 3 times")
    }

    func testSamplingWithTemperature() async throws {
        let model = MockLanguageModel()

        var config = GenerationConfig(maxNewTokens: 3)
        config.doSample = true // Sampling mode
        config.temperature = 0.1 // Low temperature = more deterministic
        config.eosTokenId = -1

        let generation = TestGeneration()
        let startTokens = [0]

        let output = await generation.generate(
            config: config,
            tokens: startTokens,
            model: model.predictNextToken
        )

        XCTAssertEqual(output.count, 4, "Should generate 3 new tokens + initial token")
        XCTAssertEqual(output[0], 0, "First token should be the start token")

        // With low temperature, sampling should still prefer high-probability tokens
        // We can't assert exact tokens due to randomness, but can verify structure
        XCTAssertTrue(output[1] < 10, "Generated token should be in vocab range")
        XCTAssertTrue(output[2] < 10, "Generated token should be in vocab range")
        XCTAssertTrue(output[3] < 10, "Generated token should be in vocab range")
    }

    func testTopKFiltering() async throws {
        let model = MockLanguageModel()

        var config = GenerationConfig(maxNewTokens: 3)
        config.doSample = true // Sampling mode
        config.topK = 3 // Only consider top 3 tokens
        config.temperature = 1.0
        config.eosTokenId = -1

        let generation = TestGeneration()
        let startTokens = [0]

        // Run generation multiple times to test top-k filtering
        for _ in 0..<5 {
            model.callCount = 0
            model.logitsHistory = []

            let output = await generation.generate(
                config: config,
                tokens: startTokens,
                model: model.predictNextToken
            )

            XCTAssertEqual(output.count, 4, "Should generate 3 new tokens + initial token")

            // Verify that generated tokens are within valid range
            for token in output[1...] {
                XCTAssertTrue(token >= 0 && token < 10, "Token \(token) should be in vocab range")
            }
        }
    }

    func testTopPFiltering() async throws {
        let model = MockLanguageModel()

        var config = GenerationConfig(maxNewTokens: 2)
        config.doSample = true // Sampling mode
        config.topP = 0.9 // Top 90% probability mass
        config.temperature = 1.0
        config.eosTokenId = -1

        let generation = TestGeneration()
        let startTokens = [0]

        let output = await generation.generate(
            config: config,
            tokens: startTokens,
            model: model.predictNextToken
        )

        XCTAssertEqual(output.count, 3, "Should generate 2 new tokens + initial token")

        // Top-P should filter out low-probability tokens
        for token in output[1...] {
            XCTAssertTrue(token >= 0 && token < 10, "Token should be in vocab range")
        }
    }

    func testRepetitionPenalty() async throws {
        // Create a model that always returns the same high-scoring token
        class RepetitiveModel {
            func predict(_ inputTokens: MLTensor, _ config: GenerationConfig) async -> MLTensor {
                // Always favor token 7
                let logits: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0]
                return MLTensor(shape: [1, 10], scalars: logits, scalarType: Float.self)
            }
        }

        let model = RepetitiveModel()

        // Test WITHOUT repetition penalty
        var configNoPenalty = GenerationConfig(maxNewTokens: 3)
        configNoPenalty.doSample = false // Greedy mode
        configNoPenalty.repetitionPenalty = 1.0 // No penalty
        configNoPenalty.eosTokenId = -1

        let generation = TestGeneration()
        let startTokens = [0]

        let outputNoPenalty = await generation.generate(
            config: configNoPenalty,
            tokens: startTokens,
            model: model.predict
        )

        // Without penalty, should keep selecting token 7
        XCTAssertEqual(outputNoPenalty, [0, 7, 7, 7], "Without penalty, should repeat token 7")

        // Test WITH repetition penalty
        var configWithPenalty = GenerationConfig(maxNewTokens: 3)
        configWithPenalty.doSample = false // Greedy mode
        configWithPenalty.repetitionPenalty = 10.0 // Strong penalty (10.0 / 10.0 = 1.0, so other tokens win)
        configWithPenalty.eosTokenId = -1

        let outputWithPenalty = await generation.generate(
            config: configWithPenalty,
            tokens: startTokens,
            model: model.predict
        )

        // With strong penalty, should select token 7 first time, but then avoid it
        XCTAssertEqual(outputWithPenalty[0], 0, "First token should be start token")
        XCTAssertEqual(outputWithPenalty[1], 7, "Second token should be 7 (highest score)")

        // After token 7 is penalized, other tokens should be chosen
        // (exact tokens depend on penalty calculation, but should NOT all be 7)
        let uniqueTokens = Set(outputWithPenalty[1...])
        XCTAssertTrue(uniqueTokens.count > 1, "With repetition penalty, should generate diverse tokens, got \(outputWithPenalty)")
    }

    func testCombinedProcessors() async throws {
        let model = MockLanguageModel()

        var config = GenerationConfig(maxNewTokens: 3)
        config.doSample = true // Sampling mode
        config.temperature = 0.8 // Slightly focused
        config.topK = 5 // Top 5 tokens
        config.topP = 0.95 // 95% probability mass
        config.repetitionPenalty = 1.1 // Slight penalty
        config.eosTokenId = -1

        let generation = TestGeneration()
        let startTokens = [0]

        let output = await generation.generate(
            config: config,
            tokens: startTokens,
            model: model.predictNextToken
        )

        XCTAssertEqual(output.count, 4, "Should generate 3 new tokens + initial token")
        XCTAssertEqual(output[0], 0, "First token should be the start token")

        // All processors should work together
        // Can't assert exact tokens due to randomness, but verify structure
        for token in output[1...] {
            XCTAssertTrue(token >= 0 && token < 10, "Token should be in vocab range")
        }

        // Verify model was called correct number of times
        XCTAssertEqual(model.callCount, 3, "Model should be called 3 times")
    }

    func testMinPFiltering() async throws {
        let model = MockLanguageModel()

        // Test with min-p: keep tokens with prob >= minP * max_prob
        var configWithMinP = GenerationConfig(maxNewTokens: 3)
        configWithMinP.doSample = true // Sampling mode
        configWithMinP.temperature = 1.0 // No temperature adjustment
        configWithMinP.minP = 0.05 // Relatively permissive threshold
        configWithMinP.eosTokenId = -1
        configWithMinP.maxLength = 10

        let generation = TestGeneration()
        let startTokens = [0]

        let output = await generation.generate(
            config: configWithMinP,
            tokens: startTokens,
            model: model.predictNextToken
        )

        XCTAssertEqual(output.count, 4, "Should generate 3 new tokens + initial token")
        XCTAssertEqual(output[0], 0, "First token should be the start token")

        // All tokens should be valid
        for token in output[1...] {
            XCTAssertTrue(token >= 0 && token < 10, "Token should be in vocab range")
        }

        // Test with more aggressive min-p
        model.callCount = 0
        var configAggressiveMinP = GenerationConfig(maxNewTokens: 3)
        configAggressiveMinP.doSample = true
        configAggressiveMinP.temperature = 1.0
        configAggressiveMinP.minP = 0.5 // Much more aggressive
        configAggressiveMinP.eosTokenId = -1
        configAggressiveMinP.maxLength = 10

        let outputAggressive = await generation.generate(
            config: configAggressiveMinP,
            tokens: startTokens,
            model: model.predictNextToken
        )

        XCTAssertEqual(outputAggressive.count, 4, "Should generate 3 new tokens + initial token")
        // With aggressive min-p, should sample from fewer options
        // (exact behavior depends on model, but verify it doesn't crash)
    }

    func testEarlyStoppingWithEOS() async throws {
        // Create a model that returns EOS token after 2 generations
        class EOSModel {
            var callCount = 0

            func predict(_ inputTokens: MLTensor, _ config: GenerationConfig) async -> MLTensor {
                callCount += 1

                let vocabSize = 10
                var logits = [Float](repeating: 1.0, count: vocabSize)

                if callCount >= 2 {
                    // After 2 calls, strongly favor EOS token (which we'll set as token 9)
                    logits[9] = 100.0
                } else {
                    // Before that, favor token 5
                    logits[5] = 10.0
                }

                return MLTensor(shape: [1, vocabSize], scalars: logits, scalarType: Float.self)
            }
        }

        let model = EOSModel()

        var config = GenerationConfig(maxNewTokens: 10) // Request many tokens
        config.doSample = false // Greedy mode
        config.eosTokenId = 9 // Token 9 is EOS

        let generation = TestGeneration()
        let startTokens = [0]

        let output = await generation.generate(
            config: config,
            tokens: startTokens,
            model: model.predict
        )

        // Should stop early when EOS is encountered
        XCTAssertLessThan(output.count, 11, "Should stop before generating all 10 tokens")
        XCTAssertEqual(output[0], 0, "First token should be start token")

        // Model should be called fewer times due to early stopping
        XCTAssertLessThan(model.callCount, 10, "Model should be called fewer times due to EOS")
    }
}

// MARK: - Test Helper

@available(macOS 15.0, iOS 18.0, *)
struct TestGeneration: Generation {
    func generate(
        config: GenerationConfig,
        prompt: String,
        model: NextTokenModel,
        tokenizer: Tokenizers.Tokenizer,
        callback: PredictionStringCallback?
    ) async -> String {
        // Not used in these tests
        return ""
    }
}
