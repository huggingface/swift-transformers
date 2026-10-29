import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

@Suite("Local tokenizer loading")
struct TokenizerTests {
    @Test
    func tokenizerFromLocalFolder() async throws {
        let bundle = Bundle.module
        guard
            let tokenizerConfigURL = bundle.url(
                forResource: "tokenizer_config",
                withExtension: "json"
            ),
            bundle.url(
                forResource: "tokenizer",
                withExtension: "json"
            ) != nil
        else {
            Issue.record("Missing offline tokenizer fixtures")
            return
        }

        let configuration = LanguageModelConfigurationFromHub(modelFolder: tokenizerConfigURL.deletingLastPathComponent())

        let tokenizerConfigOpt = try await configuration.tokenizerConfig
        #expect(tokenizerConfigOpt != nil)
        let tokenizerConfig = tokenizerConfigOpt!
        let tokenizerData = try await configuration.tokenizerData

        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData
        )

        let encoded = tokenizer.encode(text: "offline path")
        #expect(!encoded.isEmpty)
    }
}
