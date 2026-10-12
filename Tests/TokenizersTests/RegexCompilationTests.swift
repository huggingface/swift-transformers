import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

@Suite("Regex compilation")
struct RegexCompilationTests {
    @Test("Invalid Split regex fails tokenizer initialization", arguments: [false, true])
    func invalidRegex(nested: Bool) throws {
        let pattern = "["
        let split = Config(["type": "Split", "pattern": ["Regex": Config(pattern)]])
        let preTokenizer =
            nested
            ? Config(["type": "Sequence", "pretokenizers": [split]])
            : split
        let data = Config([
            "model": ["type": "WordPiece", "vocab": ["hello": 0]],
            "pre_tokenizer": preTokenizer,
        ])
        do {
            _ = try AutoTokenizer.from(
                tokenizerConfig: Config(["tokenizer_class": "BertTokenizer"]), tokenizerData: data
            )
            Issue.record("Expected regex compilation to fail")
        } catch let TokenizerError.invalidRegex(actualPattern, underlyingError) {
            #expect(actualPattern == pattern)
            #expect(!(underlyingError.localizedDescription.isEmpty))
            let error = TokenizerError.invalidRegex(pattern: actualPattern, underlyingError: underlyingError)
            #expect(error.localizedDescription.contains(pattern))
            #expect(error.localizedDescription.contains(underlyingError.localizedDescription))
        }
    }
}
