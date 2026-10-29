import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

private let byteFallbackVocab: [(String, Double)] = [
    ("<unk>", 0.0),
    ("a", -1.0),
    ("b", -1.0),
    ("<0x0A>", -5.0),
    ("<0xF0>", -5.0),
    ("<0x9F>", -5.0),
    ("<0x90>", -5.0),
    ("<0x88>", -5.0),
]

private func makeUnigramTokenizer(
    byteFallback: Bool,
    vocab: [(String, Double)] = byteFallbackVocab
) throws -> UnigramTokenizer {
    let model: [String: Any] = [
        "type": "Unigram",
        "unk_id": 0,
        "byte_fallback": byteFallback,
        "vocab": vocab.map { [$0.0, $0.1] as [Any] },
    ]
    let tokenizerData = try JSONDecoder().decode(
        Config.self,
        from: JSONSerialization.data(withJSONObject: ["model": model])
    )
    let tokenizerConfig = try JSONDecoder().decode(
        Config.self,
        from: JSONSerialization.data(withJSONObject: [String: Any]())
    )
    return try UnigramTokenizer(
        tokenizerConfig: tokenizerConfig,
        tokenizerData: tokenizerData,
        addedTokens: [:]
    )
}

@Suite("Unigram byte fallback")
struct UnigramByteFallbackTests {
    @Test("Out-of-vocabulary character becomes its byte token")
    func singleByteCharacter() throws {
        let tokenizer = try makeUnigramTokenizer(byteFallback: true)
        #expect(tokenizer.tokenize(text: "a\nb") == ["a", "<0x0A>", "b"])
    }

    @Test("Consecutive out-of-vocabulary characters are not fused")
    func consecutiveBytesAreNotFused() throws {
        let tokenizer = try makeUnigramTokenizer(byteFallback: true)
        #expect(tokenizer.tokenize(text: "a\n\nb") == ["a", "<0x0A>", "<0x0A>", "b"])
    }

    @Test("Multi-byte character expands to one token per UTF-8 byte")
    func multiByteCharacter() throws {
        let tokenizer = try makeUnigramTokenizer(byteFallback: true)
        #expect(tokenizer.tokenize(text: "\u{1F408}") == ["<0xF0>", "<0x9F>", "<0x90>", "<0x88>"])
    }

    @Test("Byte fallback is skipped when the model does not declare it")
    func disabledByteFallbackIsUnchanged() throws {
        let tokenizer = try makeUnigramTokenizer(byteFallback: false)
        let tokens = tokenizer.tokenize(text: "a\nb")
        #expect(tokens.map { tokenizer.convertTokenToId($0) } == [1, 0, 2])
    }

    @Test("A piece stays unknown when the vocabulary lacks one of its byte tokens")
    func incompleteByteVocabularyFallsBackToUnknown() throws {
        let incomplete = byteFallbackVocab.filter { $0.0 != "<0x88>" }
        let tokenizer = try makeUnigramTokenizer(byteFallback: true, vocab: incomplete)
        let tokens = tokenizer.tokenize(text: "\u{1F408}")
        #expect(tokens.count == 1)
        #expect(tokenizer.convertTokenToId(tokens[0]) == 0)
    }
}
