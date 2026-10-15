import Foundation
import Testing

@testable import Tokenizers

@Suite("Whisper Normalizer Tests")
struct WhisperNormalizerTests {
    // MARK: - Fixture-driven parity tests

    /// A single entry in `whisper-normalizer-baselines.json`. The expected
    /// outputs are produced by the pinned Python `transformers` release
    /// (see `Tools/generate_whisper_normalizer_baselines.py`); these tests
    /// assert byte-identical parity with that reference.
    struct Fixture: Decodable {
        let id: String
        let input: String
        let basic: String
        let basicRemoveDiacritics: String
        let basicSplitLetters: String
        let english: String

        enum CodingKeys: String, CodingKey {
            case id
            case input
            case basic
            case basicRemoveDiacritics = "basic_remove_diacritics"
            case basicSplitLetters = "basic_split_letters"
            case english
        }
    }

    struct Baselines: Decodable {
        let fixtures: [Fixture]
    }

    static let fixtures: [Fixture] = {
        guard
            let url = Bundle.module.url(
                forResource: "whisper-normalizer-baselines",
                withExtension: "json"
            )
        else {
            Issue.record("whisper-normalizer-baselines.json not found in test bundle")
            return []
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(Baselines.self, from: data).fixtures
        } catch {
            Issue.record("Failed to decode whisper-normalizer-baselines.json: \(error)")
            return []
        }
    }()

    @Test(
        "BasicTextNormalizer parity (default)",
        arguments: Self.fixtures
    )
    func basicDefault(fixture: Fixture) {
        let n = BasicTextNormalizer()
        let got = n(fixture.input)
        #expect(
            got == fixture.basic,
            "[\(fixture.id)] input=\(fixture.input.debugDescription) got=\(got.debugDescription) expected=\(fixture.basic.debugDescription)"
        )
    }

    @Test(
        "BasicTextNormalizer parity (removeDiacritics)",
        arguments: Self.fixtures
    )
    func basicRemoveDiacritics(fixture: Fixture) {
        let n = BasicTextNormalizer(removeDiacritics: true)
        let got = n(fixture.input)
        #expect(
            got == fixture.basicRemoveDiacritics,
            "[\(fixture.id)] input=\(fixture.input.debugDescription) got=\(got.debugDescription) expected=\(fixture.basicRemoveDiacritics.debugDescription)"
        )
    }

    @Test(
        "BasicTextNormalizer parity (splitLetters)",
        arguments: Self.fixtures
    )
    func basicSplitLetters(fixture: Fixture) {
        let n = BasicTextNormalizer(splitLetters: true)
        let got = n(fixture.input)
        #expect(
            got == fixture.basicSplitLetters,
            "[\(fixture.id)] input=\(fixture.input.debugDescription) got=\(got.debugDescription) expected=\(fixture.basicSplitLetters.debugDescription)"
        )
    }

    @Test(
        "EnglishTextNormalizer parity",
        arguments: Self.fixtures
    )
    func englishParity(fixture: Fixture) {
        let n = EnglishTextNormalizer()
        let got = n(fixture.input)
        #expect(
            got == fixture.english,
            "[\(fixture.id)] input=\(fixture.input.debugDescription) got=\(got.debugDescription) expected=\(fixture.english.debugDescription)"
        )
    }

    // MARK: - Smoke tests for the public API

    @Test("BasicTextNormalizer is callable")
    func basicCallSyntax() {
        let n = BasicTextNormalizer()
        #expect(n("Hello") == n.normalize("Hello"))
    }

    @Test("EnglishTextNormalizer is callable")
    func englishCallSyntax() {
        let n = EnglishTextNormalizer()
        #expect(n("Hello") == n.normalize("Hello"))
    }

    @Test("EnglishSpellingNormalizer passes unknown words through")
    func spellingPassthrough() {
        let n = EnglishSpellingNormalizer()
        #expect(n("hello world foo bar") == "hello world foo bar")
    }

    @Test("EnglishSpellingNormalizer maps British → American")
    func spellingMapsBritishWords() {
        let n = EnglishSpellingNormalizer()
        #expect(n("colour centre organise") == "color center organize")
    }
}

extension WhisperNormalizerTests.Fixture: CustomTestStringConvertible {
    var testDescription: String { id }
}
