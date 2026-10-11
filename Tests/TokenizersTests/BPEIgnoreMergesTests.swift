import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

// "hugs" is in the vocabulary, but the merges stop at "hug" + "s".
private let bpeIgnoreMergesVocab: [String: Int] = [
    "h": 0, "u": 1, "g": 2, "s": 3, "ug": 4, "hug": 5, "hugs": 6,
]
private let bpeIgnoreMergesMerges: [[String]] = [["u", "g"], ["h", "ug"]]

private func makeBPETokenizer(ignoreMerges: Bool?) throws -> BPETokenizer {
    var model: [String: Any] = [
        "type": "BPE", "vocab": bpeIgnoreMergesVocab, "merges": bpeIgnoreMergesMerges,
    ]
    model["ignore_merges"] = ignoreMerges
    let tokenizerData = try JSONDecoder().decode(
        Config.self, from: JSONSerialization.data(withJSONObject: ["model": model])
    )
    let tokenizerConfig = try JSONDecoder().decode(
        Config.self, from: JSONSerialization.data(withJSONObject: [String: Any]())
    )
    return try BPETokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: [:])
}

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-tests")
}()

@Suite("BPE ignore_merges")
struct BPEIgnoreMergesTests {
    @Test("A word in the vocabulary is one token when ignore_merges is true")
    func wholeWordInVocabulary() throws {
        let tokenizer = try makeBPETokenizer(ignoreMerges: true)
        let tokens = tokenizer.tokenize(text: "hugs")
        #expect(tokens == ["hugs"])
        #expect(tokens.map { tokenizer.convertTokenToId($0) } == [6])
    }

    @Test("Disabled or absent ignore_merges applies the merges", arguments: [false, nil] as [Bool?])
    func mergesApplied(ignoreMerges: Bool?) throws {
        let tokenizer = try makeBPETokenizer(ignoreMerges: ignoreMerges)
        #expect(tokenizer.tokenize(text: "hugs") == ["hug", "s"])
    }

    @Test("A word not in the vocabulary still goes through the merges", arguments: [true, false])
    func wordNotInVocabulary(ignoreMerges: Bool) throws {
        let tokenizer = try makeBPETokenizer(ignoreMerges: ignoreMerges)
        #expect(tokenizer.tokenize(text: "shug") == ["s", "hug"])
    }

    /// The tokenizer declares `"ignore_merges": true`, and the merges alone do not rebuild these words.
    /// Expected ids from `tokenizers` 0.23.2 on the same revision.
    @Test("granite-embedding-97m-multilingual-r2 matches tokenizers on words only ignore_merges reaches")
    func graniteEmbeddingMultilingual() async throws {
        let tokenizer = try await AutoTokenizer.from(
            pretrained: "ibm-granite/granite-embedding-97m-multilingual-r2",
            revision: "835ad14087e140460703cf0fae09f97d469d65c2",
            hubApi: HubApi(downloadBase: downloadDestination)
        )
        let cases: [(text: String, ids: [Int])] = [
            ("Apparently unswayed", [179934, 128200, 3906, 3438, 255, 179938]),
            ("an der Emission beteiligt war.", [179934, 230, 1180, 4265, 2670, 165354, 3592, 13, 179938]),
            ("leur nouveau territoire", [179934, 20840, 22649, 65310, 179938]),
            ("la enfermedad en particular", [179934, 1625, 62851, 428, 4937, 179938]),
            ("os próprios visitantes", [179934, 325, 96616, 74597, 179938]),
        ]
        for (text, ids) in cases {
            #expect(tokenizer.encode(text: text) == ids, "\(text)")
        }
    }
}
