import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-tests")
}()

@Suite("BPE ignore_merges integration")
struct BPEIgnoreMergesTests {
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
