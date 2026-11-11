import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-tests")
}()

@Suite("Metaspace split integration")
struct MetaspaceSplitTests {
    /// The tokenizer declares a Metaspace pre-tokenizer with `"split": false`, so runs of spaces merge into one token.
    /// Expected ids from `tokenizers` 0.23.3 on the same revision.
    @Test("Mistral-7B-Instruct-v0.3 matches tokenizers on runs of spaces")
    func mistralRunsOfSpaces() async throws {
        let tokenizer = try await AutoTokenizer.from(
            pretrained: "mistralai/Mistral-7B-Instruct-v0.3",
            revision: "c170c708c41dac9275d15a8fff4eca08d52bab71",
            hubApi: HubApi(downloadBase: downloadDestination)
        )
        let cases: [(text: String, ids: [Int])] = [
            ("a    b", [1, 1032, 3055, 1055]),
            ("x        y", [1, 2086, 6158, 1105]),
            ("def f():\n        return 1", [1, 1569, 1053, 6656, 781, 6158, 1372, 29473, 29508]),
            ("Hello  world", [1, 23325, 29473, 2294]),
            ("no double spaces here", [1, 1476, 4347, 11367, 2004]),
        ]
        for (text, ids) in cases {
            #expect(tokenizer.encode(text: text) == ids, "\(text)")
        }
    }
}
