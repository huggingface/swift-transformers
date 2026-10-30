import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

private func config(_ object: Any) throws -> Config {
    try JSONDecoder().decode(Config.self, from: JSONSerialization.data(withJSONObject: object))
}

@Suite("BPE merge ranks")
struct BPEMergesTests {
    @Test("Ranks count list and legacy entries, skip anything else, and keep a repeated pair's last rank")
    func ranks() throws {
        let merges = try config([
            ["a", "b"],  // 0
            "c d",  // 1
            ["e", "f", "g"],  // 2: the first two strings
            ["h", 1, "i"],  // 3: non-strings dropped
            7,  // skipped
            ["a", "b"],  // 4: repeated, last rank wins
        ] as [Any]).array()!
        let ranks = BPETokenizer.bpeRanks(merges: merges)
        #expect(ranks.count == 4)
        #expect(ranks[BytePair("a", "b")] == 4)
        #expect(ranks[BytePair("c", "d")] == 1)
        #expect(ranks[BytePair("e", "f")] == 2)
        #expect(ranks[BytePair("h", "i")] == 3)
    }

    @Test("A missing vocabulary throws before any merge is read, malformed merges included")
    func missingVocabThrowsFirst() throws {
        for merges in [[[String]()], ["a"]] as [[Any]] {
            let tokenizerData = try config(["model": ["type": "BPE", "merges": merges]])
            do {
                _ = try BPETokenizer(tokenizerConfig: try config([:]), tokenizerData: tokenizerData, addedTokens: [:])
                Issue.record("expected missingVocab for merges \(merges)")
            } catch TokenizerError.missingVocab {
                // expected
            } catch {
                Issue.record("expected missingVocab, got \(error) for merges \(merges)")
            }
        }
    }
}
