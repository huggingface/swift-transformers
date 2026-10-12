import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

/// `clean_up_tokenization_spaces` is a WordPiece-era step that deletes the space before
/// punctuation. Python transformers skips it for BPE models ("destructive for BPE"), unless
/// `clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output` is set.
/// See https://github.com/huggingface/transformers/pull/44915
/// Expected values below were produced with Python transformers 5.7.0.
@Suite("BPE decode skips clean-up like transformers")
struct BPECleanupTests {
    static let text = "the preju\u{02}dice of the party . . . in other proceedings , he said ."

    @Test
    func llamaBPEDecodeReturnsTheEncodedText() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
        let ids = tokenizer.encode(text: Self.text, addSpecialTokens: false)
        #expect(tokenizer.decode(tokens: ids) == Self.text)
    }
}
