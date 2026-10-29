import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

@Suite("Vocabulary identity")
struct VocabularyIdentityTests {
    @Test("Canonically equivalent tokens keep distinct IDs", arguments: ["BPE", "Unigram"])
    func distinctTokenIds(modelType: String) throws {
        // Parse JSON directly so neither Swift String dictionaries nor Foundation bridging
        // can collapse canonically equivalent vocabulary keys while constructing the fixture.
        let model =
            modelType == "BPE"
            ? #"{"type":"BPE","vocab":{"<unk>":0," ":1,";":2,"\u037e":3,"\u00e9":4,"e\u0301":5,"\u00f6":6},"merges":[],"ignore_merges":true}"#
            : #"{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0],[" ",-1],[";",-1],["\u037e",-1],["\u00e9",-1],["e\u0301",-1],["\u00f6",-1]]}"#
        let data = try YYJSONParser.parseToConfig(Data("{\"model\":\(model)}".utf8))
        let config = try YYJSONParser.parseToConfig(Data(#"{"unk_token":"<unk>","bos_token":";","eos_token":"\u037e"}"#.utf8))
        let tokenizer: any PreTrainedTokenizerModel
        if modelType == "BPE" {
            tokenizer = try BPETokenizer(tokenizerConfig: config, tokenizerData: data, addedTokens: [:])
        } else {
            tokenizer = try UnigramTokenizer(tokenizerConfig: config, tokenizerData: data, addedTokens: [:])
        }

        let tokens = ["<unk>", " ", ";", "\u{037E}", "\u{00E9}", "e\u{0301}", "\u{00F6}"]
        for (id, token) in tokens.enumerated() {
            #expect(tokenizer.convertTokenToId(token) == id)
            #expect(tokenizer.convertIdToToken(id).map { Array($0.utf8) } == Array(token.utf8))
            #expect(tokenizer.tokenize(text: token).map { tokenizer.convertTokenToId($0) } == [id])
        }
        // A canonically equivalent spelling absent from the vocabulary must stay unknown.
        #expect(tokenizer.convertTokenToId("o\u{0308}") == 0)
        #expect(tokenizer.eosTokenId == 3)
        #expect(tokenizer.bosTokenId == (modelType == "BPE" ? 2 : 1))
    }
}
