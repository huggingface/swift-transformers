import Testing
import TokenizersCore

@Test
func exposesTokenizerConfiguration() {
    let config: Config = ["tokenizer_class": "PreTrainedTokenizer"]

    #expect(config.tokenizerClass.string() == "PreTrainedTokenizer")
}
