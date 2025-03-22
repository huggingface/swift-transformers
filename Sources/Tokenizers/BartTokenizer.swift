import Hub

class BartTokenizer {
    public var bosToken: String?
    public var bosTokenId: Int?
    public var eosToken: String?
    public var eosTokenId: Int?
    public var unknownToken: String?
    public var unknownTokenId: Int?
    public var fuseUnknownTokens: Bool

    public let padToken: String
    public let sepToken: String
    public let clsToken: String
    public let maskToken: String

    private let vocab: [String: Int]
    private let ids_to_tokens: [Int: String]
    private let bpe: BPETokenizer

    required public init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws {
        guard let vocab = tokenizerData.model?.vocab?.dictionary as? [String: Int] else { throw TokenizerError.missingVocab }
        self.bosToken = tokenizerConfig.bosToken?.stringValue ?? "<s>"
        self.bosTokenId = bosToken == nil ? nil : vocab[bosToken!]
        self.eosToken = tokenizerConfig.eosToken?.stringValue ?? "</s>"
        self.eosTokenId = eosToken == nil ? nil : vocab[eosToken!]
        self.unknownToken = tokenizerConfig.unkToken?.stringValue ?? "<unk>"
        self.unknownTokenId = unknownToken == nil ? nil : vocab[unknownToken!]
        self.fuseUnknownTokens = tokenizerConfig.fuseUnk?.boolValue ?? false
        self.padToken = tokenizerConfig.padToken?.stringValue ?? "<pad>"
        self.sepToken = tokenizerConfig.sepToken?.stringValue ?? "</s>"
        self.clsToken = tokenizerConfig.clsToken?.stringValue ?? "<s>"
        self.maskToken = tokenizerConfig.maskToken?.stringValue ?? "<mask>"
        self.vocab = vocab
        self.ids_to_tokens = Utils.invert(vocab)
        self.bpe = try BPETokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }

    func callAsFunction(_ text: String) -> [String] { bpe.tokenize(text: text) }
    func unTokenize(tokens: [Int]) -> [String] { tokens.compactMap({ ids_to_tokens[$0] }) }
}

extension BartTokenizer: PreTrainedTokenizerModel {
    func convertTokenToId(_ token: String) -> Int? { vocab[token] ?? unknownTokenId }
    func convertIdToToken(_ id: Int) -> String? { ids_to_tokens[id] }
    func tokenize(text: String) -> [String] { bpe.tokenize(text: text) }
}
