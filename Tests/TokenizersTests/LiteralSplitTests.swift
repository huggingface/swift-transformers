import Testing

@testable import Tokenizers

@Suite("Literal splitting")
struct LiteralSplitTests {
    @Test
    func scalarBoundaries() {
        let text = "a▁\u{301}b▁c"
        #expect(text.split(byLiteral: "▁", behavior: .mergedWithNext) == ["a", "▁\u{301}b", "▁c"])
        #expect(text.split(byLiteral: "▁", behavior: .mergedWithPrevious) == ["a▁", "\u{301}b▁", "c"])
        #expect(text.split(byLiteral: "▁", behavior: .isolated) == ["a", "▁", "\u{301}b", "▁", "c"])
        #expect(text.split(byLiteral: "▁", behavior: .removed) == ["a", "\u{301}b", "c"])
    }

    @Test
    func exactLiteralMatching() {
        #expect("a;b\u{37E}c".split(byLiteral: ";", behavior: .mergedWithNext) == ["a", ";b\u{37E}c"])
        #expect("a.*b".split(byLiteral: ".*", behavior: .isolated) == ["a", ".*", "b"])
        #expect("::a::::b::".split(byLiteral: "::", behavior: .mergedWithNext) == ["::a", "::", "::b", "::"])
    }

    @Test
    func emptyInputsAndDelimiters() {
        for behavior in [SplitDelimiterBehavior.removed, .isolated, .mergedWithPrevious, .mergedWithNext] {
            #expect("".split(byLiteral: "▁", behavior: behavior).isEmpty)
            #expect("".split(byLiteral: "", behavior: behavior).isEmpty)
            #expect("abc".split(byLiteral: "", behavior: behavior) == ["abc"])
            #expect("abc".split(byLiteral: "▁", behavior: behavior) == ["abc"])
        }
    }
}
