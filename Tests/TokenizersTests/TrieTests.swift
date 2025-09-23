//
//  TrieTests.swift
//
//
//  Created by Pedro Cuenca on 12/1/24.
//

import Foundation
import Testing
@testable import Tokenizers

@Suite("Trie data structure functionality")
struct TrieTests {
    @Test("Trie building and traversal")
    func trieBuilding() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")
        #expect(trie.root.children.count == 1)

        let c = trie.get("c")
        #expect(c != nil)
        #expect(c!.children.count == 1) // "a"

        let ca = trie.get("ca")
        #expect(ca != nil)
        #expect(ca!.children.count == 2) // "r", "t"

        let car = trie.get("car")
        #expect(car != nil)
        #expect(car!.isLeaf)
        #expect(!ca!.isLeaf)

        #expect(trie.get("card") == nil)
    }

    @Test("Trie common prefix search")
    func trieCommonPrefixSearch() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")

        // trie.commonPrefixSearch returns [Character] not String
        let leaves = trie.commonPrefixSearch("carpooling").map { String($0) }
        #expect(leaves == ["car", "carp"])
    }

    @Test("Trie common prefix search iterator")
    func trieCommonPrefixSearchIterator() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")

        var expected = Set(["car", "carp"])
        for leaf in trie.commonPrefixSearchIterator("carpooling").map({ String($0) }) {
            #expect(expected.contains(leaf))
            expected.remove(leaf)
        }
        #expect(expected.count == 0)
    }
}
