//
//  TrieTests.swift
//
//
//  Created by Pedro Cuenca on 12/1/24.
//

import XCTest
@testable import Tokenizers

class TrieTests: XCTestCase {
    func testTrieBuilding() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")
        XCTAssertEqual(trie.root.children.count, 1)
        
        let c = trie.get("c")
        XCTAssertNotNil(c)
        XCTAssertEqual(c!.children.count, 1)     // "a"
        
        let ca = trie.get("ca")
        XCTAssertNotNil(ca)
        XCTAssertEqual(ca!.children.count, 2)    // "r", "t"
        
        let car = trie.get("car")
        XCTAssertNotNil(car)
        XCTAssertTrue(car!.isLeaf)
        XCTAssertFalse(ca!.isLeaf)
        
        XCTAssertNil(trie.get("card"))
    }
    
    func testTrieCommonPrefixSearch() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")

        // trie.commonPrefixSearch returns [Character] not String
        let leaves = trie.commonPrefixSearch("carpooling").map { String($0) }
        XCTAssertEqual(leaves, ["car", "carp"])
    }
}
