//
//  Trie.swift
//
//
//  Created by Pedro Cuenca on 20240112.
//  Copyright © 2024 Hugging Face. All rights reserved.
//

import Foundation

public struct Trie<T: Hashable> {
    public typealias Node = TrieNode<T>
    
    var root: Node
    
    public init(root: Node? = nil) {
        self.root = root ?? Node()
    }
}

extension Trie {
    public func insert(_ element: any Sequence<T>) {
        var node = root
        for item in element {
            if let child = node.children[item] {
                node = child
            } else {
                let child = Node()
                node.children[item] = child
                node = child
            }
        }
        node.isLeaf = true
    }
    
    public func get(_ element: any Sequence<T>) -> Node? {
        var node = root
        for item in element {
            guard let child = node.children[item] else { return nil }
            node = child
        }
        return node
    }
    
    /// Find all leaf nodes that share a common prefix with the input sequence (usually a text)
    // TODO: make this conform to `IteratorProtocol` instead of materializing the array
    public func commonPrefixSearch(_ text: any Sequence<T>) -> [[T]] {
        var node = root
        var seqs: [[T]] = []
        var seq: [T] = []
        for item in text {
            seq.append(item)
            guard let child = node.children[item] else { return seqs }
            node = child
            if node.isLeaf {
                seqs.append(seq)
            }
        }
        return seqs
    }    
}

// TODO: maybe store the scores here if it's helpful?
public class TrieNode<T: Hashable> {
    var isLeaf: Bool = false
    var children: [T: TrieNode] = [:]
}
