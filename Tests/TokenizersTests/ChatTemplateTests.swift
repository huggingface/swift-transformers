//
//  ChatTemplateTests.swift
//  swift-transformers
//
//  Created by Anthony DePasquale on 2/10/24.
//

import XCTest
import Tokenizers

class ChatTemplateTests: XCTestCase {
    let messages = [[
        "role": "user",
        "content": "Describe the Swift programming language.",
    ]]

    func testTemplateFromConfig() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/Phi-3-mini-128k-instruct")
        let encoded = try tokenizer.applyChatTemplate(messages: messages)
        let encodedTarget = [32010, 4002, 29581, 278, 14156, 8720, 4086, 29889, 32007, 32001]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<|user|>Describe the Swift programming language.<|end|><|assistant|>"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testDefaultTemplateFromArrayInConfig() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        let encoded = try tokenizer.applyChatTemplate(messages: messages)
        let encodedTarget = [1, 29473, 3, 28752, 1040, 4672, 2563, 17060, 4610, 29491, 29473, 4]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testTemplateFromArgumentWithEnum() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/Phi-3-mini-128k-instruct")
        // Purposely not using the correct template for this model to verify that the template from the config is not being used
        let mistral7BDefaultTemplate = "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        let encoded = try tokenizer.applyChatTemplate(messages: messages, chatTemplate: .literal(mistral7BDefaultTemplate))
        let encodedTarget = [1, 518, 25580, 29962, 20355, 915, 278, 14156, 8720, 4086, 29889, 518, 29914, 25580, 29962]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testTemplateFromArgumentWithString() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/Phi-3-mini-128k-instruct")
        // Purposely not using the correct template for this model to verify that the template from the config is not being used
        let mistral7BDefaultTemplate = "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        let encoded = try tokenizer.applyChatTemplate(messages: messages, chatTemplate: mistral7BDefaultTemplate)
        let encodedTarget = [1, 518, 25580, 29962, 20355, 915, 278, 14156, 8720, 4086, 29889, 518, 29914, 25580, 29962]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testNamedTemplateFromArgument() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        // Normally it is not necessary to specify the name `default`, but I'm not aware of models with lists of templates in the config that are not `default` or `tool_use`
        let encoded = try tokenizer.applyChatTemplate(messages: messages, chatTemplate: .name("default"))
        let encodedTarget = [1, 29473, 3, 28752, 1040, 4672, 2563, 17060, 4610, 29491, 29473, 4]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    // TODO: Add tests for tool use template
}
