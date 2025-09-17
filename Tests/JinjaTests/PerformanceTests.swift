// https://github.com/johnmai-dev/Jinja/blob/main/Tests/PerformanceTests.swift

import Foundation
import Testing

@testable import Jinja

@Suite("Performance Tests", .serialized, .enabled(if: true))
struct PerformanceTests {
    // MARK: - Test Data

    private static let llama3_2Template = """
    {%- for message in messages -%}
    {%- if message.role == 'system' -%}
    <|start_header_id|>system<|end_header_id|>

    {{ message.content }}<|eot_id|>
    {%- elif message.role == 'user' -%}
    <|start_header_id|>user<|end_header_id|>

    {{ message.content }}<|eot_id|>
    {%- elif message.role == 'assistant' -%}
    <|start_header_id|>assistant<|end_header_id|>

    {{ message.content }}<|eot_id|>
    {%- endif -%}
    {%- endfor -%}
    {%- if add_generation_prompt -%}
    <|start_header_id|>assistant<|end_header_id|>

    {%- endif -%}
    """

    /// Complex chat templates from ChatTemplateTests
    private static let qwenTemplate = """
    {% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}
    """

    private static let mistralTemplate = """
    {{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
    """

    private static let llamaTokenizerTemplate = """
    {% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}
    """

    private static let complexMistralNemoTemplate = """
    {%- if messages[0]["role"] == "system" %}
        {%- set system_message = messages[0]["content"] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set loop_messages = messages %}
    {%- endif %}
    {%- if not tools is defined %}
        {%- set tools = none %}
    {%- endif %}
    {%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

    {%- for message in loop_messages | rejectattr("role", "equalto", "tool") | rejectattr("role", "equalto", "tool_results") | selectattr("tool_calls", "undefined") %}
        {%- if (message["role"] == "user") != (loop.index0 % 2 == 0) %}
            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}
        {%- endif %}
    {%- endfor %}

    {{- bos_token }}
    {%- for message in loop_messages %}
        {%- if message["role"] == "user" %}
            {%- if tools is not none and (message == user_messages[-1]) %}
                {{- "[AVAILABLE_TOOLS][" }}
                {%- for tool in tools %}
            {%- set tool = tool.function %}
            {{- '{"type": "function", "function": {' }}
            {%- for key, val in tool.items() if key != "return" %}
                {%- if val is string %}
                {{- '"' + key + '": "' + val + '"' }}
                {%- else %}
                {{- '"' + key + '": ' + val|tojson }}
                {%- endif %}
                {%- if not loop.last %}
                {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
            {%- endif %}
            {%- if loop.last and system_message is defined %}
                {{- "[INST]" + system_message + "\\n\\n" + message["content"] + "[/INST]" }}
            {%- else %}
                {{- "[INST]" + message["content"] + "[/INST]" }}
            {%- endif %}
        {%- elif message["role"] == "tool_calls" or message.tool_calls is defined %}
            {%- if message.tool_calls is defined %}
                {%- set tool_calls = message.tool_calls %}
            {%- else %}
                {%- set tool_calls = message.content %}
            {%- endif %}
            {{- "[TOOL_CALLS][" }}
            {%- for tool_call in tool_calls %}
                {%- set out = tool_call.function|tojson %}
                {{- out[:-1] }}
                {%- if not tool_call.id is defined or tool_call.id|length != 9 %}
                    {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
                {%- endif %}
                {{- ', "id": "' + tool_call.id + '"}' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" + eos_token }}
                {%- endif %}
            {%- endfor %}
        {%- elif message["role"] == "assistant" %}
            {{- message["content"] + eos_token}}
        {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
            {%- if message.content is defined and message.content.content is defined %}
                {%- set content = message.content.content %}
            {%- else %}
                {%- set content = message.content %}
            {%- endif %}
            {{- '[TOOL_RESULTS]{"content": ' + content|string + ", " }}
            {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
            {%- endif %}
            {{- '"call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
        {%- else %}
            {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
        {%- endif %}
    {%- endfor %}
    """

    private static let weatherQueryMessages: [String: Value] = [
        "messages": .array([
            .object([
                "role": .string("system"),
                "content": .string(
                    "You are a helpful assistant that provides weather information."),
            ]),
            .object([
                "role": .string("user"),
                "content": .string("What's the weather like today?"),
            ]),
        ]),
    ]

    private static let multiTurnMessages: [String: Value] = [
        "messages": .array([
            .object([
                "role": .string("system"),
                "content": .string("You are a coding assistant."),
            ]),
            .object([
                "role": .string("user"),
                "content": .string("How do I create a for loop in Python?"),
            ]),
            .object([
                "role": .string("assistant"),
                "content": .string(
                    "In Python, you can create a for loop using this syntax:\n\n```python\nfor item in sequence:\n    # code here\n```"
                ),
            ]),
            .object([
                "role": .string("user"),
                "content": .string("Can you show me an example with numbers?"),
            ]),
            .object([
                "role": .string("assistant"),
                "content": .string(
                    "Sure! Here's an example:\n\n```python\nfor i in range(5):\n    print(i)\n```\n\nThis will print numbers 0 through 4."
                ),
            ]),
        ]),
    ]

    private static let largeMessageArray: [String: Value] = {
        var messages: [Value] = []
        for i in 0..<50 {
            messages.append(
                .object([
                    "role": .string(i % 2 == 0 ? "user" : "assistant"),
                    "content": .string(
                        "This is message number \(i) with some content that varies in length to test performance with different message sizes."
                    ),
                ]))
        }
        return ["messages": .array(messages)]
    }()

    private static let complexNestedData: [String: Value] = [
        "messages": .array([
            .object([
                "role": .string("system"),
                "content": .string("You are a helpful assistant."),
                "metadata": .object([
                    "timestamp": .string("2024-01-01T00:00:00Z"),
                    "version": .string("1.0"),
                    "features": .array([.string("chat"), .string("tools"), .string("vision")]),
                ]),
            ]),
            .object([
                "role": .string("user"),
                "content": .string("Help me with this complex task"),
                "attachments": .array([
                    .object([
                        "type": .string("image"),
                        "url": .string("https://example.com/image.jpg"),
                        "metadata": .object([
                            "width": .int(1920),
                            "height": .int(1080),
                        ]),
                    ]),
                ]),
            ]),
            .object([
                "role": .string("assistant"),
                "content": .string("I'd be happy to help you with that complex task!"),
                "tool_calls": .array([
                    .object([
                        "id": .string("call_123456"),
                        "type": .string("function"),
                        "function": .object([
                            "name": .string("get_weather"),
                            "arguments": .string("{\"location\": \"Paris\"}"),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]

    // MARK: - Performance Measurement Helper

    private func measureMs(iterations: Int = 100, warmup: Int = 10, _ body: () throws -> Void)
        rethrows -> Double
    {
        // Warmup
        for _ in 0..<warmup {
            try body()
        }

        var total: Double = 0
        for _ in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            try body()
            let end = DispatchTime.now().uptimeNanoseconds
            total += Double(end - start) / 1_000_000.0
        }
        return total / Double(iterations)
    }

    // MARK: - Performance Tests

    @Test("Basic template render performance")
    func basicTemplateRenderPerformance() async throws {
        let template = try Template(Self.llama3_2Template)

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": Self.weatherQueryMessages["messages"]!,
                "add_generation_prompt": .boolean(true),
            ])
        }
        print("Basic template.render avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Complex chat template performance - Qwen")
    func qwenTemplatePerformance() async throws {
        let template = try Template(Self.qwenTemplate)

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": Self.multiTurnMessages["messages"]!,
                "add_generation_prompt": .boolean(true),
            ])
        }
        print("Qwen template.render avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Complex chat template performance - Mistral")
    func mistralTemplatePerformance() async throws {
        let template = try Template(Self.mistralTemplate)

        // Mistral template requires alternating user/assistant roles
        let mistralMessages: [String: Value] = [
            "messages": .array([
                .object([
                    "role": .string("user"),
                    "content": .string("Hello, how are you?"),
                ]),
                .object([
                    "role": .string("assistant"),
                    "content": .string("I'm doing great. How can I help you today?"),
                ]),
                .object([
                    "role": .string("user"),
                    "content": .string("I'd like to show off how chat templating works!"),
                ]),
            ]),
        ]

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": mistralMessages["messages"]!,
                "bos_token": .string("<s>"),
                "eos_token": .string("</s>"),
            ])
        }
        print("Mistral template.render avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Complex chat template performance - Llama Tokenizer")
    func llamaTokenizerTemplatePerformance() async throws {
        let template = try Template(Self.llamaTokenizerTemplate)

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": Self.multiTurnMessages["messages"]!,
                "bos_token": .string("<s>"),
                "eos_token": .string("</s>"),
                "USE_DEFAULT_PROMPT": .boolean(true),
            ])
        }
        print("Llama Tokenizer template.render avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Ultra-complex template performance - Mistral Nemo")
    func mistralNemoTemplatePerformance() async throws {
        let template = try Template(Self.complexMistralNemoTemplate)

        // Create proper data for Mistral Nemo template
        let mistralNemoData: [String: Value] = [
            "messages": .array([
                .object([
                    "role": .string("user"),
                    "content": .string("Hello, how are you?"),
                ]),
                .object([
                    "role": .string("assistant"),
                    "content": .string("I'm doing great. How can I help you today?"),
                ]),
            ]),
            "bos_token": .string("<s>"),
            "eos_token": .string("</s>"),
        ]

        let avgMs = try measureMs {
            _ = try template.render(mistralNemoData)
        }
        print("Mistral Nemo template.render avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Large message array performance")
    func largeMessageArrayPerformance() async throws {
        let template = try Template(Self.llama3_2Template)

        let avgMs = try measureMs(iterations: 50) {
            _ = try template.render([
                "messages": Self.largeMessageArray["messages"]!,
                "add_generation_prompt": .boolean(true),
            ])
        }
        print("Large message array (50 messages) avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Multi-turn conversation performance")
    func multiTurnConversationPerformance() async throws {
        let template = try Template(Self.llama3_2Template)

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": Self.multiTurnMessages["messages"]!,
                "add_generation_prompt": .boolean(true),
            ])
        }
        print("Multi-turn conversation avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Complex nested data performance")
    func complexNestedDataPerformance() async throws {
        let template = try Template(Self.llama3_2Template)

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": Self.complexNestedData["messages"]!,
                "add_generation_prompt": .boolean(true),
            ])
        }
        print("Complex nested data avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Pipeline stages performance - Basic")
    func pipelineStagesPerformanceBasic() async throws {
        let template = Self.llama3_2Template

        // tokenize
        let tokenizeMs = try measureMs {
            _ = try Lexer.tokenize(template)
        }

        let tokens = try Lexer.tokenize(template)

        // parse
        let parseMs = try measureMs {
            _ = try Parser.parse(tokens)
        }

        let program = try Parser.parse(tokens)

        // interpret
        let env = Environment()
        env["messages"] = Self.weatherQueryMessages["messages"]!
        env["add_generation_prompt"] = .boolean(false)

        let runMs = try measureMs {
            _ = try Interpreter.interpret(program, environment: env)
        }

        print(
            "Basic - tokenize: \(String(format: "%.3f", tokenizeMs)) ms | parse: \(String(format: "%.3f", parseMs)) ms | run: \(String(format: "%.3f", runMs)) ms"
        )
    }

    @Test("Pipeline stages performance - Complex")
    func pipelineStagesPerformanceComplex() async throws {
        let template = Self.complexMistralNemoTemplate

        // tokenize
        let tokenizeMs = try measureMs {
            _ = try Lexer.tokenize(template)
        }

        let tokens = try Lexer.tokenize(template)

        // parse
        let parseMs = try measureMs {
            _ = try Parser.parse(tokens)
        }

        let program = try Parser.parse(tokens)

        // interpret
        let env = Environment()
        env["messages"] = .array([
            .object([
                "role": .string("user"),
                "content": .string("Hello, how are you?"),
            ]),
            .object([
                "role": .string("assistant"),
                "content": .string("I'm doing great. How can I help you today?"),
            ]),
        ])
        env["bos_token"] = .string("<s>")
        env["eos_token"] = .string("</s>")

        let runMs = try measureMs {
            _ = try Interpreter.interpret(program, environment: env)
        }

        print(
            "Complex - tokenize: \(String(format: "%.3f", tokenizeMs)) ms | parse: \(String(format: "%.3f", parseMs)) ms | run: \(String(format: "%.3f", runMs)) ms"
        )
    }

    @Test("Template compilation performance comparison")
    func templateCompilationPerformanceComparison() async throws {
        let templates = [
            ("Basic Llama3", Self.llama3_2Template),
            ("Qwen", Self.qwenTemplate),
            ("Mistral", Self.mistralTemplate),
            ("Llama Tokenizer", Self.llamaTokenizerTemplate),
            ("Mistral Nemo", Self.complexMistralNemoTemplate),
        ]

        for (name, templateString) in templates {
            let compileMs = try measureMs {
                _ = try Template(templateString)
            }
            print("\(name) compilation avg: \(String(format: "%.3f", compileMs)) ms")
        }
    }

    @Test("Filter performance tests")
    func filterPerformanceTests() async throws {
        let filterTemplate = """
        {% for message in messages %}
            {% if message.role == 'user' %}
                {{ message.content | upper }}
            {% elif message.role == 'assistant' %}
                {{ message.content | lower }}
            {% endif %}
            {% if not loop.last %}
                {{ message.content | length }}
            {% endif %}
        {% endfor %}
        """

        let template = try Template(filterTemplate)

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": Self.largeMessageArray["messages"]!,
            ])
        }
        print("Filter operations avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Loop performance with different iteration counts")
    func loopPerformanceWithDifferentIterationCounts() async throws {
        let loopTemplate = """
        {% for message in messages %}
            {{ loop.index }}: {{ message.role }} - {{ message.content[:50] }}
        {% endfor %}
        """

        let template = try Template(loopTemplate)

        // Test with different message counts
        let messageCounts = [5, 10, 25, 50, 100]

        for count in messageCounts {
            let testMessages: [Value] = (0..<count).map { i in
                .object([
                    "role": .string(i % 2 == 0 ? "user" : "assistant"),
                    "content": .string("Message \(i) with some content"),
                ])
            }

            let avgMs = try measureMs(iterations: 20) {
                _ = try template.render(["messages": .array(testMessages)])
            }
            print("Loop with \(count) messages avg: \(String(format: "%.3f", avgMs)) ms")
        }
    }

    @Test("Conditional logic performance")
    func conditionalLogicPerformance() async throws {
        let conditionalTemplate = """
        {% for message in messages %}
            {% if message.role == 'system' %}
                SYSTEM: {{ message.content }}
            {% elif message.role == 'user' %}
                USER: {{ message.content }}
            {% elif message.role == 'assistant' %}
                ASSISTANT: {{ message.content }}
            {% else %}
                OTHER: {{ message.content }}
            {% endif %}
            {% if loop.first %}
                (First message)
            {% elif loop.last %}
                (Last message)
            {% else %}
                (Message {{ loop.index }} of {{ loop.length }})
            {% endif %}
        {% endfor %}
        """

        let template = try Template(conditionalTemplate)

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": Self.multiTurnMessages["messages"]!,
            ])
        }
        print("Complex conditional logic avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("String concatenation performance")
    func stringConcatenationPerformance() async throws {
        let concatTemplate = """
        {% for message in messages %}
            {{ 'Role: ' + message.role + ' | Content: ' + message.content + ' | Index: ' + loop.index|string }}
        {% endfor %}
        """

        let template = try Template(concatTemplate)

        let avgMs = try measureMs {
            _ = try template.render([
                "messages": Self.largeMessageArray["messages"]!,
            ])
        }
        print("String concatenation avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Memory usage with large templates")
    func memoryUsageWithLargeTemplates() async throws {
        // Create a very large template with many nested conditions
        let largeTemplate = """
        {% for message in messages %}
            {% for i in range(10) %}
                {% if message.role == 'user' %}
                    User message {{ i }}: {{ message.content }}
                {% elif message.role == 'assistant' %}
                    Assistant response {{ i }}: {{ message.content }}
                {% endif %}
                {% if i % 2 == 0 %}
                    Even iteration: {{ i }}
                {% else %}
                    Odd iteration: {{ i }}
                {% endif %}
            {% endfor %}
        {% endfor %}
        """

        let template = try Template(largeTemplate)

        let avgMs = try measureMs(iterations: 20) {
            _ = try template.render([
                "messages": Self.largeMessageArray["messages"]!,
            ])
        }
        print("Large template with nested loops avg: \(String(format: "%.3f", avgMs)) ms")
    }

    @Test("Scalability test - increasing complexity")
    func scalabilityTestIncreasingComplexity() async throws {
        let baseTemplate = """
        {% for message in messages %}
            {% if message.role == 'user' %}
                <|start_header_id|>user<|end_header_id|>
                {{ message.content }}<|eot_id|>
            {% elif message.role == 'assistant' %}
                <|start_header_id|>assistant<|end_header_id|>
                {{ message.content }}<|eot_id|>
            {% endif %}
        {% endfor %}
        """

        let template = try Template(baseTemplate)

        // Test with progressively larger message arrays
        let sizes = [10, 25, 50, 100, 200]

        for size in sizes {
            let testMessages: [Value] = (0..<size).map { i in
                .object([
                    "role": .string(i % 2 == 0 ? "user" : "assistant"),
                    "content": .string(
                        "This is message number \(i) with some content to test performance"),
                ])
            }

            let avgMs = try measureMs(iterations: 10) {
                _ = try template.render(["messages": .array(testMessages)])
            }
            print("Scalability test - \(size) messages: \(String(format: "%.3f", avgMs)) ms")
        }
    }
}
