import Testing

@testable import Jinja

@Suite("Template Tests")
struct TemplateTests {
    @Test("Empty template")
    func emptyTemplate() throws {
        let string = ""
        let context: Context = [:]

        let tokens = try Lexer.tokenize(string)
        #expect(tokens == [Token(kind: .eof, value: "", position: 0)])

        let nodes = try Parser.parse(tokens)
        #expect(nodes == [])

        let rendered = try Template(string).render(context)
        #expect(rendered == "")
    }

    @Test("Invalid template")
    func invalidTemplate() throws {
        let string = "{{"
        let context: Context = [:]

        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .openExpression, value: "{{", position: 0),
                Token(kind: .eof, value: "", position: 2),
            ])

        #expect(throws: JinjaError.self) {
            try Parser.parse(tokens)
        }

        #expect(throws: JinjaError.self) {
            try Template(string).render(context)
        }
    }

    @Test("No template")
    func noTemplate() throws {
        let string = #"Hello, world!"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .text, value: "Hello, world!", position: 0),
                Token(kind: .eof, value: "", position: 13),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .text("Hello, world!"),
            ]
        )

        // Check result of template initialized with string
        let rendered = try Template(string).render(context)
        #expect(rendered == "Hello, world!")

        // Check result of template initialized with nodes
        #expect(try rendered == Template(nodes: nodes).render(context))
    }

    @Test("Text nodes")
    func textNodes() throws {
        let string = #"0{{ 'A' }}1{{ 'B' }}{{ 'C' }}2{{ 'D' }}3"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .text, value: "0", position: 0),
                Token(kind: .openExpression, value: "{{", position: 1),
                Token(kind: .string, value: "A", position: 4),
                Token(kind: .closeExpression, value: "}}", position: 8),
                Token(kind: .text, value: "1", position: 10),
                Token(kind: .openExpression, value: "{{", position: 11),
                Token(kind: .string, value: "B", position: 14),
                Token(kind: .closeExpression, value: "}}", position: 18),
                Token(kind: .openExpression, value: "{{", position: 20),
                Token(kind: .string, value: "C", position: 23),
                Token(kind: .closeExpression, value: "}}", position: 27),
                Token(kind: .text, value: "2", position: 29),
                Token(kind: .openExpression, value: "{{", position: 30),
                Token(kind: .string, value: "D", position: 33),
                Token(kind: .closeExpression, value: "}}", position: 37),
                Token(kind: .text, value: "3", position: 39),
                Token(kind: .eof, value: "", position: 40),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .text("0"),
                .expression(.string("A")),
                .text("1"),
                .expression(.string("B")),
                .expression(.string("C")),
                .text("2"),
                .expression(.string("D")),
                .text("3"),
            ]
        )

        // Check result of template initialized with string
        let rendered = try Template(string).render(context)
        #expect(rendered == "0A1BC2D3")

        // Check result of template initialized with nodes
        #expect(try rendered == Template(nodes: nodes).render(context))
    }

    @Test("Boolean literals")
    func booleanLiterals() throws {
        let string = #"|{{ true }}|{{ false }}|{{ True }}|{{ False }}|"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .text, value: "|", position: 0),
                Token(kind: .openExpression, value: "{{", position: 1),
                Token(kind: .boolean, value: "true", position: 4),
                Token(kind: .closeExpression, value: "}}", position: 9),
                Token(kind: .text, value: "|", position: 11),
                Token(kind: .openExpression, value: "{{", position: 12),
                Token(kind: .boolean, value: "false", position: 15),
                Token(kind: .closeExpression, value: "}}", position: 21),
                Token(kind: .text, value: "|", position: 23),
                Token(kind: .openExpression, value: "{{", position: 24),
                Token(kind: .boolean, value: "True", position: 27),
                Token(kind: .closeExpression, value: "}}", position: 32),
                Token(kind: .text, value: "|", position: 34),
                Token(kind: .openExpression, value: "{{", position: 35),
                Token(kind: .boolean, value: "False", position: 38),
                Token(kind: .closeExpression, value: "}}", position: 44),
                Token(kind: .text, value: "|", position: 46),
                Token(kind: .eof, value: "", position: 47),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .text("|"),
                .expression(.boolean(true)),
                .text("|"),
                .expression(.boolean(false)),
                .text("|"),
                .expression(.boolean(true)),
                .text("|"),
                .expression(.boolean(false)),
                .text("|"),
            ]
        )

        // Check result of template initialized with string
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|false|true|false|")

        // Check result of template initialized with nodes
        #expect(try rendered == Template(nodes: nodes).render(context))
    }

    @Test("Logical AND operator")
    func logicalAnd() throws {
        let string =
            #"{{ true and true }}{{ true and false }}{{ false and true }}{{ false and false }}"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .openExpression, value: "{{", position: 0),
                Token(kind: .boolean, value: "true", position: 3),
                Token(kind: .and, value: "and", position: 8),
                Token(kind: .boolean, value: "true", position: 12),
                Token(kind: .closeExpression, value: "}}", position: 17),
                Token(kind: .openExpression, value: "{{", position: 19),
                Token(kind: .boolean, value: "true", position: 22),
                Token(kind: .and, value: "and", position: 27),
                Token(kind: .boolean, value: "false", position: 31),
                Token(kind: .closeExpression, value: "}}", position: 37),
                Token(kind: .openExpression, value: "{{", position: 39),
                Token(kind: .boolean, value: "false", position: 42),
                Token(kind: .and, value: "and", position: 48),
                Token(kind: .boolean, value: "true", position: 52),
                Token(kind: .closeExpression, value: "}}", position: 57),
                Token(kind: .openExpression, value: "{{", position: 59),
                Token(kind: .boolean, value: "false", position: 62),
                Token(kind: .and, value: "and", position: 68),
                Token(kind: .boolean, value: "false", position: 72),
                Token(kind: .closeExpression, value: "}}", position: 78),
                Token(kind: .eof, value: "", position: 80),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(.binary(.and, .boolean(true), .boolean(true))),
                .expression(.binary(.and, .boolean(true), .boolean(false))),
                .expression(.binary(.and, .boolean(false), .boolean(true))),
                .expression(.binary(.and, .boolean(false), .boolean(false))),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "truefalsefalsefalse")
    }

    @Test("Logical OR operator")
    func logicalOr() throws {
        let string =
            #"{{ true or true }}{{ true or false }}{{ false or true }}{{ false or false }}"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .openExpression, value: "{{", position: 0),
                Token(kind: .boolean, value: "true", position: 3),
                Token(kind: .or, value: "or", position: 8),
                Token(kind: .boolean, value: "true", position: 11),
                Token(kind: .closeExpression, value: "}}", position: 16),
                Token(kind: .openExpression, value: "{{", position: 18),
                Token(kind: .boolean, value: "true", position: 21),
                Token(kind: .or, value: "or", position: 26),
                Token(kind: .boolean, value: "false", position: 29),
                Token(kind: .closeExpression, value: "}}", position: 35),
                Token(kind: .openExpression, value: "{{", position: 37),
                Token(kind: .boolean, value: "false", position: 40),
                Token(kind: .or, value: "or", position: 46),
                Token(kind: .boolean, value: "true", position: 49),
                Token(kind: .closeExpression, value: "}}", position: 54),
                Token(kind: .openExpression, value: "{{", position: 56),
                Token(kind: .boolean, value: "false", position: 59),
                Token(kind: .or, value: "or", position: 65),
                Token(kind: .boolean, value: "false", position: 68),
                Token(kind: .closeExpression, value: "}}", position: 74),
                Token(kind: .eof, value: "", position: 76),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(.binary(.or, .boolean(true), .boolean(true))),
                .expression(.binary(.or, .boolean(true), .boolean(false))),
                .expression(.binary(.or, .boolean(false), .boolean(true))),
                .expression(.binary(.or, .boolean(false), .boolean(false))),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "truetruetruefalse")
    }

    @Test("Logical NOT operator")
    func logicalNot() throws {
        let string = #"{{ not true }}{{ not false }}"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .openExpression, value: "{{", position: 0),
                Token(kind: .not, value: "not", position: 3),
                Token(kind: .boolean, value: "true", position: 7),
                Token(kind: .closeExpression, value: "}}", position: 12),
                Token(kind: .openExpression, value: "{{", position: 14),
                Token(kind: .not, value: "not", position: 17),
                Token(kind: .boolean, value: "false", position: 21),
                Token(kind: .closeExpression, value: "}}", position: 27),
                Token(kind: .eof, value: "", position: 29),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(.unary(.not, .boolean(true))),
                .expression(.unary(.not, .boolean(false))),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "falsetrue")
    }

    @Test("Logical NOT NOT operator")
    func logicalNotNot() throws {
        let string = #"{{ not not true }}{{ not not false }}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(.unary(.not, .unary(.not, .boolean(true)))),
                .expression(.unary(.not, .unary(.not, .boolean(false)))),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "truefalse")
    }

    @Test("Logical AND OR combination")
    func logicalAndOr() throws {
        let string =
            #"{{ true and true or false }}{{ true and false or true }}{{ false and true or true }}{{ false and false or true }}{{ false and false or false }}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(
                    .binary(.or, .binary(.and, .boolean(true), .boolean(true)), .boolean(false))),
                .expression(
                    .binary(.or, .binary(.and, .boolean(true), .boolean(false)), .boolean(true))),
                .expression(
                    .binary(.or, .binary(.and, .boolean(false), .boolean(true)), .boolean(true))),
                .expression(
                    .binary(.or, .binary(.and, .boolean(false), .boolean(false)), .boolean(true))),
                .expression(
                    .binary(.or, .binary(.and, .boolean(false), .boolean(false)), .boolean(false))),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "truetruetruetruefalse")
    }

    @Test("Logical AND NOT combination")
    func logicalAndNot() throws {
        let string =
            #"{{ true and not true }}{{ true and not false }}{{ false and not true }}{{ false and not false }}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(.binary(.and, .boolean(true), .unary(.not, .boolean(true)))),
                .expression(.binary(.and, .boolean(true), .unary(.not, .boolean(false)))),
                .expression(.binary(.and, .boolean(false), .unary(.not, .boolean(true)))),
                .expression(.binary(.and, .boolean(false), .unary(.not, .boolean(false)))),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "falsetruefalsefalse")
    }

    @Test("Logical OR NOT combination")
    func logicalOrNot() throws {
        let string =
            #"{{ true or not true }}{{ true or not false }}{{ false or not true }}{{ false or not false }}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(.binary(.or, .boolean(true), .unary(.not, .boolean(true)))),
                .expression(.binary(.or, .boolean(true), .unary(.not, .boolean(false)))),
                .expression(.binary(.or, .boolean(false), .unary(.not, .boolean(true)))),
                .expression(.binary(.or, .boolean(false), .unary(.not, .boolean(false)))),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "truetruefalsetrue")
    }

    @Test("Logical combined with comparison")
    func logicalCombined() throws {
        let string = #"{{ 1 == 2 and 2 == 2 }}{{ 1 == 2 or 2 == 2 }}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(
                    .binary(
                        .and, .binary(.equal, .integer(1), .integer(2)),
                        .binary(.equal, .integer(2), .integer(2))
                    )),
                .expression(
                    .binary(
                        .or, .binary(.equal, .integer(1), .integer(2)),
                        .binary(.equal, .integer(2), .integer(2))
                    )),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "falsetrue")
    }

    @Test("If statement only")
    func ifOnly() throws {
        let string = #"{% if 1 == 1 %}{{ 'A' }}{% endif %}{{ 'B' }}"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .openStatement, value: "{%", position: 0),
                Token(kind: .if, value: "if", position: 3),
                Token(kind: .number, value: "1", position: 6),
                Token(kind: .equal, value: "==", position: 8),
                Token(kind: .number, value: "1", position: 11),
                Token(kind: .closeStatement, value: "%}", position: 13),
                Token(kind: .openExpression, value: "{{", position: 15),
                Token(kind: .string, value: "A", position: 18),
                Token(kind: .closeExpression, value: "}}", position: 22),
                Token(kind: .openStatement, value: "{%", position: 24),
                Token(kind: .endif, value: "endif", position: 27),
                Token(kind: .closeStatement, value: "%}", position: 33),
                Token(kind: .openExpression, value: "{{", position: 35),
                Token(kind: .string, value: "B", position: 38),
                Token(kind: .closeExpression, value: "}}", position: 42),
                Token(kind: .eof, value: "", position: 44),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .statement(
                    .if(.binary(.equal, .integer(1), .integer(1)), [.expression(.string("A"))], [])),
                .expression(.string("B")),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "AB")
    }

    @Test("If else statement")
    func ifElseOnly() throws {
        let string = #"{% if 1 == 2 %}{{ 'A' }}{% else %}{{ 'B' }}{% endif %}{{ 'C' }}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .statement(
                    .if(
                        .binary(.equal, .integer(1), .integer(2)), [.expression(.string("A"))],
                        [.expression(.string("B"))]
                    )),
                .expression(.string("C")),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "BC")
    }

    @Test("If elif else statement")
    func ifElifElse() throws {
        let string =
            #"{% if 1 == 2 %}{{ 'A' }}{{ 'B' }}{{ 'C' }}{% elif 1 == 2 %}{{ 'D' }}{% elif 1 == 3 %}{{ 'E' }}{{ 'F' }}{% else %}{{ 'G' }}{{ 'H' }}{{ 'I' }}{% endif %}{{ 'J' }}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .statement(
                    .if(
                        .binary(.equal, .integer(1), .integer(2)),
                        [
                            .expression(.string("A")), .expression(.string("B")),
                            .expression(.string("C")),
                        ],
                        [
                            .statement(
                                .if(
                                    .binary(.equal, .integer(1), .integer(2)),
                                    [.expression(.string("D"))],
                                    [
                                        .statement(
                                            .if(
                                                .binary(.equal, .integer(1), .integer(3)),
                                                [
                                                    .expression(.string("E")),
                                                    .expression(.string("F")),
                                                ],
                                                [
                                                    .expression(.string("G")),
                                                    .expression(.string("H")),
                                                    .expression(.string("I")),
                                                ]
                                            )),
                                    ]
                                )),
                        ]
                    )),
                .expression(.string("J")),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "GHIJ")
    }

    @Test("Nested statements")
    func nestedStatements() throws {
        let string =
            #"{% set a = 0 %}{% set b = 0 %}{% set c = 0 %}{% set d = 0 %}{% if 1 == 1 %}{% set a = 2 %}{% set b = 3 %}{% elif 1 == 2 %}{% set c = 4 %}{% else %}{% set d = 5 %}{% endif %}{{ a }}{{ b }}{{ c }}{{ d }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "2300")
    }

    @Test("For loop")
    func forLoop() throws {
        let string = #"{% for message in messages %}{{ message['content'] }}{% endfor %}"#
        let context: Context = [
            "messages": [
                ["role": "user", "content": "A"],
                ["role": "assistant", "content": "B"],
                ["role": "user", "content": "C"],
            ],
        ]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .statement(
                    .for(
                        .single("message"), .identifier("messages"),
                        [
                            .expression(
                                .member(.identifier("message"), .string("content"), computed: true)),
                        ],
                        [], test: nil
                    )),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "ABC")
    }

    @Test("For loop unpacking")
    func forLoopUnpacking() throws {
        let string = #"|{% for x, y in [ [1, 2], [3, 4] ] %}|{{ x + ' ' + y }}|{% endfor %}|"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .text("|"),
                .statement(
                    .for(
                        .tuple(["x", "y"]),
                        .array([
                            .array([.integer(1), .integer(2)]), .array([.integer(3), .integer(4)]),
                        ]),
                        [
                            .text("|"),
                            .expression(
                                .binary(
                                    .add, .binary(.add, .identifier("x"), .string(" ")),
                                    .identifier("y")
                                )), .text("|"),
                        ],
                        [], test: nil
                    )),
                .text("|"),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "||1 2||3 4||")
    }

    @Test("For loop with else")
    func forLoopDefault() throws {
        let string = #"{% for x in [] %}{{ 'A' }}{% else %}{{'B'}}{% endfor %}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .statement(
                    .for(
                        .single("x"), .array([]),
                        [.expression(.string("A"))],
                        [.expression(.string("B"))], test: nil
                    )),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "B")
    }

    @Test("For loop with if filter")
    func forLoopIfFilter() throws {
        let string = #"{% for x in [1, 2, 3, 4] if x > 2 %}{{ x }}{% endfor %}"#
        let context: Context = [:]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .statement(
                    .for(
                        .single("x"),
                        .array([.integer(1), .integer(2), .integer(3), .integer(4)]),
                        [.expression(.identifier("x"))],
                        [],
                        test: .binary(.greater, .identifier("x"), .integer(2))
                    )
                ),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "34")
    }

    @Test("For loop with selectattr")
    func forLoopWithSelectAttr() throws {
        let string =
            #"{% for x in arr | selectattr('value', 'equalto', 'a') %}{{ x['value'] }}{% endfor %}"#
        let context: Context = [
            "arr": [
                ["value": "a"],
                ["value": "b"],
                ["value": "c"],
                ["value": "a"],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "aa")
    }

    @Test("For loop with break")
    func forLoopBreak() throws {
        let string =
            #"{% for x in [1, 2, 3, 4] %}{% if x == 3 %}{% break %}{% endif %}{{ x }}{% endfor %}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "12")
    }

    @Test("For loop with continue")
    func forLoopContinue() throws {
        let string =
            #"{% for x in [1, 2, 3, 4] %}{% if x == 3 %}{% continue %}{% endif %}{{ x }}{% endfor %}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "124")
    }

    @Test("For loop with objects")
    func forLoopObjects() throws {
        let string = #"{% for x in obj %}{{ x + ':' + obj[x] + ';' }}{% endfor %}"#
        let context: Context = [
            "obj": [
                "a": 1,
                "b": 2,
                "c": 3,
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "a:1;b:2;c:3;")
    }

    @Test("Variable assignment")
    func variables() throws {
        let string = #"{% set x = 'Hello' %}{% set y = 'World' %}{{ x + ' ' + y }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "Hello World")
    }

    @Test("Variable assignment with method call")
    func variableAssignmentWithMethodCall() throws {
        let string = #"{% set x = 'Hello'.split('el')[-1] %}{{ x }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "lo")
    }

    @Test("Variable block assignment")
    func variablesBlock() throws {
        let string = "{% set x %}Hello!\nMultiline/block set!\n{% endset %}{{ x }}"
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "Hello!\nMultiline/block set!\n")
    }

    @Test("Variable unpacking")
    func variablesUnpacking() throws {
        let string =
            #"|{% set x, y = 1, 2 %}{{ x }}{{ y }}|{% set (x, y) = [1, 2] %}{{ x }}{{ y }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|12|12|")
    }

    @Test("Numbers and arithmetic")
    func numbers() throws {
        let string = #"|{{ 5 }}|{{ -5 }}|{{ add(3, -1) }}|{{ (3 - 1) + (a - 5) - (a + 5)}}|"#
        let context: Context = [
            "a": 0,
            "add": .function { (args: [Value], _, _) -> Value in
                guard args.count == 2,
                      case let .int(x) = args[0],
                      case let .int(y) = args[1]
                else {
                    throw JinjaError.runtime("Invalid arguments for add function")
                }
                return .int(x + y)
            },
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|5|-5|2|-8|")
    }

    @Test("Binary expressions")
    func binopExpr() throws {
        let string =
            #"{{ 1 % 2 }}{{ 1 < 2 }}{{ 1 > 2 }}{{ 1 >= 2 }}{{ 2 <= 2 }}{{ 2 == 2 }}{{ 2 != 3 }}{{ 2 + 3 }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "1truefalsefalsetruetruetrue5")
    }

    @Test("Binary expressions with concatenation")
    func binaryExpressionsWithConcatenation() throws {
        let string = #"{{ 1 ~ "+" ~ 2 ~ "=" ~ 3 ~ " is " ~ true }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "1+2=3 is true")
    }

    @Test("String literals")
    func strings() throws {
        let string = #"{{ 'Bye' }}{{ bos_token + '[INST] ' }}"#
        let context: Context = [
            "bos_token": "<s>",
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "Bye<s>[INST] ")
    }

    @Test("String literals with quotes")
    func stringLiteralsWithQuotes() throws {
        // Test basic double quotes
        let simple1 = #"{{ "test" }}"#
        #expect(try Template(simple1).render([:]) == "test")

        // Test basic single quotes
        let simple2 = #"{{ 'test' }}"#
        #expect(try Template(simple2).render([:]) == "test")

        // Test mixed quotes in concatenation
        let simple3 = #"{{ "a" + 'b' + "c" }}"#
        #expect(try Template(simple3).render([:]) == "abc")

        // Test escaped single quote
        let simple4 = #"{{ '\'' }}"#
        #expect(try Template(simple4).render([:]) == "'")

        // Test escaped double quote
        let simple5 = #"{{ "\"" }}"#
        #expect(try Template(simple5).render([:]) == "\"")
    }

    @Test("String length")
    func stringLength() throws {
        let string = #"|{{ "" | length }}|{{ "a" | length }}|{{ '' | length }}|{{ 'a' | length }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|0|1|0|1|")
    }

    @Test("String literals with template syntax")
    func stringLiteralsWithTemplateSyntax() throws {
        let string = #"|{{ '{{ "hi" }}' }}|{{ '{% if true %}{% endif %}' }}|"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .text, value: "|", position: 0),
                Token(kind: .openExpression, value: "{{", position: 1),
                Token(kind: .string, value: "{{ \"hi\" }}", position: 4),
                Token(kind: .closeExpression, value: "}}", position: 17),
                Token(kind: .text, value: "|", position: 19),
                Token(kind: .openExpression, value: "{{", position: 20),
                Token(kind: .string, value: "{% if true %}{% endif %}", position: 23),
                Token(kind: .closeExpression, value: "}}", position: 50),
                Token(kind: .text, value: "|", position: 52),
                Token(kind: .eof, value: "", position: 53),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .text("|"),
                .expression(.string("{{ \"hi\" }}")),
                .text("|"),
                .expression(.string("{% if true %}{% endif %}")),
                .text("|"),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|{{ \"hi\" }}|{% if true %}{% endif %}|")

        // Check result of template initialized with nodes
        #expect(try rendered == Template(nodes: nodes).render(context))
    }

    @Test("String concatenation")
    func stringConcatenation() throws {
        let string = #"{{ 'a' + 'b' 'c' }}"#
        let context: Context = [:]

        // Check result of lexer
        let tokens = try Lexer.tokenize(string)
        #expect(
            tokens == [
                Token(kind: .openExpression, value: "{{", position: 0),
                Token(kind: .string, value: "a", position: 3),
                Token(kind: .plus, value: "+", position: 7),
                Token(kind: .string, value: "b", position: 9),
                Token(kind: .string, value: "c", position: 13),
                Token(kind: .closeExpression, value: "}}", position: 17),
                Token(kind: .eof, value: "", position: 19),
            ]
        )

        // Check result of parser
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .expression(
                    .binary(
                        .add,
                        .string("a"),
                        .binary(.concat, .string("b"), .string("c"))
                    )
                ),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "abc")
    }

    @Test("Function calls")
    func functions() throws {
        let string = #"{{ func() }}{{ func(apple) }}{{ func(x, 'test', 2, false) }}"#
        let context: Context = [
            "x": 10,
            "apple": "apple",
            "func": .function { (args: [Value], _, _) -> Value in
                return .int(args.count)
            },
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "014")
    }

    @Test("Object properties")
    func properties() throws {
        let string = #"{{ obj.x + obj.y }}{{ obj['x'] + obj.y }}"#
        let context: Context = [
            "obj": [
                "x": 10,
                "y": 20,
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "3030")
    }

    @Test("Object methods")
    func objMethods() throws {
        let string = #"{{ obj.x(x, y) }}{{ ' ' + obj.x() + ' ' }}{{ obj.z[x](x, y) }}"#
        let context: Context = [
            "x": "A",
            "y": "B",
            "obj": [
                "x": .function { (args: [Value], _, _) -> Value in
                    let strings = args.compactMap { value in
                        if case let .string(str) = value { return str }
                        return nil
                    }
                    return .string(strings.joined(separator: ""))
                },
                "z": [
                    "A": .function { (args: [Value], _, _) -> Value in
                        let strings = args.compactMap { value in
                            if case let .string(str) = value { return str }
                            return nil
                        }
                        return .string(strings.joined(separator: "_"))
                    },
                ],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "AB  A_B")
    }

    @Test("String methods")
    func stringMethods() throws {
        let string =
            #"{{ '  A  '.strip() }}{% set x = '  B  ' %}{{ x.strip() }}{% set y = ' aBcD ' %}{{ y.upper() }}{{ y.lower() }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "AB ABCD  abcd ")
    }

    @Test("String methods title")
    func stringMethods2() throws {
        let string = #"{{ 'test test'.title() }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "Test Test")
    }

    @Test("String rstrip")
    func rstrip() throws {
        let string = #"{{ "   test it  ".rstrip() }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "   test it")
    }

    @Test("String lstrip")
    func lstrip() throws {
        let string = #"{{ "   test it  ".lstrip() }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "test it  ")
    }

    @Test("String split")
    func split() throws {
        let string = #"|{{ "   test it  ".split() | join("|") }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|test|it|")
    }

    @Test("String split with separator")
    func stringSplitWithSeparator() throws {
        let string = #"|{{ "   test it  ".split(" ") | join("|") }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "||||test|it|||")
    }

    @Test("String split with limit")
    func stringSplitWithLimit() throws {
        let string = #"|{{ "   test it  ".split(" ", 4) | join("|") }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "||||test|it  |")
    }

    @Test("String replace")
    func replace() throws {
        let string =
            #"|{{ "test test".replace("test", "TEST") }}|{{ "test test".replace("test", "TEST", 1) }}|{{ "test test".replace("", "_", 2) }}|{{ "abcabc".replace("a", "x", count=1) }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|TEST TEST|TEST test|_t_est test|xbcabc|")
    }

    @Test("String slicing")
    func stringSlicing() throws {
        let string =
            #"|{{ x[0] }}|{{ x[:] }}|{{ x[:3] }}|{{ x[1:4] }}|{{ x[1:-1] }}|{{ x[1::2] }}|{{ x[5::-1] }}|"#
        let context: Context = [
            "x": "0123456789",
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|0|0123456789|012|123|12345678|13579|543210|")
    }

    @Test("Array slicing")
    func arraySlicing() throws {
        let string =
            #"|{{ strings[0] }}|{% for s in strings[:] %}{{ s }}{% endfor %}|{% for s in strings[:3] %}{{ s }}{% endfor %}|{% for s in strings[1:4] %}{{ s }}{% endfor %}|{% for s in strings[1:-1] %}{{ s }}{% endfor %}|{% for s in strings[1::2] %}{{ s }}{% endfor %}|{% for s in strings[5::-1] %}{{ s }}{% endfor %}|"#
        let context: Context = [
            "strings": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|0|0123456789|012|123|12345678|13579|543210|")
    }

    @Test("Membership operators")
    func membership() throws {
        let string =
            #"|{{ 0 in arr }}|{{ 1 in arr }}|{{ true in arr }}|{{ false in arr }}|{{ 'a' in arr }}|{{ 'b' in arr }}|"#
        let context: Context = [
            "arr": [0, true, "a"],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|false|true|false|true|false|")
    }

    @Test("Membership negation with not")
    func membershipNegationWithNot() throws {
        let string =
            #"|{{ not 0 in arr }}|{{ not 1 in arr }}|{{ not true in arr }}|{{ not false in arr }}|{{ not 'a' in arr }}|{{ not 'b' in arr }}|"#
        let context: Context = [
            "arr": [0, true, "a"],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|false|true|false|true|false|true|")
    }

    @Test("Membership negation with not in")
    func membershipNegationWithNotIn() throws {
        let string =
            #"|{{ 0 not in arr }}|{{ 1 not in arr }}|{{ true not in arr }}|{{ false not in arr }}|{{ 'a' not in arr }}|{{ 'b' not in arr }}|"#
        let context: Context = [
            "arr": [0, true, "a"],
        ]

        // Check result of parser
        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .text("|"),
                .expression(.binary(.notIn, .integer(0), .identifier("arr"))),
                .text("|"),
                .expression(.binary(.notIn, .integer(1), .identifier("arr"))),
                .text("|"),
                .expression(.binary(.notIn, .boolean(true), .identifier("arr"))),
                .text("|"),
                .expression(.binary(.notIn, .boolean(false), .identifier("arr"))),
                .text("|"),
                .expression(.binary(.notIn, .string("a"), .identifier("arr"))),
                .text("|"),
                .expression(.binary(.notIn, .string("b"), .identifier("arr"))),
                .text("|"),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|false|true|false|true|false|true|")
    }

    @Test("Membership with undefined")
    func membershipUndefined() throws {
        let string =
            #"|{{ x is defined }}|{{ y is defined }}|{{ x in y }}|{{ y in x }}|{{ 1 in y }}|{{ 1 in x }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|false|false|false|false|false|false|")
    }

    @Test("Escaped characters")
    func escapedChars() throws {
        // Start with simpler cases
        let simple1 = "{{ '\\n' }}"
        #expect(try Template(simple1).render([:]) == "\n")

        let simple2 = "{{ '\\t' }}"
        #expect(try Template(simple2).render([:]) == "\t")

        let simple3 = "{{ '\\\\' }}"
        #expect(try Template(simple3).render([:]) == "\\")

        // More complex case
        let string = "{{ '\\n' }}{{ '\\t' }}{{ '\\\\' }}"
        let rendered = try Template(string).render([:])
        #expect(rendered == "\n\t\\")
    }

    @Test("Substring inclusion")
    func substringInclusion() throws {
        let string =
            #"|{{ '' in 'abc' }}|{{ 'a' in 'abc' }}|{{ 'd' in 'abc' }}|{{ 'ab' in 'abc' }}|{{ 'ac' in 'abc' }}|{{ 'abc' in 'abc' }}|{{ 'abcd' in 'abc' }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|true|false|true|false|true|false|")
    }

    @Test("Filter operator")
    func filterOperator() throws {
        let string =
            #"{{ arr | length }}{{ 1 + arr | length }}{{ 2 + arr | sort | length }}{{ (arr | sort)[0] }}"#
        let context: Context = [
            "arr": [3, 2, 1],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "3451")
    }

    @Test("Filter operator string transformations")
    func filterOperatorStringTransformations() throws {
        let string =
            #"|{{ 'abc' | length }}|{{ 'aBcD' | upper }}|{{ 'aBcD' | lower }}|{{ 'test test' | capitalize}}|{{ 'test test' | title }}|{{ ' a b ' | trim }}|{{ '  A  B  ' | trim | lower | length }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|3|ABCD|abcd|Test test|Test Test|a b|4|")
    }

    @Test("Filter operator abs")
    func filterOperatorAbs() throws {
        let string = #"|{{ -1 | abs }}|{{ 1 | abs }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|1|1|")
    }

    @Test("Filter operator selectattr")
    func filterOperatorSelectAttr() throws {
        let string = #"{{ items | selectattr('key') | length }}"#
        let context: Context = [
            "items": [
                ["key": "a"],
                ["key": 0],
                ["key": 1],
                [:],
                ["key": false],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "2")
    }

    @Test("Filter operator selectattr with equalto")
    func filterOperatorSelectAttrWithEqualTo() throws {
        let string = #"{{ messages | selectattr('role', 'equalto', 'system') | length }}"#
        let context: Context = [
            "messages": [
                ["role": "system"],
                ["role": "user"],
                ["role": "assistant"],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "1")
    }

    @Test("Filter operator tojson")
    func filterOperatorToJson() throws {
        let string =
            #"|{{ obj | tojson }}|{{ "test" | tojson }}|{{ 1 | tojson }}|{{ true | tojson }}|{{ null | tojson }}|{{ [1,2,3] | tojson }}|"#
        let context: Context = [
            "obj": [
                "string": "world",
                "number": 5,
                "boolean": true,
                "null": nil,
                "array": [1, 2, 3],
                "object": ["key": "value"],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered.contains("\"key\":\"value\""))
        #expect(rendered.contains("\"test\""))
        #expect(rendered.contains("1"))
        #expect(rendered.contains("true"))
        #expect(rendered.contains("null"))
        #expect(rendered.contains("[1,2,3]"))
    }

    @Test("Filter statements")
    func filterStatements() throws {
        let string = #"{% filter upper %}text{% endfilter %}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "TEXT")
    }

    @Test("Boolean operations with numbers")
    func booleanNumerical() throws {
        let string =
            #"|{{ 1 and 2 }}|{{ 1 and 0 }}|{{ 0 and 1 }}|{{ 0 and 0 }}|{{ 1 or 2 }}|{{ 1 or 0 }}|{{ 0 or 1 }}|{{ 0 or 0 }}|{{ not 1 }}|{{ not 0 }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|2|0|0|0|1|1|1|0|false|true|")
    }

    @Test("Boolean operations with strings")
    func booleanStrings() throws {
        let string =
            #"|{{ 'a' and 'b' }}|{{ 'a' and '' }}|{{ '' and 'a' }}|{{ '' and '' }}|{{ 'a' or 'b' }}|{{ 'a' or '' }}|{{ '' or 'a' }}|{{ '' or '' }}|{{ not 'a' }}|{{ not '' }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|b||||a|a|a||false|true|")
    }

    @Test("Boolean operations mixed")
    func booleanMixed() throws {
        let string =
            #"|{{ true and 1 }}|{{ true and 0 }}|{{ false and 1 }}|{{ false and 0 }}|{{ true or 1 }}|{{ true or 0 }}|{{ false or 1 }}|{{ false or 0 }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|1|0|false|false|true|true|1|0|")
    }

    @Test("Boolean operations mixed with strings")
    func booleanOperationsMixedWithStrings() throws {
        let string =
            #"|{{ true and '' }}|{{ true and 'a' }}|{{ false or '' }}|{{ false or 'a' }}|{{ '' and true }}|{{ 'a' and true }}|{{ '' or false }}|{{ 'a' or false }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "||a||a||true|false|a|")
    }

    @Test("Boolean operations in if statements")
    func booleanMixedIf() throws {
        let string =
            #"{% if '' %}{{ 'A' }}{% endif %}{% if 'a' %}{{ 'B' }}{% endif %}{% if true and '' %}{{ 'C' }}{% endif %}{% if true and 'a' %}{{ 'D' }}{% endif %}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "BD")
    }

    @Test("Ternary operator")
    func ternaryOperator() throws {
        let string =
            #"|{{ 'a' if true else 'b' }}|{{ 'a' if false else 'b' }}|{{ 'a' if 1 + 1 == 2 else 'b' }}|{{ 'a' if 1 + 1 == 3 or 1 * 2 == 3 else 'b' }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|a|b|a|b|")
    }

    @Test("Ternary operator with length")
    func ternaryOperatorWithLength() throws {
        let string = #"{{ (x if true else []) | length }}"#
        let context: Context = [
            "x": [[:], [:], [:]],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "3")
    }

    @Test("Ternary set")
    func ternarySet() throws {
        let string = #"{% set x = 1 if True else 2 %}{{ x }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "1")
    }

    @Test("Ternary consecutive")
    func ternaryConsecutive() throws {
        let string = #"{% set x = 1 if False else 2 if False else 3 %}{{ x }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "3")
    }

    @Test("Ternary shortcut")
    func ternaryShortcut() throws {
        let string = #"{{ 'foo' if false }}{{ 'bar' if true }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "bar")
    }

    @Test("Array literals")
    func arrayLiterals() throws {
        let string = #"{{ [1, true, 'hello', [1, 2, 3, 4], var] | length }}"#
        let context: Context = [
            "var": true,
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "5")
    }

    @Test("Tuple literals")
    func tupleLiterals() throws {
        let string = #"{{ (1, (1, 2)) | length }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "2")
    }

    @Test("Object literals")
    func objectLiterals() throws {
        let string =
            #"{{ { 'key': 'value', 'key2': 'value2', "key3": [1, {'foo': 'bar'} ] }['key'] }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "value")
    }

    @Test("Object literals nested")
    func objectLiteralsNested() throws {
        // Test simple object with minimal spacing
        let simple = #"{{ {'key': 'value'}}}"#
        let simpleRendered = try Template(simple).render([:])
        #expect(simpleRendered.contains("key"))

        // Test nested object
        let nested = #"{{ {'outer': {'inner': 'value'}} }}"#
        let nestedRendered = try Template(nested).render([:])
        #expect(nestedRendered.contains("inner"))

        // Test with member access - this was failing
        let string = #"{{ {'key': {'key': 'value'}}['key']['key'] }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "value")
    }

    @Test("Array operators")
    func arrayOperators() throws {
        let string = #"{{ ([1, 2, 3] + [4, 5, 6]) | length }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "6")
    }

    @Test("Object operators")
    func objectOperators() throws {
        let string =
            #"|{{ 'known' in obj }}|{{ 'known' not in obj }}|{{ 'unknown' in obj }}|{{ 'unknown' not in obj }}|"#
        let context: Context = [
            "obj": [
                "known": true,
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|false|false|true|")
    }

    @Test("Object get method")
    func objectGetMethod() throws {
        let string =
            #"|{{ obj.get('known') }}|{{ obj.get('unknown') is none }}|{{ obj.get('unknown') is defined }}|"#
        let context: Context = [
            "obj": [
                "known": true,
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|true|true|")
    }

    @Test("Object items method")
    func objectItemsMethod() throws {
        let string = #"|{% for x, y in obj.items() %}|{{ x + ' ' + y }}|{% endfor %}|"#
        let context: Context = [
            "obj": [
                "a": 1,
                "b": 2,
                "c": 3,
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "||a 1||b 2||c 3||")
    }

    @Test("Scope without namespace")
    func scopeWithoutNamespace() throws {
        let string =
            #"{% set found = false %}{% for num in nums %}{% if num == 1 %}{{ 'found=' }}{% set found = true %}{% endif %}{% endfor %}{{ found }}"#
        let context: Context = [
            "nums": [1, 2, 3],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "found=false")
    }

    @Test("Undefined variables")
    func undefinedVariables() throws {
        let string = #"{{ undefined_variable }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "")
    }

    @Test("Undefined access")
    func undefinedAccess() throws {
        let string = #"{{ object.undefined_attribute }}"#
        let context: Context = [
            "object": [:],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "")
    }

    @Test("Null variable")
    func nullVariable() throws {
        let string =
            #"{% if not null_val is defined %}{% set null_val = none %}{% endif %}{% if null_val is not none %}{{ 'fail' }}{% else %}{{ 'pass' }}{% endif %}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "pass")
    }

    @Test("Macros")
    func macros() throws {
        let string =
            #"{% macro hello(name) %}{{ 'Hello ' + name }}{% endmacro %}|{{ hello('Bob') }}|{{ hello('Alice') }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|Hello Bob|Hello Alice|")
    }

    @Test("Macros with default parameters")
    func macrosWithDefaultParameters() throws {
        let string =
            #"{% macro hello(name, suffix='.') %}{{ 'Hello ' + name + suffix }}{% endmacro %}|{{ hello('A') }}|{{ hello('B', '!') }}|{{ hello('C', suffix='?') }}|"#
        let context: Context = [:]

        let tokens = try Lexer.tokenize(string)
        let nodes = try Parser.parse(tokens)
        #expect(
            nodes == [
                .statement(
                    .macro(
                        "hello", ["name", "suffix"], ["suffix": .string(".")],
                        [
                            .expression(
                                .binary(
                                    .add,
                                    .binary(.add, .string("Hello "), .identifier("name")),
                                    .identifier("suffix")
                                )
                            ),
                        ]
                    )),
                .text("|"),
                .expression(.call(.identifier("hello"), [.string("A")], [:])),
                .text("|"),
                .expression(.call(.identifier("hello"), [.string("B"), .string("!")], [:])),
                .text("|"),
                .expression(.call(.identifier("hello"), [.string("C")], ["suffix": .string("?")])),
                .text("|"),
            ]
        )

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|Hello A.|Hello B!|Hello C?|")
    }

    @Test("Macros with multiple default parameters")
    func macrosWithMultipleDefaultParameters() throws {
        let string =
            #"{% macro fn(x, y=2, z=3) %}{{ x + ',' + y + ',' + z }}{% endmacro %}|{{ fn(1) }}|{{ fn(1, 0) }}|{{ fn(1, 0, -1) }}|{{ fn(1, y=0, z=-1) }}|{{ fn(1, z=0) }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|1,2,3|1,0,3|1,0,-1|1,0,-1|1,2,0|")
    }

    @Test("Macros with caller")
    func macrosWithCaller() throws {
        let string =
            #"{%- macro dummy(a, b='!') -%}{{ a }} {{ caller() }}{{ b }}{%- endmacro %}{%- call dummy('hello') -%}name{%- endcall -%}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "hello name!")
    }

    @Test("Macros with caller and parameters")
    func macrosWithCallerAndParameters() throws {
        let string =
            "{%- macro print_users(users) -%}{%- for user in users -%}{{ caller(user) }}{%- endfor -%}{%- endmacro %}{% call(user) print_users(users) %}  - {{ user.firstname }} {{ user.lastname }}\n{% endcall %}"
        let context: Context = [
            "users": [
                ["firstname": "John", "lastname": "Doe"],
                ["firstname": "Jane", "lastname": "Smith"],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "  - John Doe\n  - Jane Smith\n")
    }

    @Test("Context-specific keywords")
    func contextKeywords() throws {
        let string =
            #"{% if if in in %}a{% endif %}{% set if = "a" %}{% set in = "abc" %}{% if if in in %}b{% endif %}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "b")
    }

    @Test("Context-specific keywords with assignment")
    func contextKeywordsWithAssignment() throws {
        let string = #"|{{ if }}|{% set if = 2 %}{% if if == 2 %}{{ if }}{% endif %}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "||2|")
    }

    @Test("Unpacking")
    func unpacking() throws {
        let string =
            #"{% macro mul(a, b, c) %}{{ a * b * c }}{% endmacro %}|{{ mul(1, 2, 3) }}|{{ mul(*[1, 2, 3]) }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|6|6|")
    }

    @Test("Filter operator with items")
    func filterOperatorWithItems() throws {
        let string = #"|{{ obj | length }}|{{ (obj | items)[1:] | length }}|"#
        let context: Context = [
            "obj": [
                "a": 1,
                "b": 2,
                "c": 3,
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|3|2|")
    }

    @Test("Filter operator with tojson indent")
    func filterOperatorWithToJsonIndent() throws {
        let string = #"{{ obj | tojson(indent=2) }}"#
        let context: Context = [
            "obj": [
                "string": "world",
                "number": 5,
                "boolean": true,
                "null": nil,
                "array": [1, 2, 3],
                "object": ["key": "value"],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered.contains(#""string" : "world""#))
        #expect(rendered.contains(#""number" : 5"#))
    }

    @Test("Filter operator with map")
    func filterOperatorWithMap() throws {
        let string = #"{{ data | map(attribute='val') | list | tojson }}"#
        let context: Context = [
            "data": [
                ["val": "a"],
                ["val": "b"],
                ["val": "c"],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered.contains("a"))
        #expect(rendered.contains("b"))
        #expect(rendered.contains("c"))
    }

    @Test("Filter operator with indent")
    func filterOperatorWithIndent() throws {
        let string =
            #"|{{ " 1 \n 2 \n 3 \n\n " | indent }}|{{ " 1 \n 2 \n 3 \n\n " | indent(2) }}|{{ " 1 \n 2 \n 3 \n\n " | indent(first=True) }}|{{ " 1 \n 2 \n 3 \n\n " | indent(blank=True) }}|{{ " 1 \n 2 \n 3 \n\n " | indent(4, first=True) }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered.contains(" 1"))
        #expect(rendered.contains(" 2"))
        #expect(rendered.contains(" 3"))
    }

    @Test("Filter operator with rejectattr")
    func filterOperatorWithRejectAttr() throws {
        let string = #"{{ items | rejectattr('key') | length }}"#
        let context: Context = [
            "items": [
                ["key": "a"],
                ["key": 0],
                ["key": 1],
                [:],
                ["key": false],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "3")
    }

    @Test("Filter operator with rejectattr equalto")
    func filterOperatorWithRejectAttrEqualTo() throws {
        let string = #"{{ messages | rejectattr('role', 'equalto', 'system') | length }}"#
        let context: Context = [
            "messages": [
                ["role": "system"],
                ["role": "user"],
                ["role": "assistant"],
            ],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "2")
    }

    @Test("Filter operator with string conversion")
    func filterOperatorWithStringConversion() throws {
        let string = #"{{ tools | string }}"#
        let context: Context = [
            "tools": [1, 2, 3],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "[1, 2, 3]")
    }

    @Test("Filter operator with int conversion")
    func filterOperatorWithIntConversion() throws {
        // Test simple filter first
        let simple = #"{{ "1" | int }}"#
        #expect(try Template(simple).render([:]) == "1")

        // Test simple arithmetic
        let arithmetic = #"{{ 1 + 2 }}"#
        #expect(try Template(arithmetic).render([:]) == "3")

        // Test filter with arithmetic - this currently fails
        let withArith = #"{{ "1" | int + 2 }}"#
        #expect(try Template(withArith).render([:]) == "3")
    }

    @Test("Filter operator with float conversion")
    func filterOperatorWithFloatConversion() throws {
        let string =
            #"|{{ "1.5" | float }}|{{ "invalid" | float }}|{{ "invalid" | float("hello") }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|1.5|0.0|0.0|")
    }

    @Test("Filter operator with replace count")
    func filterOperatorWithReplaceCount() throws {
        let string =
            #"|{{ "abcabcabc" | replace("a", "b") }}|{{ "abcabcabc" | replace("a", "b", 1) }}|{{ "abcabcabc" | replace("a", "b", count=1) }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|bbcbbcbbc|bbcabcabc|bbcabcabc|")
    }

    @Test("Filter operator with default")
    func filterOperatorWithDefault() throws {
        let string =
            #"|{{ undefined | default("hello") }}|{{ false | default("hello") }}|{{ false | default("hello", true) }}|{{ 0 | default("hello", boolean=true) }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|hello|false|hello|0|")
    }

    @Test("Filter operator with unique")
    func filterOperatorWithUnique() throws {
        let string = #"{{ [1, 2, 1, -1, 2] | unique | list | length }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "3")
    }

    @Test("Is operator with defined")
    func isOperatorWithDefined() throws {
        let string =
            #"|{{ unknown_var is defined }}|{{ unknown_var is not defined }}|{{ known_var is defined }}|{{ known_var is not defined }}|"#
        let context: Context = [
            "known_var": "test",
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|false|true|true|false|")
    }

    @Test("Is operator with boolean")
    func isOperatorWithBoolean() throws {
        let string =
            #"|{{ true is true }}|{{ true is not true }}|{{ true is false }}|{{ true is not false }}|{{ true is boolean }}|{{ 1 is boolean }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|false|false|true|true|false|")
    }

    @Test("Is operator with odd even")
    func isOperatorWithOddEven() throws {
        let string =
            #"|{{ 1 is odd }}|{{ 2 is odd }}|{{ 1 is even }}|{{ 2 is even }}|{{ 2 is number }}|{{ '2' is number }}|{{ 2 is integer }}|{{ '2' is integer }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|false|false|true|true|false|true|false|")
    }

    @Test("Is operator with callable")
    func isOperatorWithCallable() throws {
        let string =
            #"|{{ func is callable }}|{{ 2 is callable }}|{{ 1 is iterable }}|{{ 'hello' is iterable }}|"#
        let context: Context = [
            "func": .function { _, _, _ in .string("test") },
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|false|false|true|")
    }

    @Test("Is operator with case")
    func isOperatorWithCase() throws {
        let string =
            #"|{{ 'a' is lower }}|{{ 'A' is lower }}|{{ 'a' is upper }}|{{ 'A' is upper }}|"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|false|false|true|")
    }

    @Test("Is operator with mapping")
    func isOperatorWithMapping() throws {
        let string =
            #"|{{ string is mapping }}|{{ number is mapping }}|{{ array is mapping }}|{{ object is mapping }}|"#
        let context: Context = [
            "string": "hello",
            "number": 42,
            "array": [1, 2, 3],
            "object": ["key": "value"],
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|false|false|false|true|")
    }

    @Test("Short circuit with false and")
    func shortCircuitWithFalseAnd() throws {
        let string = #"{{ false and raise_exception('This should not be printed') }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "false")
    }

    @Test("Short circuit with true or")
    func shortCircuitWithTrueOr() throws {
        let string = #"{{ true or raise_exception('This should not be printed') }}"#
        let context: Context = [:]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "true")
    }

    @Test("Filter with arguments")
    func filterWithArguments() throws {
        let string = #"{{ "hello world"|replace("world", "jinja") }}"#
        let context: Context = [
            "a": 0,
            "add": .function { (args: [Value], _, _) -> Value in
                guard args.count == 2,
                      case let .int(x) = args[0],
                      case let .int(y) = args[1]
                else {
                    throw JinjaError.runtime("Invalid arguments for add function")
                }
                return .int(x + y)
            },
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "hello jinja")
    }

    @Test("Callable test")
    func callableTest() throws {
        let string =
            #"|{{ func is callable }}|{{ 2 is callable }}|{{ 1 is iterable }}|{{ 'hello' is iterable }}|"#
        let context: Context = [
            "func": .function { _, _, _ in .string("test") },
        ]

        // Check result of template
        let rendered = try Template(string).render(context)
        #expect(rendered == "|true|false|false|true|")
    }

    @Test("Keyword arguments")
    func keywordArguments() throws {
        let string = #"{{ greet(name="world") }}"#
        let context: Context = [
            "greet": .function { args, kwargs, _ in
                guard case let .string(name) = kwargs["name"] else {
                    return .string("Hello, stranger!")
                }
                return .string("Hello, \(name)!")
            },
        ]

        let rendered = try Template(string).render(context)
        #expect(rendered == "Hello, world!")
    }

    @Test("Floor division operator template")
    func floorDivisionTemplate() throws {
        let string = "{{ 20 // 7 }}"
        let context: Context = [:]

        let rendered = try Template(string).render(context)
        #expect(rendered == "2")
    }

    @Test("Exponentiation operator template")
    func exponentiationTemplate() throws {
        let string = "{{ 2**3 }}"
        let context: Context = [:]

        let rendered = try Template(string).render(context)
        #expect(rendered == "8")
    }

    @Test("Chained exponentiation left-to-right")
    func chainedExponentiationTemplate() throws {
        let string = "{{ 3**3**3 }}"
        let context: Context = [:]

        let rendered = try Template(string).render(context)
        // This should evaluate as (3**3)**3 = 27**3 = 19683
        #expect(rendered == "19683")
    }

    @Test("Mixed operators precedence")
    func mixedOperatorsPrecedence() throws {
        let string = "{{ 2 + 3 * 4**2 }}"
        let context: Context = [:]

        let rendered = try Template(string).render(context)
        // This should evaluate as 2 + (3 * (4**2)) = 2 + (3 * 16) = 2 + 48 = 50
        #expect(rendered == "50")
    }

    @Test("raise_exception template")
    func raiseExceptionTemplate() throws {
        let string = "{{ raise_exception() }}"
        let context: Context = [:]

        #expect(throws: TemplateException.self) {
            try Template(string).render(context)
        }
    }

    @Test("raise_exception with message template")
    func raiseExceptionWithMessageTemplate() throws {
        let string = #"{{ raise_exception("Template error: invalid input") }}"#
        let context: Context = [:]

        do {
            _ = try Template(string).render(context)
            Issue.record("Expected exception to be thrown")
        } catch let error as TemplateException {
            #expect(error.message == "Template error: invalid input")
        }
    }
}
