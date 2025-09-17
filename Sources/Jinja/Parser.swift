import Foundation
import OrderedCollections

/// Parses tokens into an abstract syntax tree for Jinja templates.
///
/// The parser takes a sequence of tokens from the lexer and builds an abstract syntax tree
/// that represents the structure and semantics of the template.
public struct Parser: Sendable {
    private let tokens: [Token]
    private var current: Int = 0

    private init(tokens: [Token]) {
        self.tokens = tokens
    }

    /// Parses tokens into an abstract syntax tree of nodes.
    ///
    /// - Parameter tokens: The sequence of tokens to parse
    /// - Returns: An array of parsed AST nodes representing the template
    /// - Throws: `JinjaError.parser` if the tokens contain syntax errors
    public static func parse(_ tokens: [Token]) throws -> [Node] {
        var parser = Parser(tokens: tokens)
        var nodes: [Node] = []
        nodes.reserveCapacity(tokens.count / 3) // Rough estimate

        while !parser.isAtEnd {
            let node = try parser.parseNode()

            // Text node coalescing: merge adjacent text nodes
            if case let .text(newText) = node,
               case let .text(existingText)? = nodes.last
            {
                // Replace the last node with merged text
                nodes[nodes.count - 1] = .text(existingText + newText)
            } else if case let .text(text) = node, text.isEmpty {
                // do not add empty text nodes
            } else {
                nodes.append(node)
            }
        }

        return nodes
    }

    // MARK: - Node Parsing

    private mutating func parseNode() throws -> Node {
        let token = peek()

        switch token.kind {
        case .text:
            advance()
            return .text(String(token.value))

        case .comment:
            advance()
            return .comment(String(token.value))

        case .openExpression:
            try consume(.openExpression, message: "Expected '{{'.")
            let expression = try parseExpression()
            try consume(.closeExpression, message: "Expected '}}' after expression.")
            return .expression(expression)

        case .openStatement:
            try consume(.openStatement, message: "Expected '{%'.")
            let statement = try parseStatement()
            return .statement(statement)

        default:
            if isAtEnd { return .text("") }
            throw JinjaError.parser("Unexpected token type: \(token.kind)")
        }
    }

    /// Parse nodes until we hit one of the specified statement keywords
    private mutating func parseNodesUntil(_ kinds: Set<Token.Kind>) throws -> [Node] {
        var nodes: [Node] = []

        while !isAtEnd {
            if check(.openStatement) {
                let nextToken = tokens[current + 1]
                if kinds.contains(nextToken.kind) {
                    break
                }
            }

            let node = try parseNode()

            if case let .text(newText) = node,
               case let .text(existingText)? = nodes.last
            {
                nodes[nodes.count - 1] = .text(existingText + newText)
            } else {
                nodes.append(node)
            }
        }

        return nodes
    }

    // MARK: - Statement Parsing

    private mutating func parseStatement() throws -> Statement {
        let keywordToken = peek()

        switch keywordToken.kind {
        case .if:
            advance()
            return try parseIfStatement()
        case .for:
            advance()
            return try parseForStatement()
        case .set:
            advance()
            return try parseSetStatement()
        case .macro:
            advance()
            return try parseMacroStatement()
        case .break:
            advance()
            try consume(.closeStatement, message: "Expected '%}' after break.")
            return .break
        case .continue:
            advance()
            try consume(.closeStatement, message: "Expected '%}' after continue.")
            return .continue
        case .call:
            advance()
            return try parseCallStatement()
        case .filter:
            advance()
            return try parseFilterStatement()
        default:
            throw JinjaError.parser("Unknown statement: \(keywordToken.value)")
        }
    }

    private mutating func parseIfStatement() throws -> Statement {
        let condition = try parseExpression()
        try consume(.closeStatement, message: "Expected '%}' after if condition.")
        let body = try parseNodesUntil([.elif, .else, .endif])

        var alternate: [Node] = []

        if match(.openStatement) {
            if peek().kind == .elif {
                advance()
                try alternate.append(.statement(parseIfStatement()))
            } else if peek().kind == .else {
                advance()
                try consume(.closeStatement, message: "Expected '%}' after else.")
                alternate = try parseNodesUntil([.endif])
                try consume(.openStatement, message: "Expected '{%' for endif.")
                try consume(.endif, message: "Expected 'endif'.")
                try consume(.closeStatement, message: "Expected '%}' after endif.")
            } else {
                try consume(.endif, message: "Expected 'endif'.")
                try consume(.closeStatement, message: "Expected '%}' after endif.")
            }
        } else {
            throw JinjaError.parser("Unclosed if statement")
        }

        return .if(condition, body, alternate)
    }

    private mutating func parseForStatement() throws -> Statement {
        var loopVarParts: [String] = []
        repeat {
            try loopVarParts.append(consumeIdentifier())
        } while match(.comma)

        let loopVar: Statement.LoopVar = if loopVarParts.count == 1 {
            .single(loopVarParts[0])
        } else {
            .tuple(loopVarParts)
        }

        try consume(.in, message: "Expected 'in' in for loop.")

        let iterableExpr = try parseFilter()
        var testExpr: Expression?

        if match(.if) {
            testExpr = try parseExpression()
        }

        try consume(.closeStatement, message: "Expected '%}' after for loop.")

        let body = try parseNodesUntil([.else, .endfor])
        var elseBody: [Node] = []

        if match(.openStatement) {
            if match(.else) {
                try consume(.closeStatement, message: "Expected '%}' after else.")
                elseBody = try parseNodesUntil([.endfor])
                try consume(.openStatement, message: "Expected '{%' for endfor.")
                try consume(.endfor, message: "Expected 'endfor'.")
                try consume(.closeStatement, message: "Expected '%}' after endfor.")
            } else {
                try consume(.endfor, message: "Expected 'endfor'.")
                try consume(.closeStatement, message: "Expected '%}' after endfor.")
            }
        } else {
            throw JinjaError.parser("Unclosed for loop")
        }

        return .for(loopVar, iterableExpr, body, elseBody, test: testExpr)
    }

    private mutating func parseSetStatement() throws -> Statement {
        let target = try parseExpressionSequence()

        if match(.equals) {
            let value = try parseExpressionSequence()
            try consume(.closeStatement, message: "Expected '%}' after set statement.")
            return .set(target: target, value: value, body: [])
        } else {
            try consume(.closeStatement, message: "Expected '%}' after set statement.")
            let body = try parseNodesUntil([.endset])
            try consume(.openStatement, message: "Expected '{%' for endset.")
            try consume(.endset, message: "Expected 'endset'.")
            try consume(.closeStatement, message: "Expected '%}' after endset.")
            return .set(target: target, value: nil, body: body)
        }
    }

    private mutating func parseMacroStatement() throws -> Statement {
        let macroName = try consumeIdentifier()
        var parameters: [String] = []
        var defaults: OrderedDictionary<String, Expression> = [:]

        try consume(.openParen, message: "Expected '(' after macro name.")
        if !check(.closeParen) {
            repeat {
                let paramName = try consumeIdentifier()
                parameters.append(paramName)
                if match(.equals) {
                    defaults[paramName] = try parseExpression()
                }
            } while match(.comma)
        }
        try consume(.closeParen, message: "Expected ')' after macro parameters.")
        try consume(.closeStatement, message: "Expected '%}' after macro definition.")

        let body = try parseNodesUntil([.endmacro])

        try consume(.openStatement, message: "Expected '{%' for endmacro.")
        try consume(.endmacro, message: "Expected 'endmacro'.")
        try consume(.closeStatement, message: "Expected '%}' after endmacro.")

        return .macro(macroName, parameters, defaults, body)
    }

    private mutating func parseCallStatement() throws -> Statement {
        var callerArgs: [Expression]?
        if match(.openParen) {
            let (args, _) = try parseArguments()
            callerArgs = args
            try consume(.closeParen, message: "Expected ')' after caller arguments.")
        }

        let callable = try parseExpression()

        try consume(.closeStatement, message: "Expected '%}' after call statement.")

        let body = try parseNodesUntil([.endcall])

        try consume(.openStatement, message: "Expected '{%' for endcall.")
        try consume(.endcall, message: "Expected 'endcall'.")
        try consume(.closeStatement, message: "Expected '%}' after endcall.")

        return .call(callable: callable, callerArgs: callerArgs, body: body)
    }

    private mutating func parseFilterStatement() throws -> Statement {
        let filterExpr = try parseExpression()

        try consume(.closeStatement, message: "Expected '%}' after filter statement.")

        let body = try parseNodesUntil([.endfilter])

        try consume(.openStatement, message: "Expected '{%' for endfilter.")
        try consume(.endfilter, message: "Expected 'endfilter'.")
        try consume(.closeStatement, message: "Expected '%}' after endfilter.")

        return .filter(filterExpr: filterExpr, body: body)
    }

    // MARK: - Expression Parsing

    private mutating func parseExpression() throws -> Expression {
        try parseTernary()
    }

    private mutating func parseExpressionSequence() throws -> Expression {
        var expressions: [Expression] = try [parseExpression()]
        while match(.comma) {
            try expressions.append(parseExpression())
        }
        return expressions.count == 1 ? expressions[0] : .tuple(expressions)
    }

    private mutating func parseTernary() throws -> Expression {
        let expr = try parseOr()

        if match(.if) {
            let test = try parseOr()
            var alternate: Expression?

            if match(.else) {
                alternate = try parseExpression()
            }

            return .ternary(expr, test: test, alternate: alternate)
        }

        return expr
    }

    private mutating func parseFilter() throws -> Expression {
        var expr = try parseFactor()
        while match(.pipe) {
            let filterName = try consumeIdentifier()
            var args: [Expression] = []
            var kwargs: [String: Expression] = [:]

            if match(.openParen) {
                (args, kwargs) = try parseArguments()
                try consume(.closeParen, message: "Expected ')' after filter arguments.")
            }

            expr = .filter(expr, filterName, args, kwargs)
        }
        return expr
    }

    private mutating func parseOr() throws -> Expression {
        var expr = try parseAnd()
        while match(.or) {
            let right = try parseAnd()
            expr = .binary(.or, expr, right)
        }
        return expr
    }

    private mutating func parseAnd() throws -> Expression {
        var expr = try parseNot()
        while match(.and) {
            let right = try parseNot()
            expr = .binary(.and, expr, right)
        }
        return expr
    }

    private mutating func parseNot() throws -> Expression {
        if match(.not) {
            let expr = try parseNot()
            return .unary(.not, expr)
        }
        return try parseComparison()
    }

    private mutating func parseComparison() throws -> Expression {
        var expr = try parseTerm()
        while true {
            if match(.equal) {
                let right = try parseTerm()
                expr = .binary(.equal, expr, right)
            } else if match(.notEqual) {
                let right = try parseTerm()
                expr = .binary(.notEqual, expr, right)
            } else if match(.less) {
                let right = try parseTerm()
                expr = .binary(.less, expr, right)
            } else if match(.lessEqual) {
                let right = try parseTerm()
                expr = .binary(.lessEqual, expr, right)
            } else if match(.greater) {
                let right = try parseTerm()
                expr = .binary(.greater, expr, right)
            } else if match(.greaterEqual) {
                let right = try parseTerm()
                expr = .binary(.greaterEqual, expr, right)
            } else if match(.in) {
                let right = try parseTerm()
                expr = .binary(.in, expr, right)
            } else if match(.not), match(.in) {
                let right = try parseTerm()
                expr = .binary(.notIn, expr, right)
            } else if match(.is) {
                let negated = match(.not)
                let testName: String
                let token = peek()
                if token.kind == .identifier {
                    testName = try consumeIdentifier()
                } else if token.kind == .boolean || token.kind == .null {
                    testName = String(token.value)
                    advance()
                } else {
                    throw JinjaError.parser("Expected test name but found \(token.kind).")
                }
                var args: [Expression] = []
                if match(.openParen) {
                    (args, _) = try parseArguments()
                    try consume(.closeParen, message: "Expected ')' after test arguments.")
                }
                expr = .test(expr, testName, args, negated: negated)
            } else {
                break
            }
        }
        return expr
    }

    private mutating func parseTerm() throws -> Expression {
        var expr = try parseConcat()
        while true {
            if match(.plus) {
                let right = try parseConcat()
                expr = .binary(.add, expr, right)
            } else if match(.minus) {
                let right = try parseConcat()
                expr = .binary(.subtract, expr, right)
            } else {
                break
            }
        }
        return expr
    }

    private mutating func parseConcat() throws -> Expression {
        var expr = try parseFilter()
        while true {
            if match(.concat) {
                let right = try parseFilter()
                expr = .binary(.concat, expr, right)
            } else if check(anyOf: Token.Kind.primaryExpressionStarters) {
                // Implicit string concatenation - if we see another primary expression, concatenate it
                let right = try parseFilter()
                expr = .binary(.concat, expr, right)
            } else {
                break
            }
        }
        return expr
    }

    private mutating func parseFactor() throws -> Expression {
        var expr = try parseExponent()
        while true {
            if match(.multiply) {
                let right = try parseExponent()
                expr = .binary(.multiply, expr, right)
            } else if match(.divide) {
                let right = try parseExponent()
                expr = .binary(.divide, expr, right)
            } else if match(.floorDivide) {
                let right = try parseExponent()
                expr = .binary(.floorDivide, expr, right)
            } else if match(.modulo) {
                let right = try parseExponent()
                expr = .binary(.modulo, expr, right)
            } else {
                break
            }
        }
        return expr
    }

    private mutating func parseExponent() throws -> Expression {
        var expr = try parseUnary()
        while match(.power) {
            let right = try parseUnary()
            expr = .binary(.power, expr, right)
        }
        return expr
    }

    private mutating func parseUnary() throws -> Expression {
        if match(.minus) {
            let expr = try parseUnary()
            return .unary(.minus, expr)
        }
        if match(.plus) {
            let expr = try parseUnary()
            return .unary(.plus, expr)
        }
        return try parseAccess()
    }

    private mutating func parseAccess() throws -> Expression {
        var expr = try parsePrimary()

        while true {
            if match(.dot) {
                let name = try consumeIdentifier()
                expr = .member(expr, .identifier(name), computed: false)
            } else if match(.openBracket) {
                // Slice or subscript
                let start = try? parseExpression()
                if match(.colon) {
                    let stop = try? parseExpression()
                    let step = match(.colon) ? try parseExpression() : nil
                    expr = .slice(expr, start: start, stop: stop, step: step)
                } else if let index = start {
                    expr = .member(expr, index, computed: true)
                } else {
                    throw JinjaError.parser("Invalid subscript")
                }
                try consume(.closeBracket, message: "Expected ']' after index.")
            } else if match(.openParen) {
                let (args, kwargs) = try parseArguments()
                try consume(.closeParen, message: "Expected ')' after arguments.")
                expr = .call(expr, args, kwargs)
            } else {
                break
            }
        }

        return expr
    }

    private mutating func parsePrimary() throws -> Expression {
        let token = peek()
        switch token.kind {
        case .string:
            advance()
            return .string(String(token.value))
        case .number:
            advance()
            if token.value.contains(".") {
                return .number(Double(String(token.value)) ?? 0.0)
            } else {
                return .integer(Int(String(token.value)) ?? 0)
            }
        case .boolean:
            advance()
            return .boolean(token.value == "true" || token.value == "True")
        case .null:
            advance()
            return .null
        case .openBracket:
            advance()
            var elements: [Expression] = []
            if !check(.closeBracket) {
                repeat {
                    try elements.append(parseExpression())
                    if !check(.closeBracket), !match(.comma) {
                        break
                    }
                } while !check(.closeBracket)
            }
            try consume(.closeBracket, message: "Expected ']' after array literal.")
            return .array(elements)
        case .openBrace:
            advance()
            var pairs: OrderedDictionary<String, Expression> = [:]
            if !check(.closeBrace) {
                repeat {
                    let keyToken: Token
                    if check(.string) {
                        keyToken = try consume(
                            .string, message: "Expected string literal for object key."
                        )
                    } else if check(.identifier) {
                        keyToken = try consume(
                            .identifier, message: "Expected identifier for object key."
                        )
                    } else {
                        throw JinjaError.parser(
                            "Expected string literal or identifier for object key. Got \(peek().kind) instead"
                        )
                    }
                    try consume(.colon, message: "Expected ':' after object key.")
                    let value = try parseExpression()
                    pairs[String(keyToken.value)] = value
                    if !check(.closeBrace), !match(.comma) {
                        break
                    }
                } while !check(.closeBrace)
            }
            try consume(.closeBrace, message: "Expected '}' after object literal.")
            return .object(pairs)
        case .openParen:
            advance()
            // Handle empty tuple
            if check(.closeParen) {
                advance()
                return .tuple([])
            }

            let firstExpr = try parseExpression()

            // Check if this is a tuple (has comma) or just a parenthesized expression
            if match(.comma) {
                var expressions = [firstExpr]

                // Handle trailing comma case like (a,)
                if !check(.closeParen) {
                    repeat {
                        try expressions.append(parseExpression())
                    } while match(.comma) && !check(.closeParen)
                }

                try consume(.closeParen, message: "Expected ')' after tuple.")
                return .tuple(expressions)
            } else {
                try consume(.closeParen, message: "Expected ')' after expression.")
                return firstExpr
            }
        case .identifier:
            advance()
            return .identifier(String(token.value))
        // Handle context-specific keywords that can be used as identifiers in expressions
        case .if, .for, .in, .and, .or, .not, .is, .else, .set, .break, .continue:
            advance()
            return .identifier(String(token.value))
        case .multiply:
            // Handle unpacking operator *
            advance()
            let expr = try parsePrimary()
            return .unary(.splat, expr)
        default:
            throw JinjaError.parser("Unexpected token for primary expression: \(token.kind)")
        }
    }

    private mutating func parseArguments() throws -> ([Expression], [String: Expression]) {
        var args: [Expression] = []
        var kwargs: [String: Expression] = [:]

        if !check(.closeParen) {
            repeat {
                // Look ahead for keyword argument
                let lookahead = peek()
                if lookahead.kind == .identifier,
                   current + 1 < tokens.count,
                   tokens[current + 1].kind == .equals
                {
                    advance() // consume identifier
                    advance() // consume equals
                    let value = try parseExpression()
                    kwargs[String(lookahead.value)] = value
                } else {
                    let arg = try parseExpression()
                    args.append(arg)
                }
            } while match(.comma)
        }

        return (args, kwargs)
    }

    // MARK: - Helpers

    @inline(__always)
    private var isAtEnd: Bool {
        current >= tokens.count || peek().kind == .eof
    }

    @inline(__always)
    private func peek() -> Token {
        guard current < tokens.count else {
            return Token(kind: .eof, value: "", position: tokens.last?.position ?? 0)
        }
        return tokens[current]
    }

    @discardableResult
    @inline(__always)
    private mutating func advance() -> Token {
        if !isAtEnd { current += 1 }
        return previous()
    }

    @inline(__always)
    private func previous() -> Token {
        guard current > 0 else {
            return Token(kind: .eof, value: "", position: 0)
        }
        return tokens[current - 1]
    }

    @inline(__always)
    private func check(_ kind: Token.Kind) -> Bool {
        isAtEnd ? false : peek().kind == kind
    }

    @inline(__always)
    private func check(anyOf kinds: Set<Token.Kind>) -> Bool {
        isAtEnd ? false : kinds.contains(peek().kind)
    }

    @inline(__always)
    private mutating func match(_ kind: Token.Kind) -> Bool {
        if check(kind) {
            advance()
            return true
        }
        return false
    }

    @discardableResult
    private mutating func consume(_ kind: Token.Kind, message: String) throws -> Token {
        if check(kind) { return advance() }
        throw JinjaError.parser("\(message). Got \(peek().kind) instead")
    }

    @discardableResult
    private mutating func consumeIdentifier(_ name: String? = nil) throws -> String {
        let token = peek()

        if Token.Kind.allowedAsIdentifier.contains(token.kind) {
            if let name, token.value != name {
                throw JinjaError.parser("Expected identifier '\(name)' but found '\(token.value)'.")
            }
            advance()
            return String(token.value)
        }
        throw JinjaError.parser("Expected identifier but found \(token.kind).")
    }
}

// MARK: -

private extension Token.Kind {
    static let allowedAsIdentifier: Set<Token.Kind> = [
        .identifier, .if, .for, .in, .and, .or, .not, .is, .else, .set, .break, .continue,
    ]

    static let primaryExpressionStarters: Set<Token.Kind> = [
        .string, .identifier, .number, .boolean, .openParen, .openBracket, .openBrace,
    ]
}
