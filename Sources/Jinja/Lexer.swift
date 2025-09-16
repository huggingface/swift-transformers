/// Tokenizes Jinja template source code into a sequence of tokens.
public enum Lexer: Sendable {
    private static let keywords: [String: Token.Kind] = [
        "if": .if, "else": .else, "elif": .elif, "endif": .endif,
        "for": .for, "endfor": .endfor, "in": .in, "not": .not,
        "and": .and, "or": .or, "is": .is, "set": .set, "endset": .endset,
        "macro": .macro, "endmacro": .endmacro,
        "true": .boolean, "false": .boolean,
        "null": .null, "none": .null,
        "break": .break, "continue": .continue,
        "call": .call, "endcall": .endcall,
        "filter": .filter, "endfilter": .endfilter,

        // Python-compatible keywords
        "True": .boolean, "False": .boolean, "None": .null,
    ]

    private static let operators: [String: Token.Kind] = [
        "+": .plus, "-": .minus, "*": .multiply, "/": .divide,
        "//": .floorDivide, "**": .power,
        "%": .modulo, "~": .concat,
        "==": .equal, "!=": .notEqual, "<": .less, "<=": .lessEqual,
        ">": .greater, ">=": .greaterEqual, "=": .equals, "|": .pipe,
    ]

    /// Tokenizes a template source string into an array of tokens.
    ///
    /// - Parameter source: The Jinja template source code to tokenize
    /// - Returns: An array of tokens representing the lexical structure
    /// - Throws: `JinjaError.lexer` if the source contains invalid syntax
    public static func tokenize(_ source: String) throws
        -> [Token]
    {
        let preprocessed = preprocess(source)
        var tokens: [Token] = []
        tokens.reserveCapacity(preprocessed.count / 4)

        var position = preprocessed.startIndex
        var inTag = false
        var curlyBracketDepth = 0

        while position < preprocessed.endIndex {
            if inTag {
                position = skipWhitespace(in: preprocessed, at: position)
                if position >= preprocessed.endIndex {
                    break
                }
            }

            let (token, newPosition) = try extractToken(
                from: preprocessed, at: position, inTag: inTag, curlyBracketDepth: curlyBracketDepth
            )

            switch token.kind {
            case .openExpression, .openStatement:
                inTag = true
                curlyBracketDepth = 0
            case .closeExpression, .closeStatement:
                inTag = false
            case .openBrace:
                curlyBracketDepth += 1
            case .closeBrace:
                curlyBracketDepth -= 1
            default:
                break
            }

            if token.kind == .text, token.value.isEmpty {
                position = newPosition
                continue
            }

            tokens.append(token)
            position = newPosition

            if token.kind == .eof {
                break
            }
        }

        if tokens.isEmpty || tokens.last?.kind != .eof {
            let charPosition = preprocessed.distance(from: preprocessed.startIndex, to: position)
            tokens.append(Token(kind: .eof, value: "", position: charPosition))
        }

        return tokens
    }

    private static func skipWhitespace(
        in source: String, at position: String.Index
    ) -> String.Index {
        var pos = position
        while pos < source.endIndex {
            let char = source[pos]
            if char.isWhitespace {
                pos = source.index(after: pos)
            } else {
                break
            }
        }
        return pos
    }

    private static func preprocess(_ template: String) -> String {
        // Optimized preprocessing with single pass
        var result = template

        // Handle whitespace control
        result = result.replacing(#/-%}\s*/#, with: "%}")
        result = result.replacing(#/\s*{%-/#, with: "{%")
        result = result.replacing(#/-}}\s*/#, with: "}}")
        result = result.replacing(#/\s*{{-/#, with: "{{")

        return result
    }

    private static func extractToken(
        from source: String, at position: String.Index, inTag: Bool,
        curlyBracketDepth: Int = 0
    ) throws -> (Token, String.Index) {
        guard position < source.endIndex else {
            let charPosition = source.distance(from: source.startIndex, to: position)
            return (Token(kind: .eof, value: "", position: charPosition), position)
        }

        let char = source[position]
        let charPosition = source.distance(from: source.startIndex, to: position)

        // Template delimiters - check for {{, {%, and {#
        if char == "{" {
            let nextIndex = source.index(after: position)
            if nextIndex < source.endIndex {
                let nextChar = source[nextIndex]
                if nextChar == "{" { // "{{"
                    let endIndex = source.index(after: nextIndex)
                    return (
                        Token(
                            kind: .openExpression, value: source[position..<endIndex],
                            position: charPosition
                        ), endIndex
                    )
                } else if nextChar == "%" { // "{%"
                    let endIndex = source.index(after: nextIndex)
                    return (
                        Token(
                            kind: .openStatement, value: source[position..<endIndex],
                            position: charPosition
                        ), endIndex
                    )
                } else if nextChar == "#" { // "{#"
                    return try extractCommentToken(from: source, at: position)
                }
            }
        }

        // Check for closing delimiters
        if char == "}" {
            let nextIndex = source.index(after: position)
            if nextIndex < source.endIndex, source[nextIndex] == "}", curlyBracketDepth == 0 {
                let endIndex = source.index(after: nextIndex)
                return (
                    Token(
                        kind: .closeExpression, value: source[position..<endIndex],
                        position: charPosition
                    ), endIndex
                )
            }
        }
        if char == "%" {
            let nextIndex = source.index(after: position)
            if nextIndex < source.endIndex, source[nextIndex] == "}" {
                let endIndex = source.index(after: nextIndex)
                return (
                    Token(
                        kind: .closeStatement, value: source[position..<endIndex],
                        position: charPosition
                    ), endIndex
                )
            }
        }

        if !inTag {
            return extractTextToken(from: source, at: position)
        }

        // Single character tokens
        let nextIndex = source.index(after: position)
        switch char {
        case "(":
            return (
                Token(
                    kind: .openParen, value: source[position..<nextIndex], position: charPosition
                ),
                nextIndex
            )
        case ")":
            return (
                Token(
                    kind: .closeParen, value: source[position..<nextIndex], position: charPosition
                ),
                nextIndex
            )
        case "[":
            return (
                Token(
                    kind: .openBracket, value: source[position..<nextIndex], position: charPosition
                ),
                nextIndex
            )
        case "]":
            return (
                Token(
                    kind: .closeBracket, value: source[position..<nextIndex], position: charPosition
                ), nextIndex
            )
        case "{":
            return (
                Token(
                    kind: .openBrace, value: source[position..<nextIndex], position: charPosition
                ),
                nextIndex
            )
        case "}":
            return (
                Token(
                    kind: .closeBrace, value: source[position..<nextIndex], position: charPosition
                ),
                nextIndex
            )
        case ",":
            return (
                Token(kind: .comma, value: source[position..<nextIndex], position: charPosition),
                nextIndex
            )
        case ".":
            return (
                Token(kind: .dot, value: source[position..<nextIndex], position: charPosition),
                nextIndex
            )
        case ":":
            return (
                Token(kind: .colon, value: source[position..<nextIndex], position: charPosition),
                nextIndex
            )
        case "|":
            return (
                Token(kind: .pipe, value: source[position..<nextIndex], position: charPosition),
                nextIndex
            )
        default: break
        }

        // Multi-character operators
        for length in [2, 1] {
            var endIndex = position
            for _ in 0..<length {
                guard endIndex < source.endIndex else { break }
                endIndex = source.index(after: endIndex)
            }
            if endIndex <= source.endIndex {
                let op = String(source[position..<endIndex])
                if let tokenKind = operators[op] {
                    return (
                        Token(
                            kind: tokenKind, value: source[position..<endIndex],
                            position: charPosition
                        ), endIndex
                    )
                }
            }
        }

        // String literals
        if char == "'" || char == "\"" {
            return try extractStringToken(from: source, at: position, delimiter: char)
        }

        // Numbers
        if char.isNumber {
            return extractNumberToken(from: source, at: position)
        }

        // Identifiers and keywords
        if char.isLetter || char == "_" {
            return extractIdentifierToken(from: source, at: position)
        }

        throw JinjaError.lexer(
            "Unexpected character '\(char)' at position \(charPosition)"
        )
    }

    private static func extractTextToken(
        from source: String, at position: String.Index
    ) -> (Token, String.Index) {
        var pos = position
        let startPos = position

        while pos < source.endIndex {
            let char = source[pos]
            let nextIndex = source.index(after: pos)

            if nextIndex <= source.endIndex {
                if char == "{", nextIndex < source.endIndex {
                    let nextChar = source[nextIndex]
                    if nextChar == "{" || nextChar == "%" || nextChar == "#" {
                        break
                    }
                }
                if char == "}", nextIndex < source.endIndex, source[nextIndex] == "}" {
                    break
                }
                if char == "%", nextIndex < source.endIndex, source[nextIndex] == "}" {
                    break
                }
                if char == "#", nextIndex < source.endIndex, source[nextIndex] == "}" {
                    break
                }
            }
            pos = nextIndex
        }

        let charPosition = source.distance(from: source.startIndex, to: position)
        return (Token(kind: .text, value: source[startPos..<pos], position: charPosition), pos)
    }

    private static func extractStringToken(
        from source: String, at position: String.Index, delimiter: Character
    ) throws -> (Token, String.Index) {
        var pos = source.index(after: position)
        var value = ""
        let charPosition = source.distance(from: source.startIndex, to: position)

        while pos < source.endIndex {
            let char = source[pos]

            if char == delimiter {
                let nextPos = source.index(after: pos)
                return (Token(kind: .string, value: value, position: charPosition), nextPos)
            }

            if char == "\\" {
                pos = source.index(after: pos)
                if pos < source.endIndex {
                    let escaped = source[pos]
                    switch escaped {
                    case "n": value += "\n"
                    case "t": value += "\t"
                    case "r": value += "\r"
                    case "b": value += "\u{8}" // backspace
                    case "f": value += "\u{C}" // form feed
                    case "v": value += "\u{B}" // vertical tab
                    case "\\": value += "\\"
                    case "\"": value += "\""
                    case "'": value += "'"
                    default:
                        value += String(escaped)
                    }
                }
            } else {
                value += String(char)
            }

            pos = source.index(after: pos)
        }

        throw JinjaError.lexer("Unclosed string at position \(charPosition)")
    }

    private static func extractNumberToken(
        from source: String, at position: String.Index
    ) -> (Token, String.Index) {
        var pos = position
        var hasDot = false
        let startPos = position

        while pos < source.endIndex {
            let char = source[pos]
            if char.isNumber {
                pos = source.index(after: pos)
            } else if char == ".", !hasDot {
                hasDot = true
                pos = source.index(after: pos)
            } else {
                break
            }
        }

        let charPosition = source.distance(from: source.startIndex, to: position)
        return (Token(kind: .number, value: source[startPos..<pos], position: charPosition), pos)
    }

    private static func extractIdentifierToken(
        from source: String, at position: String.Index
    ) -> (Token, String.Index) {
        var pos = position
        let startPos = position

        while pos < source.endIndex {
            let char = source[pos]
            if char.isLetter || char.isNumber || char == "_" {
                pos = source.index(after: pos)
            } else {
                break
            }
        }

        let value = String(source[startPos..<pos])
        let tokenKind = keywords[value] ?? .identifier
        let charPosition = source.distance(from: source.startIndex, to: position)
        return (Token(kind: tokenKind, value: source[startPos..<pos], position: charPosition), pos)
    }

    private static func extractCommentToken(
        from source: String, at position: String.Index
    ) throws -> (Token, String.Index) {
        // Skip the opening {#
        var pos = source.index(position, offsetBy: 2)
        var value = ""
        let charPosition = source.distance(from: source.startIndex, to: position)

        while pos < source.endIndex {
            let char = source[pos]
            let nextIndex = source.index(after: pos)

            if nextIndex < source.endIndex, char == "#", source[nextIndex] == "}" {
                let endPos = source.index(after: nextIndex)
                return (Token(kind: .comment, value: value, position: charPosition), endPos)
            }

            value += String(char)
            pos = nextIndex
        }

        throw JinjaError.lexer("Unclosed comment at position \(charPosition)")
    }
}
