/// A lexical token produced by parsing Jinja template source code.
///
/// Tokens represent the smallest meaningful units of a Jinja template,
/// such as keywords, operators, literals, and delimiters.
public struct Token: Hashable, Sendable {
    /// The specific type of token representing different syntactic elements.
    public enum Kind: CaseIterable, Hashable, Codable, Sendable {
        /// Plain text content outside of Jinja template constructs.
        case text
        /// `{{` delimiter.
        case openExpression
        /// `}}` delimiter.
        case closeExpression
        /// `{%` delimiter.
        case openStatement
        /// `%}` delimiter.
        case closeStatement
        /// String literal value enclosed in quotes.
        case string
        /// Numeric literal value.
        case number
        /// Boolean literal (`true` or `false`).
        case boolean
        /// Null literal value.
        case null
        /// Variable or function identifier.
        case identifier
        /// Opening parenthesis `(`.
        case openParen
        /// Closing parenthesis `)`.
        case closeParen
        /// Opening square bracket `[`.
        case openBracket
        /// Closing square bracket `]`.
        case closeBracket
        /// Opening curly brace `{`.
        case openBrace
        /// Closing curly brace `}`.
        case closeBrace
        /// Comma separator `,`.
        case comma
        /// Dot accessor `.`.
        case dot
        /// Colon separator `:`.
        case colon
        /// Pipe operator `|` for filters.
        case pipe
        /// Assignment operator `=`.
        case equals
        /// Addition operator `+`.
        case plus
        /// Subtraction operator `-`.
        case minus
        /// Multiplication operator `*`.
        case multiply
        /// Division operator `/`.
        case divide
        /// Floor division operator `//`.
        case floorDivide
        /// Exponentiation operator `**`.
        case power
        /// Modulo operator `%`.
        case modulo
        /// String concatenation operator `~`.
        case concat
        /// Equality comparison operator `==`.
        case equal
        /// Inequality comparison operator `!=`.
        case notEqual
        /// Less than comparison operator `<`.
        case less
        /// Less than or equal comparison operator `<=`.
        case lessEqual
        /// Greater than comparison operator `>`.
        case greater
        /// Greater than or equal comparison operator `>=`.
        case greaterEqual
        /// Logical AND operator `and`.
        case and
        /// Logical OR operator `or`.
        case or
        /// Logical NOT operator `not`.
        case not
        /// Membership test operator `in`.
        case `in`
        /// Negative membership test operator `not in`.
        case notIn
        /// Identity test operator `is`.
        case `is`
        /// Conditional statement keyword `if`.
        case `if`
        /// Alternative conditional keyword `else`.
        case `else`
        /// Chained conditional keyword `elif`.
        case elif
        /// End of conditional block keyword `endif`.
        case endif
        /// Loop statement keyword `for`.
        case `for`
        /// End of loop block keyword `endfor`.
        case endfor
        /// Variable assignment keyword `set`.
        case set
        /// End of set block keyword `endset`.
        case endset
        /// Macro definition keyword `macro`.
        case macro
        /// End of macro block keyword `endmacro`.
        case endmacro
        /// Loop control keyword `break`.
        case `break`
        /// Loop control keyword `continue`.
        case `continue`
        /// Call block keyword `call`.
        case call
        /// End of call block keyword `endcall`.
        case endcall
        /// Filter block keyword `filter`.
        case filter
        /// End of filter block keyword `endfilter`.
        case endfilter
        /// Comment content `{# ... #}`.
        case comment
        /// End of file marker.
        case eof
    }

    /// The classification of this token.
    public let kind: Kind

    /// The raw text content of this token from the source.
    public let value: Substring

    /// The character position of this token in the original source text.
    public let position: Int

    /// Initialize a token with a string or substring value.
    public init(kind: Kind, value: some StringProtocol, position: Int) {
        self.kind = kind
        self.value = Substring(value)
        self.position = position
    }
}

// MARK: - Codable

extension Token: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind, value, position
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kind = try container.decode(Kind.self, forKey: .kind)
        let stringValue = try container.decode(String.self, forKey: .value)
        value = Substring(stringValue)
        position = try container.decode(Int.self, forKey: .position)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(kind, forKey: .kind)
        try container.encode(String(value), forKey: .value)
        try container.encode(position, forKey: .position)
    }
}
