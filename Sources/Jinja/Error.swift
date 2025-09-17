/// Errors that can occur during Jinja template processing.
public enum JinjaError: Error, Sendable {
    /// Error during tokenization of template source.
    case lexer(String)
    /// Error during parsing of tokens into AST.
    case parser(String)
    /// Error during template execution or evaluation.
    case runtime(String)
    /// Error due to invalid template syntax.
    case syntax(String)
}
