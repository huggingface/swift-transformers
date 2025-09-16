/// Resolves positional and keyword arguments, mimicking Python's argument handling.
///
/// This function takes the arguments passed to a filter and resolves them against a list of
/// parameter names and a dictionary of default values. It ensures that arguments are not
/// passed both positionally and by keyword, and that all required arguments are present.
///
/// Based on the argument passing conventions in Jinja's Python implementation.
/// See: https://github.com/pallets/jinja/blob/main/src/jinja2/filters.py
///
/// - Parameters:
///   - args: An array of positional arguments (`Value`).
///   - kwargs: A dictionary of keyword arguments (`[String: Value]`).
///   - parameters: An ordered list of parameter names for the filter.
///   - defaults: A dictionary of default values for optional parameters.
/// - Returns: A dictionary of resolved argument names and their `Value`.
/// - Throws: `JinjaError.runtime` if arguments are invalid (e.g., duplicate, unexpected).
func resolveCallArguments(
    args: [Value],
    kwargs: [String: Value],
    parameters: [String],
    defaults: [String: Value] = [:]
) throws -> [String: Value] {
    var resolvedArgs: [String: Value] = [:]

    // Handle positional arguments
    for (i, arg) in args.enumerated() {
        if i >= parameters.count {
            break // Allow for filters with variable arguments
        }
        let paramName = parameters[i]
        if kwargs.keys.contains(paramName) {
            throw JinjaError.runtime(
                "Argument '\(paramName)' passed both positionally and as keyword.")
        }
        resolvedArgs[paramName] = arg
    }

    // Handle keyword arguments
    for (name, value) in kwargs {
        guard parameters.contains(name) else {
            throw JinjaError.runtime("Unexpected keyword argument '\(name)' for filter.")
        }
        // This check is technically redundant if the positional loop is correct,
        // but it's a good safeguard.
        if resolvedArgs[name] != nil {
            throw JinjaError.runtime("Argument '\(name)' passed both positionally and as keyword.")
        }
        resolvedArgs[name] = value
    }

    // Apply defaults and check for missing required arguments
    for paramName in parameters {
        if resolvedArgs[paramName] == nil {
            if let defaultValue = defaults[paramName] {
                resolvedArgs[paramName] = defaultValue
            } else {
                // This is a required argument that wasn't provided
                throw JinjaError.runtime("Missing required argument '\(paramName)' for filter.")
            }
        }
    }

    return resolvedArgs
}
