# Contributing to Swift Transformers

## Code Styling and Linting

Code formatting is enforced with `swift-format` default utility from Apple.
To install and run it on all the files in the project, use the following command:

```bash
brew install swift-format
swift-format . -i -r
```

The style is controlled by the `.swift-format` JSON file in the root of the repository.
As there is no standard for Swift formatting, even Apple's own `swift-format` tool and Xcode differ in their formatting rules, and available settings.
