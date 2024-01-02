import ArgumentParser
import Foundation

import Hub

let defaultTokenLocation = NSString("~/.cache/huggingface/token").expandingTildeInPath

@main
struct HubCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Hugging Face Hub Client",
        version: "0.0.1",
        subcommands: [Download.self, Whoami.self]
    )
}

protocol SubcommandWithToken {

    var token: String? { get }
}

extension SubcommandWithToken {
    var hfToken: String? {
        if let token = token { return token }
        return try? String(contentsOfFile: defaultTokenLocation, encoding: .utf8)
    }
}

struct Download: AsyncParsableCommand, SubcommandWithToken {
    static let configuration = CommandConfiguration(abstract: "Snapshot download from the Hub")

    enum RepoType: String, ExpressibleByArgument {
        case model
        case dataset
        case space
        
        var asHubApiRepoType: HubApi.RepoType {
            switch self {
            case .model: return .models
            case .dataset: return .datasets
            case .space: return .spaces
            }
        }
    }
    
    @Argument(help: "Repo ID")
    var repo: String

    @Option(help: "Repo type")
    var repoType: RepoType = .model

    @Option(help: "Glob patterns for files to include")
    var include: [String] = []
    
    @Option(help: "Hugging Face token. If empty, will attempt to read from the filesystem at \(defaultTokenLocation)")
    var token: String? = nil
        
    func run() async throws {
        let hubApi = HubApi(hfToken: hfToken)
        let repo = Hub.Repo(id: repo, type: repoType.asHubApiRepoType)
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: include) { progress in
            DispatchQueue.main.async {
                let totalPercent = 100 * progress.fractionCompleted
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) \(totalPercent.formatted("%.02f"))%", terminator: "\r")
                fflush(stdout)
            }
        }
        print("Snapshot downloaded to: \(downloadedTo.path)")
    }
}

struct Whoami: AsyncParsableCommand, SubcommandWithToken {
    static let configuration = CommandConfiguration(abstract: "whoami")
         
    @Option(help: "Hugging Face token. If empty, will attempt to read from the filesystem at \(defaultTokenLocation)")
    var token: String? = nil
    
    func run() async throws {
        let hubApi = HubApi(hfToken: hfToken)
        let userInfo = try await hubApi.whoami()
        if let name = userInfo.name?.stringValue,
           let fullname = userInfo.fullname?.stringValue,
           let email = userInfo.email?.stringValue
        {
            print("\(name) (\(fullname) <\(email)>)")
        } else {
            print("Cannot retrieve user info")
        }
    }
}

extension Double {
    func formatted(_ format: String) -> String {
        return String(format: "\(format)", self)
    }
}
