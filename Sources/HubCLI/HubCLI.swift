import ArgumentParser
import Foundation

import Hub

@main
struct HubCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Hugging Face Hub Client",
        version: "0.0.1"
    )

    enum Action: String, ExpressibleByArgument {
        case download
    }
    
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
    
    @Argument(help: "Action")
    var action: Action

    @Argument(help: "Repo ID")
    var repo: String

    @Option(help: "Repo type")
    var repoType: RepoType = .model

    @Option(help: "Glob pattern for files to include")
    var include: String?
        
    func run() async throws {
        let downloadedTo = try await Hub.snapshot(from: repo, repoType: repoType.asHubApiRepoType, matching: include) { progress in
            DispatchQueue.main.async {
                let totalPercent = 100 * progress.fractionCompleted
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) \(totalPercent.formatted("%.02f"))%", terminator: "\r")
                fflush(stdout)
            }
        }
        print("Snapshot downloaded to: \(downloadedTo.path)")
    }
}

extension Double {
    func formatted(_ format: String) -> String {
        return String(format: "\(format)", self)
    }
}
