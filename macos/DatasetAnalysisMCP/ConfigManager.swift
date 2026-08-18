import Foundation

final class ConfigManager {
    private let fileManager = FileManager.default

    var configURL: URL {
        let base = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Claude/claude_desktop_config.json")
        return base
    }

    var stateURL: URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".local/share/dataset-analysis-mcp/disabled")
    }

    var dataDirectoryPath: String {
        if let env = readConfig()?["env"] as? [String: Any],
           let dataDir = env["MCP_DATA_DIR"] as? String {
            return NSString(string: dataDir).expandingTildeInPath
        }
        return NSString(string: "~/datasets").expandingTildeInPath
    }

    var isDisabled: Bool {
        fileManager.fileExists(atPath: stateURL.path)
    }

    var isConfigured: Bool {
        guard let config = readFullConfig(),
              let servers = config["mcpServers"] as? [String: Any] else {
            return false
        }
        return servers["dataset-analysis"] != nil
    }

    func disableServer() {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["dataset-analysis-mcp-disable"]
        try? process.run()
        process.waitUntilExit()
    }

    func enableServer() {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["dataset-analysis-mcp-enable"]
        try? process.run()
        process.waitUntilExit()
    }

    func runDoctor() -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["dataset-analysis-mcp-doctor"]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
        } catch {
            return "Could not run doctor: \(error.localizedDescription)"
        }

        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8) ?? "No output"
    }

    func doctorPassesQuickCheck() -> Bool {
        guard isConfigured, !isDisabled else { return false }
        guard fileManager.fileExists(atPath: configURL.path) else { return false }
        return fileManager.isWritableFile(atPath: dataDirectoryPath) || createDataDirectory()
    }

    private func createDataDirectory() -> Bool {
        do {
            try fileManager.createDirectory(atPath: dataDirectoryPath, withIntermediateDirectories: true)
            return true
        } catch {
            return false
        }
    }

    private func readFullConfig() -> [String: Any]? {
        guard let data = try? Data(contentsOf: configURL) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private func readConfig() -> [String: Any]? {
        guard let full = readFullConfig(),
              let servers = full["mcpServers"] as? [String: Any],
              let entry = servers["dataset-analysis"] as? [String: Any] else {
            return nil
        }
        return entry
    }
}
