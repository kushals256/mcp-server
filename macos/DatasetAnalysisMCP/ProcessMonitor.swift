import Foundation

final class ProcessMonitor {
    func currentState(configManager: ConfigManager) -> ServerState {
        if configManager.isDisabled || !configManager.isConfigured {
            return .disabled
        }

        if !configManager.doctorPassesQuickCheck() {
            return .error
        }

        if isServerProcessRunning() {
            return .active
        }

        return .idle
    }

    private func isServerProcessRunning() -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-ax"]

        let pipe = Pipe()
        process.standardOutput = pipe

        do {
            try process.run()
        } catch {
            return false
        }

        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let output = String(data: data, encoding: .utf8) else {
            return false
        }

        return output.contains("dataset-analysis-mcp") || output.contains("dataset_analysis_mcp")
    }
}
