import AppKit
import Foundation

enum PythonCheckResult: Equatable {
    case ok(path: String)
    case missing
    case tooOld(String)
}

struct PrerequisitesStatus: Equatable {
    let installedInApplications: Bool
    let python: PythonCheckResult
    let claudeInstalled: Bool

    var readyForSetup: Bool {
        guard installedInApplications, claudeInstalled else { return false }
        if case .ok = python { return true }
        return false
    }

    var canAttemptSetup: Bool {
        if case .ok = python { return true }
        return false
    }
}

final class PrerequisitesChecker {
    private let claudeAppPath = "/Applications/Claude.app"
    private let pythonDownloadURL = "https://www.python.org/downloads/"
    private let claudeDownloadURL = "https://claude.ai/download"

    func evaluate() -> PrerequisitesStatus {
        PrerequisitesStatus(
            installedInApplications: isInstalledInApplications(),
            python: checkPython(),
            claudeInstalled: FileManager.default.fileExists(atPath: claudeAppPath)
        )
    }

    func isInstalledInApplications() -> Bool {
        Bundle.main.bundlePath.hasPrefix("/Applications/")
    }

    func checkPython() -> PythonCheckResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "--version"]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return .missing
        }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        guard process.terminationStatus == 0,
              let versionToken = output.split(separator: " ").last else {
            return .missing
        }

        let parts = versionToken.split(separator: ".").compactMap { Int($0) }
        guard parts.count >= 2 else { return .missing }
        if parts[0] < 3 || (parts[0] == 3 && parts[1] < 10) {
            return .tooOld(String(versionToken))
        }

        let which = Process()
        which.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        which.arguments = ["python3", "-c", "import sys; print(sys.executable)"]
        let whichPipe = Pipe()
        which.standardOutput = whichPipe
        which.standardError = whichPipe

        do {
            try which.run()
            which.waitUntilExit()
        } catch {
            return .missing
        }

        let whichData = whichPipe.fileHandleForReading.readDataToEndOfFile()
        let path = String(data: whichData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return path.isEmpty ? .missing : .ok(path: path)
    }

    func openPythonDownload() {
        if let url = URL(string: pythonDownloadURL) {
            NSWorkspace.shared.open(url)
        }
    }

    func openClaudeDownload() {
        if let url = URL(string: claudeDownloadURL) {
            NSWorkspace.shared.open(url)
        }
    }

    func openClaudeApp() {
        NSWorkspace.shared.open(URL(fileURLWithPath: claudeAppPath))
    }
}
