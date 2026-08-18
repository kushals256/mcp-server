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
        PythonLocator.resolveResult()
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
