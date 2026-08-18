import Foundation

enum SetupStep: Int, CaseIterable {
    case checkingPython
    case installingPackage
    case configuringClaude
    case runningDoctor

    var title: String {
        switch self {
        case .checkingPython:
            return "Checking Python"
        case .installingPackage:
            return "Installing Prism"
        case .configuringClaude:
            return "Configuring Claude Desktop"
        case .runningDoctor:
            return "Running health check"
        }
    }
}

enum SetupStepStatus {
    case pending
    case running
    case success
    case failed
}

enum SetupInstallerError: LocalizedError {
    case pythonMissing
    case pythonTooOld(String)
    case commandFailed(String)
    case doctorFailed(String)

    var errorDescription: String? {
        switch self {
        case .pythonMissing:
            return "Python 3.10+ was not found. Install from python.org or run: brew install python@3.12"
        case .pythonTooOld(let version):
            return "Python \(version) is too old. Prism requires Python 3.10 or newer."
        case .commandFailed(let detail):
            return detail
        case .doctorFailed(let output):
            return "Health check failed. Some setup steps did not complete.\n\n\(output)"
        }
    }
}
