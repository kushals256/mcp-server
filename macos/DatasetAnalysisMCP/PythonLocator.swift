import Foundation

enum PythonLocator {
    static func candidatePaths() -> [String] {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return [
            "\(home)/.local/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
            "/usr/bin/python3",
        ]
    }

    static func augmentedEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let prefix = candidatePaths()
            .map { ($0 as NSString).deletingLastPathComponent }
            .filter { !$0.isEmpty }
            .joined(separator: ":")
        env["PATH"] = prefix + ":" + (env["PATH"] ?? "")
        return env
    }

    static func resolve() throws -> String {
        for candidate in candidatePaths() {
            guard FileManager.default.isExecutableFile(atPath: candidate) else { continue }
            if try isSupportedVersion(at: candidate) {
                return candidate
            }
        }
        throw SetupInstallerError.pythonMissing
    }

    static func resolveResult() -> PythonCheckResult {
        var foundOld: String?
        for candidate in candidatePaths() {
            guard FileManager.default.isExecutableFile(atPath: candidate) else { continue }
            switch versionResult(at: candidate) {
            case .ok:
                return .ok(path: candidate)
            case .tooOld(let version):
                foundOld = foundOld ?? version
            case .missing:
                continue
            }
        }
        if let foundOld {
            return .tooOld(foundOld)
        }
        return .missing
    }

    private enum VersionResult {
        case ok
        case tooOld(String)
        case missing
    }

    private static func versionResult(at path: String) -> VersionResult {
        do {
            let output = try runCommand(executable: path, arguments: ["--version"])
            guard let versionToken = output.split(separator: " ").last else {
                return .missing
            }
            let parts = versionToken.split(separator: ".").compactMap { Int($0) }
            guard parts.count >= 2 else { return .missing }
            if parts[0] < 3 || (parts[0] == 3 && parts[1] < 10) {
                return .tooOld(String(versionToken))
            }
            return .ok
        } catch {
            return .missing
        }
    }

    private static func isSupportedVersion(at path: String) throws -> Bool {
        switch versionResult(at: path) {
        case .ok:
            return true
        case .tooOld:
            return false
        case .missing:
            return false
        }
    }

    @discardableResult
    private static func runCommand(executable: String, arguments: [String]) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.environment = augmentedEnvironment()

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""

        guard process.terminationStatus == 0 else {
            throw SetupInstallerError.commandFailed(output)
        }

        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
