import Foundation

enum PipBootstrapper {
    static func ensurePip(python: String, onLog: @escaping (String) -> Void) throws {
        if pipAvailable(python: python) {
            let version = try runCommand(
                executable: python,
                arguments: ["-m", "pip", "--version"],
                onLog: onLog
            )
            onLog(version)
            return
        }

        onLog("pip not found — bootstrapping with ensurepip...")
        let output = try runCommand(
            executable: python,
            arguments: ["-m", "ensurepip", "--upgrade"],
            onLog: onLog
        )
        if !output.isEmpty {
            onLog(output)
        }

        guard pipAvailable(python: python) else {
            throw SetupInstallerError.commandFailed(
                "Could not install pip. Run in Terminal:\n\(python) -m ensurepip --upgrade"
            )
        }
    }

    private static func pipAvailable(python: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = ["-m", "pip", "--version"]
        process.environment = PythonLocator.augmentedEnvironment()
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }

    @discardableResult
    private static func runCommand(
        executable: String,
        arguments: [String],
        onLog: ((String) -> Void)? = nil
    ) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.environment = PythonLocator.augmentedEnvironment()

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        if !output.isEmpty {
            onLog?(output.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        guard process.terminationStatus == 0 else {
            let detail = output.isEmpty
                ? "Command failed: \(executable) \(arguments.joined(separator: " "))"
                : output
            throw SetupInstallerError.commandFailed(detail)
        }

        return output
    }
}
