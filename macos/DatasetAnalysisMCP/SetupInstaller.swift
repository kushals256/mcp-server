import Foundation

final class SetupInstaller {
    private let packageName = "dataset-analysis-mcp"
    private let gitInstallURL = "git+https://github.com/kushals256/mcp-server.git"

    func run(
        onStepChange: @escaping (SetupStep, SetupStepStatus) -> Void,
        onLog: @escaping (String) -> Void,
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                onStepChange(.checkingPython, .running)
                let python = try self.resolvePython()
                onLog("Found Python at \(python)")
                onStepChange(.checkingPython, .success)

                onStepChange(.installingPackage, .running)
                try self.installPackage(python: python, onLog: onLog)
                onStepChange(.installingPackage, .success)

                onStepChange(.configuringClaude, .running)
                try self.runSetup(python: python, onLog: onLog)
                onStepChange(.configuringClaude, .success)

                onStepChange(.runningDoctor, .running)
                let doctorOutput = try self.runDoctor(onLog: onLog)
                if doctorOutput.contains("[FAIL]") {
                    onStepChange(.runningDoctor, .failed)
                    throw SetupInstallerError.doctorFailed(doctorOutput)
                }
                onStepChange(.runningDoctor, .success)

                DispatchQueue.main.async {
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }

    private func augmentedEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let pathPrefix = "\(home)/.local/bin:/opt/homebrew/bin:/usr/local/bin"
        env["PATH"] = "\(pathPrefix):" + (env["PATH"] ?? "")
        return env
    }

    private func resolvePython() throws -> String {
        let output = try runCommand(executable: "/usr/bin/env", arguments: ["python3", "--version"])
        let versionLine = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let version = versionLine.split(separator: " ").last else {
            throw SetupInstallerError.pythonMissing
        }
        let parts = version.split(separator: ".").compactMap { Int($0) }
        guard parts.count >= 2 else {
            throw SetupInstallerError.pythonMissing
        }
        if parts[0] < 3 || (parts[0] == 3 && parts[1] < 10) {
            throw SetupInstallerError.pythonTooOld(String(version))
        }
        let which = try runCommand(executable: "/usr/bin/env", arguments: ["python3", "-c", "import sys; print(sys.executable)"])
        return which.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func installPackage(python: String, onLog: @escaping (String) -> Void) throws {
        for target in [packageName, gitInstallURL] {
            onLog("Trying pip install \(target)...")
            do {
                let output = try runCommand(
                    executable: python,
                    arguments: ["-m", "pip", "install", "--user", "--upgrade", target]
                )
                if !output.isEmpty {
                    onLog(output)
                }
                return
            } catch {
                onLog("Failed: \(error.localizedDescription)")
            }
        }
        throw SetupInstallerError.commandFailed(
            "Could not install Prism. Check your internet connection and try again."
        )
    }

    private func runSetup(python: String, onLog: @escaping (String) -> Void) throws {
        if let setupCLI = userLocalCommand(named: "dataset-analysis-mcp-setup") {
            let output = try runCommand(executable: setupCLI, arguments: ["--yes"])
            if !output.isEmpty { onLog(output) }
            return
        }

        let userBase = try runCommand(executable: python, arguments: ["-m", "site", "--user-base"])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let setupPath = "\(userBase)/bin/dataset-analysis-mcp-setup"
        if FileManager.default.isExecutableFile(atPath: setupPath) {
            let output = try runCommand(executable: setupPath, arguments: ["--yes"])
            if !output.isEmpty { onLog(output) }
            return
        }

        throw SetupInstallerError.commandFailed(
            "Could not find dataset-analysis-mcp-setup after install. Try restarting Prism."
        )
    }

    private func runDoctor(onLog: @escaping (String) -> Void) throws -> String {
        if let doctorCLI = userLocalCommand(named: "dataset-analysis-mcp-doctor") {
            let output = try runCommand(executable: doctorCLI, arguments: [])
            onLog(output)
            return output
        }

        let output = try runCommand(
            executable: "/usr/bin/env",
            arguments: ["dataset-analysis-mcp-doctor"]
        )
        onLog(output)
        return output
    }

    private func userLocalCommand(named: String) -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let candidate = "\(home)/.local/bin/\(named)"
        return FileManager.default.isExecutableFile(atPath: candidate) ? candidate : nil
    }

    @discardableResult
    private func runCommand(executable: String, arguments: [String]) throws -> String {
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
            let detail = output.isEmpty ? "Command failed: \(executable) \(arguments.joined(separator: " "))" : output
            throw SetupInstallerError.commandFailed(detail)
        }

        return output
    }
}
