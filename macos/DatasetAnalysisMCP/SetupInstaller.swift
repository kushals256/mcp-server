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
        PythonLocator.augmentedEnvironment()
    }

    private func resolvePython() throws -> String {
        try PythonLocator.resolve()
    }

    private func installPackage(python: String, onLog: @escaping (String) -> Void) throws {
        try PipBootstrapper.ensurePip(python: python, onLog: onLog)
        onLog("First install downloads pandas, scikit-learn, and more — this can take 3–5 minutes.")
        onLog("Watch the log below for progress. It is normal if it pauses briefly.")

        for target in [packageName, gitInstallURL] {
            onLog("Trying pip install \(target)...")
            do {
                let output = try runCommandStreaming(
                    executable: python,
                    arguments: ["-m", "pip", "install", "--user", "--upgrade", target],
                    onLog: onLog
                )
                if !output.isEmpty {
                    onLog(output)
                }
                onLog("Package installed successfully.")
                return
            } catch {
                onLog("Failed: \(error.localizedDescription)")
            }
        }
        throw SetupInstallerError.commandFailed(
            "Could not install Prism. Open Terminal and run:\n\(python) -m pip install --user \(packageName)"
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
    private func runCommandStreaming(
        executable: String,
        arguments: [String],
        onLog: @escaping (String) -> Void
    ) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.environment = augmentedEnvironment()

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        let handle = pipe.fileHandleForReading
        var collected = ""
        handle.readabilityHandler = { fileHandle in
            let data = fileHandle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            collected += chunk
            chunk.split(whereSeparator: \.isNewline).forEach { line in
                let text = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty {
                    DispatchQueue.main.async {
                        onLog(text)
                    }
                }
            }
        }

        try process.run()
        process.waitUntilExit()
        handle.readabilityHandler = nil

        let remainder = handle.readDataToEndOfFile()
        if let tail = String(data: remainder, encoding: .utf8), !tail.isEmpty {
            collected += tail
            tail.split(whereSeparator: \.isNewline).forEach { line in
                let text = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty {
                    onLog(text)
                }
            }
        }

        guard process.terminationStatus == 0 else {
            let detail = collected.isEmpty
                ? "Command failed: \(executable) \(arguments.joined(separator: " "))"
                : collected
            throw SetupInstallerError.commandFailed(detail)
        }

        return collected
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
