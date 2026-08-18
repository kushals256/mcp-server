import AppKit

enum ManualStep: CaseIterable {
    case installFromDMG
    case python
    case claudeDesktop
    case quitAndReopenClaude
    case firstPrompt

    var title: String {
        switch self {
        case .installFromDMG:
            return "Download the DMG and drag Prism to Applications"
        case .python:
            return "Install Python 3.10+ (python.org)"
        case .claudeDesktop:
            return "Install Claude Desktop (claude.ai/download)"
        case .quitAndReopenClaude:
            return "Quit Claude Desktop completely (Cmd+Q), then reopen it"
        case .firstPrompt:
            return "Try the starter prompt in Claude"
        }
    }
}

final class SetupWindowController: NSWindowController {
    static let setupCompletedKey = "PrismHasCompletedSetup"
    static let starterPrompt = "Load ~/datasets/sample_sales.csv and run a data quality check"

    private let installer = SetupInstaller()
    private let prerequisitesChecker = PrerequisitesChecker()

    private var prereqIcons: [ManualStep: NSTextField] = [:]
    private var prereqLabels: [ManualStep: NSTextField] = [:]
    private var afterSetupIcons: [ManualStep: NSTextField] = [:]
    private var afterSetupLabels: [ManualStep: NSTextField] = [:]
    private var stepLabels: [SetupStep: NSTextField] = [:]
    private var stepIcons: [SetupStep: NSTextField] = [:]

    private var logView: NSTextView!
    private var actionButton: NSButton!
    private var statusLabel: NSTextField!
    private var setupStepsStack: NSStackView!
    private var isRunning = false
    private var logBuffer = ""
    private var hasNotifiedMissingPrereqs = false

    convenience init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 560, height: 680),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Set Up Prism"
        window.center()
        self.init(window: window)
        setupUI()
        refreshPrerequisites()
    }

    private func setupUI() {
        guard let contentView = window?.contentView else { return }

        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        scrollView.translatesAutoresizingMaskIntoConstraints = false

        let documentView = NSView()
        documentView.translatesAutoresizingMaskIntoConstraints = false

        let title = sectionTitle("Welcome to Prism")
        let subtitle = sectionBody(
            "Prism automates server install and Claude configuration. A few steps still need you — listed below."
        )

        let beforeHeader = sectionHeader("Before you start")
        let beforeStack = makeChecklistStack(
            steps: [.installFromDMG, .python, .claudeDesktop],
            icons: &prereqIcons,
            labels: &prereqLabels
        )

        let autoHeader = sectionHeader("Prism does this automatically")
        setupStepsStack = NSStackView()
        setupStepsStack.orientation = .vertical
        setupStepsStack.alignment = .leading
        setupStepsStack.spacing = 8
        setupStepsStack.translatesAutoresizingMaskIntoConstraints = false

        for step in SetupStep.allCases {
            let row = makeChecklistRow(
                title: step.title,
                iconMap: &stepIcons,
                labelMap: &stepLabels,
                key: step
            )
            setupStepsStack.addArrangedSubview(row)
        }

        let afterHeader = sectionHeader("You'll still need to do this")
        let afterStack = makeChecklistStack(
            steps: [.quitAndReopenClaude, .firstPrompt],
            icons: &afterSetupIcons,
            labels: &afterSetupLabels
        )

        let starterField = NSTextField(wrappingLabelWithString: "Starter prompt: \"\(Self.starterPrompt)\"")
        starterField.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        starterField.textColor = .secondaryLabelColor
        starterField.translatesAutoresizingMaskIntoConstraints = false

        statusLabel = NSTextField(labelWithString: "Click Set up Prism when the checklist above is ready.")
        statusLabel.font = .systemFont(ofSize: 12)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.translatesAutoresizingMaskIntoConstraints = false

        let logScroll = NSScrollView()
        logScroll.hasVerticalScroller = true
        logScroll.borderType = .bezelBorder
        logScroll.translatesAutoresizingMaskIntoConstraints = false

        logView = NSTextView()
        logView.isEditable = false
        logView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.textColor = .secondaryLabelColor
        logScroll.documentView = logView

        actionButton = NSButton(title: "Set up Prism", target: self, action: #selector(startSetup))
        actionButton.bezelStyle = .rounded
        actionButton.keyEquivalent = "\r"
        actionButton.translatesAutoresizingMaskIntoConstraints = false

        let copyPromptButton = NSButton(title: "Copy starter prompt", target: self, action: #selector(copyStarterPrompt))
        copyPromptButton.bezelStyle = .rounded
        copyPromptButton.translatesAutoresizingMaskIntoConstraints = false

        let copyLogButton = NSButton(title: "Copy log", target: self, action: #selector(copyLog))
        copyLogButton.bezelStyle = .rounded
        copyLogButton.translatesAutoresizingMaskIntoConstraints = false

        let refreshButton = NSButton(title: "Refresh checklist", target: self, action: #selector(refreshPrerequisites))
        refreshButton.bezelStyle = .rounded
        refreshButton.translatesAutoresizingMaskIntoConstraints = false

        documentView.addSubview(title)
        documentView.addSubview(subtitle)
        documentView.addSubview(beforeHeader)
        documentView.addSubview(beforeStack)
        documentView.addSubview(autoHeader)
        documentView.addSubview(setupStepsStack)
        documentView.addSubview(afterHeader)
        documentView.addSubview(afterStack)
        documentView.addSubview(starterField)
        documentView.addSubview(statusLabel)
        documentView.addSubview(logScroll)
        documentView.addSubview(actionButton)
        documentView.addSubview(copyPromptButton)
        documentView.addSubview(copyLogButton)
        documentView.addSubview(refreshButton)

        scrollView.documentView = documentView
        contentView.addSubview(scrollView)

        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: contentView.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),

            documentView.widthAnchor.constraint(equalTo: scrollView.widthAnchor),

            title.topAnchor.constraint(equalTo: documentView.topAnchor, constant: 24),
            title.leadingAnchor.constraint(equalTo: documentView.leadingAnchor, constant: 24),
            title.trailingAnchor.constraint(equalTo: documentView.trailingAnchor, constant: -24),

            subtitle.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 8),
            subtitle.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            subtitle.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            beforeHeader.topAnchor.constraint(equalTo: subtitle.bottomAnchor, constant: 20),
            beforeHeader.leadingAnchor.constraint(equalTo: title.leadingAnchor),

            beforeStack.topAnchor.constraint(equalTo: beforeHeader.bottomAnchor, constant: 8),
            beforeStack.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            beforeStack.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            autoHeader.topAnchor.constraint(equalTo: beforeStack.bottomAnchor, constant: 18),
            autoHeader.leadingAnchor.constraint(equalTo: title.leadingAnchor),

            setupStepsStack.topAnchor.constraint(equalTo: autoHeader.bottomAnchor, constant: 8),
            setupStepsStack.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            setupStepsStack.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            afterHeader.topAnchor.constraint(equalTo: setupStepsStack.bottomAnchor, constant: 18),
            afterHeader.leadingAnchor.constraint(equalTo: title.leadingAnchor),

            afterStack.topAnchor.constraint(equalTo: afterHeader.bottomAnchor, constant: 8),
            afterStack.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            afterStack.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            starterField.topAnchor.constraint(equalTo: afterStack.bottomAnchor, constant: 10),
            starterField.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            starterField.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            statusLabel.topAnchor.constraint(equalTo: starterField.bottomAnchor, constant: 16),
            statusLabel.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            logScroll.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 12),
            logScroll.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            logScroll.trailingAnchor.constraint(equalTo: title.trailingAnchor),
            logScroll.heightAnchor.constraint(equalToConstant: 100),

            refreshButton.topAnchor.constraint(equalTo: logScroll.bottomAnchor, constant: 16),
            refreshButton.leadingAnchor.constraint(equalTo: title.leadingAnchor),

            copyLogButton.centerYAnchor.constraint(equalTo: refreshButton.centerYAnchor),
            copyLogButton.leadingAnchor.constraint(equalTo: refreshButton.trailingAnchor, constant: 12),

            copyPromptButton.centerYAnchor.constraint(equalTo: refreshButton.centerYAnchor),
            copyPromptButton.leadingAnchor.constraint(equalTo: copyLogButton.trailingAnchor, constant: 12),

            actionButton.centerYAnchor.constraint(equalTo: refreshButton.centerYAnchor),
            actionButton.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            actionButton.bottomAnchor.constraint(equalTo: documentView.bottomAnchor, constant: -24),
        ])
    }

    private func sectionTitle(_ text: String) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.font = .systemFont(ofSize: 22, weight: .semibold)
        field.translatesAutoresizingMaskIntoConstraints = false
        return field
    }

    private func sectionHeader(_ text: String) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.font = .systemFont(ofSize: 13, weight: .semibold)
        field.translatesAutoresizingMaskIntoConstraints = false
        return field
    }

    private func sectionBody(_ text: String) -> NSTextField {
        let field = NSTextField(wrappingLabelWithString: text)
        field.font = .systemFont(ofSize: 13)
        field.textColor = .secondaryLabelColor
        field.translatesAutoresizingMaskIntoConstraints = false
        return field
    }

    private func makeChecklistStack(
        steps: [ManualStep],
        icons: inout [ManualStep: NSTextField],
        labels: inout [ManualStep: NSTextField]
    ) -> NSStackView {
        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 8
        stack.translatesAutoresizingMaskIntoConstraints = false

        for step in steps {
            let row = makeChecklistRow(title: step.title, iconMap: &icons, labelMap: &labels, key: step)
            stack.addArrangedSubview(row)
        }
        return stack
    }

    private func makeChecklistRow<K: Hashable>(
        title: String,
        iconMap: inout [K: NSTextField],
        labelMap: inout [K: NSTextField],
        key: K
    ) -> NSStackView {
        let row = NSStackView()
        row.orientation = .horizontal
        row.spacing = 10

        let icon = NSTextField(labelWithString: "○")
        icon.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        iconMap[key] = icon

        let label = NSTextField(wrappingLabelWithString: title)
        label.font = .systemFont(ofSize: 12)
        labelMap[key] = label

        row.addArrangedSubview(icon)
        row.addArrangedSubview(label)
        return row
    }

    func showWindow() {
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        refreshPrerequisites()
        NotificationManager.shared.notify(
            title: "Prism setup",
            body: "Review the checklist, then click Set up Prism."
        )
    }

    @objc private func refreshPrerequisites() {
        let status = prerequisitesChecker.evaluate()
        updateManualStep(.installFromDMG, met: status.installedInApplications, icons: prereqIcons, labels: prereqLabels)

        switch status.python {
        case .ok(let path):
            updateManualStep(.python, met: true, icons: prereqIcons, labels: prereqLabels)
            prereqLabels[.python]?.stringValue = "Python 3.10+ found at \(path)"
        case .missing:
            updateManualStep(.python, met: false, icons: prereqIcons, labels: prereqLabels)
        case .tooOld(let version):
            prereqIcons[.python]?.stringValue = "✕"
            prereqLabels[.python]?.stringValue = "Upgrade Python \(version) to 3.10+ (python.org)"
            prereqLabels[.python]?.textColor = .systemRed
        }

        updateManualStep(.claudeDesktop, met: status.claudeInstalled, icons: prereqIcons, labels: prereqLabels)

        if status.readyForSetup {
            statusLabel.stringValue = "Ready — click Set up Prism to install and configure everything."
            actionButton.isEnabled = !isRunning
        } else if status.canAttemptSetup {
            statusLabel.stringValue = "You can run setup, but install Claude Desktop before using Prism."
            actionButton.isEnabled = !isRunning
        } else {
            statusLabel.stringValue = "Install Python 3.10+ first, then click Refresh checklist."
            actionButton.isEnabled = false
        }

        if !status.readyForSetup && !hasNotifiedMissingPrereqs {
            hasNotifiedMissingPrereqs = true
            var missing: [String] = []
            if !status.installedInApplications { missing.append("move Prism to Applications") }
            if !status.canAttemptSetup { missing.append("install Python 3.10+") }
            if !status.claudeInstalled { missing.append("install Claude Desktop") }
            if !missing.isEmpty {
                NotificationManager.shared.notify(
                    title: "Before setup",
                    body: "Please " + missing.joined(separator: ", ") + "."
                )
            }
        }

        for step in SetupStep.allCases {
            updateStep(step, status: .pending)
        }
    }

    private func updateManualStep(
        _ step: ManualStep,
        met: Bool,
        icons: [ManualStep: NSTextField],
        labels: [ManualStep: NSTextField]
    ) {
        icons[step]?.stringValue = met ? "✓" : "○"
        labels[step]?.textColor = met ? .systemGreen : .labelColor
        if met {
            labels[step]?.stringValue = step.title
        }
    }

    @objc private func startSetup() {
        guard !isRunning else { return }
        let status = prerequisitesChecker.evaluate()
        guard status.canAttemptSetup else {
            NotificationManager.shared.notify(
                title: "Python required",
                body: "Install Python 3.10+ from python.org, then refresh the checklist."
            )
            prerequisitesChecker.openPythonDownload()
            return
        }

        isRunning = true
        logBuffer = ""
        logView.string = ""
        actionButton.isEnabled = false
        actionButton.title = "Setting up…"
        statusLabel.stringValue = "Installing packages — first run can take 3–5 minutes. Watch the log below."

        for step in SetupStep.allCases {
            updateStep(step, status: .pending)
        }

        NotificationManager.shared.notify(
            title: "Prism setup started",
            body: "Installing the server and configuring Claude Desktop."
        )

        installer.run(
            onStepChange: { [weak self] step, stepStatus in
                self?.updateStep(step, status: stepStatus)
                self?.notifyForStep(step, status: stepStatus)
            },
            onLog: { [weak self] line in
                self?.appendLog(line)
            },
            completion: { [weak self] result in
                self?.finishSetup(result: result)
            }
        )
    }

    private func notifyForStep(_ step: SetupStep, status: SetupStepStatus) {
        switch status {
        case .running:
            NotificationManager.shared.notify(title: "Prism", body: step.title + "…")
        case .success:
            NotificationManager.shared.notify(title: "Prism", body: step.title + " — done")
        case .failed:
            NotificationManager.shared.notify(title: "Prism setup issue", body: step.title + " failed")
        case .pending:
            break
        }
    }

    private func updateStep(_ step: SetupStep, status: SetupStepStatus) {
        guard let icon = stepIcons[step], let label = stepLabels[step] else { return }
        switch status {
        case .pending:
            icon.stringValue = "○"
            label.textColor = .labelColor
        case .running:
            icon.stringValue = "◌"
            label.textColor = .controlAccentColor
        case .success:
            icon.stringValue = "✓"
            label.textColor = .systemGreen
        case .failed:
            icon.stringValue = "✕"
            label.textColor = .systemRed
        }
    }

    private func appendLog(_ line: String) {
        logBuffer += line + "\n"
        logView.string = logBuffer
        logView.scrollToEndOfDocument(nil)
    }

    private func finishSetup(result: Result<Void, Error>) {
        isRunning = false
        actionButton.isEnabled = true
        refreshPrerequisites()

        switch result {
        case .success:
            UserDefaults.standard.set(true, forKey: Self.setupCompletedKey)
            actionButton.title = "Done"
            statusLabel.stringValue = "Automated setup finished. Complete the two steps below, then use Claude."

            updateManualStep(.quitAndReopenClaude, met: false, icons: afterSetupIcons, labels: afterSetupLabels)
            updateManualStep(.firstPrompt, met: false, icons: afterSetupIcons, labels: afterSetupLabels)
            afterSetupIcons[.quitAndReopenClaude]?.stringValue = "→"
            afterSetupIcons[.firstPrompt]?.stringValue = "→"
            afterSetupLabels[.quitAndReopenClaude]?.textColor = .controlAccentColor
            afterSetupLabels[.firstPrompt]?.textColor = .controlAccentColor

            NotificationManager.shared.notify(
                title: "Prism is ready",
                body: "Quit Claude Desktop (Cmd+Q), reopen it, then try the starter prompt."
            )

            let alert = NSAlert()
            alert.messageText = "Automated setup complete"
            alert.informativeText = """
            Prism installed the server and configured Claude.

            You still need to:
            1. Quit Claude Desktop completely (Cmd+Q)
            2. Reopen Claude Desktop
            3. Ask: \(Self.starterPrompt)
            """
            alert.addButton(withTitle: "Copy starter prompt")
            alert.addButton(withTitle: "Open Claude")
            alert.addButton(withTitle: "OK")
            let response = alert.runModal()
            if response == .alertFirstButtonReturn {
                copyStarterPrompt()
            } else if response == .alertSecondButtonReturn {
                prerequisitesChecker.openClaudeApp()
            }
        case .failure(let error):
            actionButton.title = "Try again"
            statusLabel.stringValue = error.localizedDescription
            NotificationManager.shared.notify(
                title: "Prism setup failed",
                body: error.localizedDescription
            )
            let alert = NSAlert()
            alert.messageText = "Setup failed"
            alert.informativeText = error.localizedDescription
            alert.runModal()
        }
    }

    @objc private func copyLog() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(logBuffer, forType: .string)
        NotificationManager.shared.notify(title: "Copied", body: "Setup log copied to clipboard.")
    }

    @objc private func copyStarterPrompt() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(Self.starterPrompt, forType: .string)
        NotificationManager.shared.notify(title: "Copied", body: "Starter prompt copied to clipboard.")
    }
}
