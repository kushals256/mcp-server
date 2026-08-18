import AppKit

final class SetupWindowController: NSWindowController {
    static let setupCompletedKey = "PrismHasCompletedSetup"

    private let installer = SetupInstaller()
    private var stepLabels: [SetupStep: NSTextField] = [:]
    private var stepIcons: [SetupStep: NSTextField] = [:]
    private var logView: NSTextView!
    private var actionButton: NSButton!
    private var statusLabel: NSTextField!
    private var isRunning = false
    private var logBuffer = ""

    convenience init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 520, height: 460),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Set Up Prism"
        window.center()
        self.init(window: window)
        setupUI()
    }

    private func setupUI() {
        guard let contentView = window?.contentView else { return }

        let title = NSTextField(labelWithString: "Welcome to Prism")
        title.font = .systemFont(ofSize: 22, weight: .semibold)
        title.translatesAutoresizingMaskIntoConstraints = false

        let subtitle = NSTextField(wrappingLabelWithString: "One click installs the MCP server, configures Claude Desktop, and copies a sample dataset.")
        subtitle.font = .systemFont(ofSize: 13)
        subtitle.textColor = .secondaryLabelColor
        subtitle.translatesAutoresizingMaskIntoConstraints = false

        statusLabel = NSTextField(labelWithString: "Click Set up Prism to begin.")
        statusLabel.font = .systemFont(ofSize: 12)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.translatesAutoresizingMaskIntoConstraints = false

        let stepsStack = NSStackView()
        stepsStack.orientation = .vertical
        stepsStack.alignment = .leading
        stepsStack.spacing = 8
        stepsStack.translatesAutoresizingMaskIntoConstraints = false

        for step in SetupStep.allCases {
            let row = NSStackView()
            row.orientation = .horizontal
            row.spacing = 10

            let icon = NSTextField(labelWithString: "○")
            icon.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
            icon.translatesAutoresizingMaskIntoConstraints = false
            stepIcons[step] = icon

            let label = NSTextField(labelWithString: step.title)
            label.font = .systemFont(ofSize: 13)
            label.translatesAutoresizingMaskIntoConstraints = false
            stepLabels[step] = label

            row.addArrangedSubview(icon)
            row.addArrangedSubview(label)
            stepsStack.addArrangedSubview(row)
        }

        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.borderType = .bezelBorder
        scrollView.translatesAutoresizingMaskIntoConstraints = false

        logView = NSTextView()
        logView.isEditable = false
        logView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.textColor = .secondaryLabelColor
        scrollView.documentView = logView

        actionButton = NSButton(title: "Set up Prism", target: self, action: #selector(startSetup))
        actionButton.bezelStyle = .rounded
        actionButton.keyEquivalent = "\r"
        actionButton.translatesAutoresizingMaskIntoConstraints = false

        let copyButton = NSButton(title: "Copy log", target: self, action: #selector(copyLog))
        copyButton.bezelStyle = .rounded
        copyButton.translatesAutoresizingMaskIntoConstraints = false

        contentView.addSubview(title)
        contentView.addSubview(subtitle)
        contentView.addSubview(stepsStack)
        contentView.addSubview(statusLabel)
        contentView.addSubview(scrollView)
        contentView.addSubview(actionButton)
        contentView.addSubview(copyButton)

        NSLayoutConstraint.activate([
            title.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 24),
            title.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 24),

            subtitle.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 8),
            subtitle.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            subtitle.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -24),

            stepsStack.topAnchor.constraint(equalTo: subtitle.bottomAnchor, constant: 20),
            stepsStack.leadingAnchor.constraint(equalTo: title.leadingAnchor),

            scrollView.topAnchor.constraint(equalTo: stepsStack.bottomAnchor, constant: 16),
            scrollView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 24),
            scrollView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -24),
            scrollView.heightAnchor.constraint(equalToConstant: 120),

            statusLabel.topAnchor.constraint(equalTo: scrollView.bottomAnchor, constant: 12),
            statusLabel.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -24),

            actionButton.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -20),
            actionButton.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -24),

            copyButton.centerYAnchor.constraint(equalTo: actionButton.centerYAnchor),
            copyButton.trailingAnchor.constraint(equalTo: actionButton.leadingAnchor, constant: -12),
        ])
    }

    func showWindow() {
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func startSetup() {
        guard !isRunning else { return }
        isRunning = true
        logBuffer = ""
        logView.string = ""
        actionButton.isEnabled = false
        actionButton.title = "Setting up…"
        statusLabel.stringValue = "Please wait — this may take a minute."

        for step in SetupStep.allCases {
            updateStep(step, status: .pending)
        }

        installer.run(
            onStepChange: { [weak self] step, status in
                self?.updateStep(step, status: status)
            },
            onLog: { [weak self] line in
                self?.appendLog(line)
            },
            completion: { [weak self] result in
                self?.finishSetup(result: result)
            }
        )
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

        switch result {
        case .success:
            UserDefaults.standard.set(true, forKey: Self.setupCompletedKey)
            actionButton.title = "Done"
            statusLabel.stringValue = "Success! Quit Claude Desktop (Cmd+Q), reopen it, then try the starter prompt."
            let alert = NSAlert()
            alert.messageText = "Prism is ready"
            alert.informativeText = "Quit Claude Desktop completely (Cmd+Q), reopen it, and ask:\n\nLoad ~/datasets/sample_sales.csv and run a data quality check"
            alert.runModal()
        case .failure(let error):
            actionButton.title = "Try again"
            statusLabel.stringValue = error.localizedDescription
            let alert = NSAlert()
            alert.messageText = "Setup failed"
            alert.informativeText = error.localizedDescription
            alert.runModal()
        }
    }

    @objc private func copyLog() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(logBuffer, forType: .string)
    }
}
