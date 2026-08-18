import AppKit
import Foundation

final class StatusBarController {
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private let monitor = ProcessMonitor()
    private let configManager = ConfigManager()
    private var timer: Timer?
    private let onRequestSetup: () -> Void

    init(onRequestSetup: @escaping () -> Void = {}) {
        self.onRequestSetup = onRequestSetup
        configureStatusItem()
        refreshStatus()
        timer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.refreshStatus()
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func configureStatusItem() {
        if let button = statusItem.button {
            button.image = StatusIcon.image(for: .idle)
            button.image?.isTemplate = false
            button.toolTip = "Prism"
        }
        statusItem.menu = buildMenu()
    }

    private func refreshStatus() {
        let state = monitor.currentState(configManager: configManager)
        statusItem.button?.image = StatusIcon.image(for: state)
        statusItem.button?.image?.isTemplate = false
        statusItem.menu = buildMenu()
    }

    private func buildMenu() -> NSMenu {
        let menu = NSMenu()
        let state = monitor.currentState(configManager: configManager)

        let titleItem = NSMenuItem(title: statusTitle(for: state), action: nil, keyEquivalent: "")
        titleItem.isEnabled = false
        menu.addItem(titleItem)
        menu.addItem(.separator())

        menu.addItem(menuItem("Open Claude Desktop", action: #selector(openClaude)))
        menu.addItem(menuItem("Open Data Folder", action: #selector(openDataFolder)))
        menu.addItem(.separator())

        if !configManager.isConfigured {
            menu.addItem(menuItem("Set Up Prism…", action: #selector(showSetup)))
            menu.addItem(.separator())
        } else {
            menu.addItem(menuItem("Run Setup Again…", action: #selector(showSetup)))
            menu.addItem(.separator())
        }

        if configManager.isConfigured && !configManager.isDisabled {
            menu.addItem(menuItem("Disable Server", action: #selector(disableServer)))
        } else {
            menu.addItem(menuItem("Enable Server", action: #selector(enableServer)))
        }

        menu.addItem(menuItem("Run Health Check", action: #selector(runDoctor)))
        menu.addItem(menuItem("Copy Starter Prompt", action: #selector(copyStarterPrompt)))
        menu.addItem(.separator())
        menu.addItem(menuItem("View Docs", action: #selector(openDocs)))
        menu.addItem(menuItem("Quit", action: #selector(quit)))

        return menu
    }

    private func menuItem(_ title: String, action: Selector) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
        item.target = self
        return item
    }

    private func statusTitle(for state: ServerState) -> String {
        switch state {
        case .active:
            return "Prism — Active"
        case .idle:
            return "Prism — Ready"
        case .disabled:
            return "Prism — Disabled"
        case .error:
            return "Prism — Needs Attention"
        }
    }

    @objc private func openClaude() {
        NSWorkspace.shared.open(URL(fileURLWithPath: "/Applications/Claude.app"))
    }

    @objc private func openDataFolder() {
        let url = URL(fileURLWithPath: configManager.dataDirectoryPath, isDirectory: true)
        NSWorkspace.shared.open(url)
    }

    @objc private func disableServer() {
        configManager.disableServer()
        notify(title: "Server Disabled", body: "Quit and reopen Claude Desktop to apply.")
        refreshStatus()
    }

    @objc private func enableServer() {
        configManager.enableServer()
        notify(title: "Server Enabled", body: "Quit and reopen Claude Desktop to connect.")
        refreshStatus()
    }

    @objc private func runDoctor() {
        let output = configManager.runDoctor()
        let alert = NSAlert()
        alert.messageText = "Health Check"
        alert.informativeText = output
        alert.runModal()
        refreshStatus()
    }

    @objc private func copyStarterPrompt() {
        let prompt = "Load ~/datasets/sample_sales.csv and run a data quality check"
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(prompt, forType: .string)
        notify(title: "Copied", body: "Starter prompt copied to clipboard.")
    }

    @objc private func openDocs() {
        if let url = URL(string: "https://kushals256.github.io/mcp-server/install/") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func showSetup() {
        onRequestSetup()
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    private func notify(title: String, body: String) {
        let notification = NSUserNotification()
        notification.title = title
        notification.informativeText = body
        NSUserNotificationCenter.default.deliver(notification)
    }
}

enum ServerState {
    case active
    case idle
    case disabled
    case error
}

enum StatusIcon {
    static func image(for state: ServerState) -> NSImage? {
        let resourceName: String
        switch state {
        case .active:
            resourceName = "menubar-active"
        case .idle, .disabled:
            resourceName = "menubar-idle"
        case .error:
            return NSImage(systemSymbolName: "exclamationmark.triangle", accessibilityDescription: "Prism")
        }

        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "png"),
              let image = NSImage(contentsOf: url) else {
            return NSImage(systemSymbolName: "triangle.fill", accessibilityDescription: "Prism")
        }

        image.size = NSSize(width: 18, height: 18)
        return image
    }
}
