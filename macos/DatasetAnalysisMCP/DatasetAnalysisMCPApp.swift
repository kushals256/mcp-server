import AppKit
import Foundation

@main
struct DatasetAnalysisMCPApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusBarController: StatusBarController?
    private var setupWindowController: SetupWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusBarController = StatusBarController(
            onRequestSetup: { [weak self] in
                self?.showSetupWindow()
            }
        )

        let configManager = ConfigManager()
        let hasCompleted = UserDefaults.standard.bool(forKey: SetupWindowController.setupCompletedKey)
        if !configManager.isConfigured || !hasCompleted {
            showSetupWindow()
        }
    }

    func showSetupWindow() {
        if setupWindowController == nil {
            setupWindowController = SetupWindowController()
        }
        setupWindowController?.showWindow()
    }

    func applicationWillTerminate(_ notification: Notification) {
        statusBarController?.stop()
    }
}
