"""
Main entry point for the DCE-MRI Analysis Tool GUI application.

This script initializes the PyQt5 application, creates an instance of the
`MainWindow` from the `ui.main_window` module, displays the main window,
and starts the Qt event loop.
"""
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow # Import the main window class

if __name__ == "__main__":
    # Create a QApplication instance. This is necessary for any PyQt5 GUI application.
    # sys.argv allows passing command-line arguments to the application (if any).
    app = QApplication(sys.argv)

    # Create an instance of the main application window.
    main_win = MainWindow()

    # Show the main window.
    main_win.show()

    # Start the Qt event loop.
    # sys.exit() ensures a clean exit, passing the application's exit status.
    # app.exec_() starts the event loop, which processes user interactions and other events.
    sys.exit(app.exec_())
