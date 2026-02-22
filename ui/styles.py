LIGHT_STYLE = """
    QMainWindow { background-color: #F8F9FA; color: #333; }

    QDockWidget { color: #333; font-weight: bold; }
    QDockWidget::title {
        background-color: #E9ECEF;
        padding: 8px;
        border-bottom: 1px solid #DEE2E6;
    }

    QGroupBox {
        font-weight: bold;
        border: 1px solid #DEE2E6;
        border-radius: 4px;
        margin-top: 12px;
        background-color: #FFFFFF;
        padding-top: 15px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        background-color: #FFFFFF;
        color: #007ACC;
    }

    QLabel { color: #495057; font-size: 13px; }

    QLineEdit {
        background-color: #FFFFFF;
        color: #212529;
        border: 1px solid #CED4DA;
        padding: 6px;
        border-radius: 4px;
        selection-background-color: #007ACC;
    }
    QLineEdit:focus { border: 1px solid #007ACC; }

    QComboBox {
        background-color: #FFFFFF;
        color: #212529;
        border: 1px solid #CED4DA;
        padding: 6px;
        border-radius: 4px;
        min-width: 6em;
    }
    QComboBox:hover { border: 1px solid #ADB5BD; }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 0px;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }
    QComboBox::down-arrow {
        width: 0;
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid #555; /* Darker arrow */
        margin-right: 6px;
        margin-top: 2px;
    }

    QPushButton {
        background-color: #E1E1E1;
        color: #333;
        border: 1px solid #999; /* Stronger border */
        padding: 8px 16px;
        border-radius: 3px;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #D1D1D1; border-color: #666; }

    QTabWidget::pane { border: 1px solid #CCC; background: white; border-radius: 2px; }
    QTabBar::tab { background: #E1E1E1; color: #333; padding: 8px 15px; border: 1px solid #CCC; margin-bottom: -1px; }
    QTabBar::tab:selected { background: #FFF; border-bottom: 1px solid #FFF; font-weight: bold; }
"""
