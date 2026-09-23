# -*- coding: utf-8 -*-

"""
Main Window - Pi-UVMorphe GUI Application
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .styles import MAIN_STYLE, TABLE_STYLE
from .widgets import FileBrowseWidget, DirectoryBrowseWidget, CollapsibleBox, LogTextEdit
from .worker_thread import WorkerThread


class PiMorpheMainWindow(QMainWindow):
    """Pi-UVMorphe Main Window"""

    APP_NAME = "Pi-UVMorphe"
    APP_VERSION = "1.0.0"
    APP_COLOR = "#6C5CE7"

    def __init__(self):
        super().__init__()
        self.worker = None
        self.results = None
        self.init_ui()
        self.setup_connections()

        # Load recent config
        self.load_recent_config()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle(f"{self.APP_NAME} - UVPD Mass Spectrometry Data Analysis Tool v{self.APP_VERSION}")
        self.setGeometry(100, 100, 600, 750)
        self.setMinimumSize(800, 600)
        self.setStyleSheet(MAIN_STYLE)

        # Set icon
        icon_path = Path(__file__).parent.parent / 'resources' / 'icon.ico'
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ---------- Title ----------
        title_layout = QHBoxLayout()

        # Logo text
        title = QLabel("π-UVMorphe")
        title.setStyleSheet(f"""
            font-size: 26px;
            font-weight: bold;
            color: {self.APP_COLOR};
            padding: 10px 0;
            font-family: 'Times New Roman', serif;
        """)
        title_layout.addWidget(title)

        subtitle = QLabel("UVPD Mass Spectrometry Data Analysis")
        subtitle.setStyleSheet("""
            font-size: 14px;
            color: #7f8c8d;
            padding: 10px 0 10px 5px;
        """)
        title_layout.addWidget(subtitle)

        title_layout.addStretch()

        # Version info
        version_label = QLabel(f"v{self.APP_VERSION}")
        version_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        title_layout.addWidget(version_label)
        main_layout.addLayout(title_layout)

        # ---------- Tab Widget ----------
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Analysis tab
        self.analysis_tab = QWidget()
        self.tab_widget.addTab(self.analysis_tab, "🔬 Analysis")

        # Results tab
        self.results_tab = QWidget()
        self.tab_widget.addTab(self.results_tab, "📊 Results")

        # Settings tab
        self.settings_tab = QWidget()
        self.tab_widget.addTab(self.settings_tab, "⚙️ Settings")

        # About tab
        self.about_tab = QWidget()
        self.tab_widget.addTab(self.about_tab, "ℹ️ About")

        # ---------- Analysis Tab ----------
        self.setup_analysis_tab()

        # ---------- Results Tab ----------
        self.setup_results_tab()

        # ---------- Settings Tab ----------
        self.setup_settings_tab()

        # ---------- About Tab ----------
        self.setup_about_tab()

        # ---------- Status Bar ----------
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        # Add progress bar to status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.setObjectName("warning_btn")
        self.clear_log_btn.setFixedWidth(100)
        button_layout.addWidget(self.clear_log_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setObjectName("danger_btn")
        self.exit_btn.setFixedWidth(100)
        button_layout.addWidget(self.exit_btn)

        main_layout.addLayout(button_layout)

    def setup_analysis_tab(self):
        layout = QVBoxLayout(self.analysis_tab)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        input_group = QGroupBox("📂 Input Settings")
        input_layout = QGridLayout(input_group)
        input_layout.setVerticalSpacing(3)  # 行间距调小
        input_layout.setHorizontalSpacing(3)
        input_layout.setContentsMargins(5, 5, 5, 5)


        # MSALIGN file
        input_layout.addWidget(QLabel("MSALIGN File:"), 0, 0)
        self.msalign_widget = FileBrowseWidget(
            placeholder="Select .msalign file",
            file_filter="MSALIGN files (*.msalign);;All files (*.*)"
        )
        input_layout.addWidget(self.msalign_widget, 0, 1, 1, 2)

        # spectrum0.js file
        input_layout.addWidget(QLabel("spectrum0.js:"), 1, 0)
        self.js_widget = FileBrowseWidget(
            placeholder="Select spectrum0.js file",
            file_filter="JavaScript files (*.js);;All files (*.*)"
        )
        input_layout.addWidget(self.js_widget, 1, 1, 1, 2)

        # Protein sequence
        input_layout.addWidget(QLabel("Protein Sequence:"), 2, 0)
        self.seq_edit = QTextEdit()
        self.seq_edit.setPlaceholderText("Enter protein sequence (single-letter amino acid codes)")
        self.seq_edit.setMaximumHeight(80)
        input_layout.addWidget(self.seq_edit, 2, 1, 1, 2)

        # Modification type
        input_layout.addWidget(QLabel("Modification:"), 3, 0)
        self.mod_combo = QComboBox()
        self.mod_combo.addItem("None", None)
        self.mod_combo.addItem("N-Acetylated", "n_acetylated")
        self.mod_combo.addItem("N-Deaminated", "n_deaminated")
        self.mod_combo.addItem("N-Methylated", "n_methylated")
        self.mod_combo.addItem("N-Dimethylated", "n_dimethylated")
        self.mod_combo.addItem("N-Formylated", "n_formylated")
        self.mod_combo.addItem("C-Amided", "c_amided")
        self.mod_combo.addItem("C-Methylated", "c_methylated")
        self.mod_combo.addItem("C-Dehydrated", "c_dehydrated")
        input_layout.addWidget(self.mod_combo, 3, 1)

        # PPM tolerance
        input_layout.addWidget(QLabel("PPM Tolerance:"), 4, 0)
        self.ppm_spin = QSpinBox()
        self.ppm_spin.setRange(1, 100)
        self.ppm_spin.setValue(5)
        input_layout.addWidget(self.ppm_spin, 4, 1)

        layout.addWidget(input_group)

        # ---------- Output Section ----------
        output_group = QGroupBox("💾 Output Settings")
        output_layout = QGridLayout(output_group)
        output_layout.setVerticalSpacing(10)
        output_layout.setHorizontalSpacing(10)

        # Output directory
        output_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self.output_widget = DirectoryBrowseWidget(placeholder="Select output directory")
        self.output_widget.setText("./output")
        output_layout.addWidget(self.output_widget, 0, 1, 1, 2)

        # Options
        options_widget = QWidget()
        options_layout = QHBoxLayout(options_widget)
        options_layout.setSpacing(20)

        self.save_csv_cb = QCheckBox("Save CSV Results")
        self.save_csv_cb.setChecked(True)
        options_layout.addWidget(self.save_csv_cb)

        self.plot_maps_cb = QCheckBox("Generate Cleavage Maps")
        self.plot_maps_cb.setChecked(True)
        options_layout.addWidget(self.plot_maps_cb)

        self.plot_abundance_cb = QCheckBox("Generate Abundance Plots")
        self.plot_abundance_cb.setChecked(True)
        options_layout.addWidget(self.plot_abundance_cb)

        options_layout.addStretch()
        output_layout.addWidget(options_widget, 1, 0, 1, 3)

        layout.addWidget(output_group)

        # ---------- Run Buttons ----------
        run_layout = QHBoxLayout()
        run_layout.addStretch()

        self.run_btn = QPushButton("▶ Run Analysis")
        self.run_btn.setMinimumHeight(45)
        self.run_btn.setMinimumWidth(200)
        self.run_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px;
                background-color: {self.APP_COLOR};
            }}
            QPushButton:hover {{
                background-color: #5A4BD1;
            }}
            QPushButton:disabled {{
                background-color: #bdbdbd;
            }}
        """)
        run_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setMinimumWidth(100)
        self.stop_btn.setObjectName("danger_btn")
        self.stop_btn.setEnabled(False)
        run_layout.addWidget(self.stop_btn)

        run_layout.addStretch()
        layout.addLayout(run_layout)

        # ---------- Log Section ----------
        log_group = QGroupBox("📋 Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = LogTextEdit()
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # Set log font
        font = QFont("Consolas", 9)
        self.log_text.setFont(font)

    def setup_results_tab(self):
        """Setup results tab"""
        layout = QVBoxLayout(self.results_tab)

        # Results summary
        summary_group = QGroupBox("📊 Results Summary")
        summary_layout = QGridLayout(summary_group)

        self.summary_labels = {
            'Matches': QLabel("-"),
            'IRS Score': QLabel("-"),
            'Terminal Coverage': QLabel("-"),
            'Internal Coverage': QLabel("-"),
            'Total Coverage': QLabel("-"),
        }

        row = 0
        for i, (key, label) in enumerate(self.summary_labels.items()):
            row = i // 2
            col = (i % 2) * 2
            summary_layout.addWidget(QLabel(f"{key}:"), row, col)
            label.setStyleSheet(f"font-weight: bold; color: {self.APP_COLOR};")
            summary_layout.addWidget(label, row, col + 1)

        layout.addWidget(summary_group)

        # Results table
        table_group = QGroupBox("📋 Matching Results")
        table_layout = QVBoxLayout(table_group)

        self.result_table = QTableWidget()
        self.result_table.setStyleSheet(TABLE_STYLE)
        self.result_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.result_table)

        layout.addWidget(table_group)

    def setup_settings_tab(self):
        """Setup settings tab"""
        layout = QVBoxLayout(self.settings_tab)

        # Model settings
        model_group = QGroupBox("🤖 Model Settings")
        model_layout = QGridLayout(model_group)

        model_layout.addWidget(QLabel("Model File:"), 0, 0)
        self.model_widget = FileBrowseWidget(
            placeholder="Select model file (.pth)",
            file_filter="PyTorch models (*.pth);;All files (*.*)"
        )
        self.model_widget.setText("best_model.pth")
        model_layout.addWidget(self.model_widget, 0, 1)

        model_layout.addWidget(QLabel("Scaler File:"), 1, 0)
        self.scaler_widget = FileBrowseWidget(
            placeholder="Select scaler file (.pkl)",
            file_filter="Pickle files (*.pkl);;All files (*.*)"
        )
        self.scaler_widget.setText("scaler0424.pkl")
        model_layout.addWidget(self.scaler_widget, 1, 1)

        layout.addWidget(model_group)

        # Device settings
        device_group = QGroupBox("💻 Device Settings")
        device_layout = QHBoxLayout(device_group)

        device_layout.addWidget(QLabel("Compute Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", None)
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA", "cuda")
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()

        layout.addWidget(device_group)

        # Save config button
        save_btn = QPushButton("💾 Save Configuration")
        save_btn.setStyleSheet(f"background-color: {self.APP_COLOR};")
        save_btn.clicked.connect(self.save_config)
        layout.addWidget(save_btn)

        layout.addStretch()

    def setup_about_tab(self):
        """Setup about tab"""
        layout = QVBoxLayout(self.about_tab)
        layout.setAlignment(Qt.AlignCenter)

        # Logo
        logo_label = QLabel("π-UVMorphe")
        logo_label.setStyleSheet(f"""
            font-size: 48px;
            font-weight: bold;
            color: {self.APP_COLOR};
            font-family: 'Times New Roman', serif;
        """)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # Description
        desc_label = QLabel("UVPD Mass Spectrometry Data Analysis Tool")
        desc_label.setStyleSheet("font-size: 18px; color: #7f8c8d;")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        layout.addSpacing(20)

        # Version info
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: transparent;
                font-size: 12px;
                color: #2c3e50;
                text-align: center;
            }
        """)
        info_text.setHtml(f"""
            <div style="text-align: center;">
                <p><b>Version:</b> {self.APP_VERSION}</p>
                <p><b>Description:</b> Deep learning-based UVPD mass spectrometry data analysis platform</p>
                <p><b>Tech Stack:</b> PyTorch, ViT, PyQt5</p>
                <p><b>Author:</b> Bo Yao</p>
                <p><b>License:</b> MIT License</p>
                <br>
                <p style="color: #95a5a6; font-size: 11px;">
                    © 2026 Pi-UVMorphe. All rights reserved.
                </p>
            </div>
        """)
        layout.addWidget(info_text)

        layout.addStretch()

    def setup_connections(self):
        """Setup signal connections"""
        self.run_btn.clicked.connect(self.run_analysis)
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        self.exit_btn.clicked.connect(self.close)

    def run_analysis(self):
        """Run analysis"""
        # Validate input
        msalign = self.msalign_widget.text().strip()
        js = self.js_widget.text().strip()
        seq = self.seq_edit.toPlainText().strip().replace('\n', '').replace('\r', '')

        if not msalign:
            QMessageBox.warning(self, "Error", "Please select an MSALIGN file")
            self.msalign_widget.line_edit.setFocus()
            return

        if not os.path.exists(msalign):
            QMessageBox.warning(self, "Error", f"MSALIGN file not found:\n{msalign}")
            return

        if not js:
            QMessageBox.warning(self, "Error", "Please select a spectrum0.js file")
            self.js_widget.line_edit.setFocus()
            return

        if not os.path.exists(js):
            QMessageBox.warning(self, "Error", f"spectrum0.js file not found:\n{js}")
            return

        if not seq:
            QMessageBox.warning(self, "Error", "Please enter a protein sequence")
            self.seq_edit.setFocus()
            return

        # Get parameters
        mod_index = self.mod_combo.currentIndex()
        mod_value = self.mod_combo.itemData(mod_index)

        output_dir = self.output_widget.text().strip() or "./output"

        scaler_path = self.scaler_widget.text().strip() or "scaler0424.pkl"
        model_path = self.model_widget.text().strip() or "best_model.pth"

        device = self.device_combo.currentData()

        # Check model file
        if not os.path.exists(model_path):
            reply = QMessageBox.question(
                self, "Model File Not Found",
                f"Model file not found:\n{model_path}\n\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Build arguments
        args = argparse.Namespace(
            msalign=msalign,
            js=js,
            sequence=seq,
            modification=mod_value,
            output=output_dir,
            scaler=scaler_path,
            model=model_path,
            device=device,
            batch_size=64,
            ppm=self.ppm_spin.value(),
            no_save_images=False,
            save_csv=self.save_csv_cb.isChecked(),
            no_plot_maps=not self.plot_maps_cb.isChecked(),
            no_plot_abundance=not self.plot_abundance_cb.isChecked(),
            verbose=True
        )

        # Update UI state
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_bar.showMessage("Running...")
        self.log_text.clear()

        # Save current config
        self.save_recent_config(args)

        # Clear result table
        self.result_table.setRowCount(0)
        for key in self.summary_labels:
            self.summary_labels[key].setText("-")

        # Start worker thread
        self.worker = WorkerThread(args)
        self.worker.log.connect(self.log_text.append_log)
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.status_bar.showMessage)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self, matches_df, reliability):
        """Analysis finished"""
        self.reset_ui()

        self.log_text.append_log("✅ Analysis completed!")

        # Update results summary
        if matches_df is not None:
            self.summary_labels['Matches'].setText(str(len(matches_df)))
        else:
            self.summary_labels['Matches'].setText("0")

        if reliability:
            self.summary_labels['IRS Score'].setText(f"{reliability.get('IRS', 0):.4f}")
            self.summary_labels['Terminal Coverage'].setText(f"{reliability.get('terminal_cleavage_coverage', 0):.2%}")
            self.summary_labels['Internal Coverage'].setText(f"{reliability.get('internal_cleavage_coverage', 0):.2%}")
            self.summary_labels['Total Coverage'].setText(f"{reliability.get('total_cleavage_coverage', 0):.2%}")
        else:
            self.summary_labels['IRS Score'].setText("N/A")
            self.summary_labels['Terminal Coverage'].setText("N/A")
            self.summary_labels['Internal Coverage'].setText("N/A")
            self.summary_labels['Total Coverage'].setText("N/A")

        # Update results table
        if matches_df is not None and len(matches_df) > 0:
            self.populate_table(matches_df)
            self.tab_widget.setCurrentIndex(1)  # Switch to results tab

        # Notify user
        summary_msg = (
            f"Analysis completed successfully!\n\n"
            f"Matches: {len(matches_df) if matches_df is not None else 0}"
        )
        if reliability:
            summary_msg += f"\nIRS Score: {reliability.get('IRS', 0):.4f}"

        QMessageBox.information(self, "Complete", summary_msg)

    def on_error(self, error_msg):
        """Analysis error"""
        self.reset_ui()
        self.log_text.append_log(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error_msg}")

    def stop_analysis(self):
        """Stop analysis"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.worker = None
            self.log_text.append_log("⚠ Analysis stopped by user")
            self.reset_ui()

    def reset_ui(self):
        """Reset UI state"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Ready")

    def update_progress(self, value):
        """Update progress"""
        if value > 0 and value < 100:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)

    def populate_table(self, df):
        """Populate results table"""
        # Limit display rows
        max_rows = min(len(df), 1000)

        # Select columns to display
        columns = ['Frag Type', 'Observed Mass', 'Theoretical Mass',
                   'Start AA', 'End AA', 'Error', 'Intensity', 'Charge', 'mz']
        available_cols = [c for c in columns if c in df.columns]

        self.result_table.setRowCount(max_rows)
        self.result_table.setColumnCount(len(available_cols))
        self.result_table.setHorizontalHeaderLabels(available_cols)

        for i in range(max_rows):
            for j, col in enumerate(available_cols):
                value = df.iloc[i][col]
                if isinstance(value, float):
                    value = f"{value:.4f}" if col in ['Error'] else f"{value:.2f}"
                self.result_table.setItem(i, j, QTableWidgetItem(str(value)))

        # Adjust column widths
        self.result_table.resizeColumnsToContents()

    def save_recent_config(self, args):
        """Save recent configuration"""
        config = {
            'app': self.APP_NAME,
            'msalign': args.msalign,
            'js': args.js,
            'sequence': args.sequence[:100] + "..." if len(args.sequence) > 100 else args.sequence,
            'modification': args.modification,
            'output': args.output,
            'ppm': args.ppm,
            'save_csv': args.save_csv,
            'plot_maps': not args.no_plot_maps,
            'plot_abundance': not args.no_plot_abundance
        }
        try:
            config_path = Path.home() / '.pi_morphe_config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass

    def load_recent_config(self):
        """Load recent configuration"""
        try:
            config_path = Path.home() / '.pi_morphe_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)

                if config.get('app') == self.APP_NAME:
                    if 'msalign' in config:
                        self.msalign_widget.setText(config['msalign'])
                    if 'js' in config:
                        self.js_widget.setText(config['js'])
                    if 'sequence' in config and not self.seq_edit.toPlainText():
                        self.seq_edit.setText(config['sequence'])
                    if 'output' in config:
                        self.output_widget.setText(config['output'])
                    if 'ppm' in config:
                        self.ppm_spin.setValue(config['ppm'])
                    if 'save_csv' in config:
                        self.save_csv_cb.setChecked(config['save_csv'])
                    if 'plot_maps' in config:
                        self.plot_maps_cb.setChecked(config['plot_maps'])
                    if 'plot_abundance' in config:
                        self.plot_abundance_cb.setChecked(config['plot_abundance'])

                    self.log_text.append_log("📂 Loaded previous configuration")
        except Exception:
            pass

    def save_config(self):
        """Save configuration"""
        config = {
            'app': self.APP_NAME,
            'model_path': self.model_widget.text().strip(),
            'scaler_path': self.scaler_widget.text().strip(),
            'device': self.device_combo.currentData(),
            'output_dir': self.output_widget.text().strip(),
            'ppm': self.ppm_spin.value(),
        }
        try:
            config_path = Path.home() / '.pi_morphe_config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Success", "Configuration saved successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save configuration:\n{str(e)}")

    def closeEvent(self, event):
        """Close event"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Analysis is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.worker.terminate()
            self.worker.wait()
        event.accept()