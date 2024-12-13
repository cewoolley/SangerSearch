import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                            QFileDialog, QScrollArea, QFrame)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqIO import AbiIO 

def find_ab1_files(start_path):
    # recursively find all .ab1 files starting from the given path
    ab1_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.lower().endswith('.ab1'):
                ab1_files.append(os.path.join(root, file))
    return ab1_files

def search_sequence(ab1_file, search_seq):
    try:
        # Read the AB1 file
        record = SeqIO.read(ab1_file, "abi")
        sequence = str(record.seq)
        
        # Search for seq in both forward and reverse complement
        search_seq = search_seq.upper()
        sequence = sequence.upper()
        
        # Create rev comp to search that too
        rev_comp = str(Seq(search_seq).reverse_complement())
        
        # Get the position of the match
        forward_pos = sequence.find(search_seq)
        reverse_pos = sequence.find(rev_comp)
        
        if forward_pos != -1:
            return True, "forward", forward_pos
        elif reverse_pos != -1:
            return True, "reverse complement", reverse_pos
        return False, None, None
    
    except Exception as e:
        print(f"Error reading {ab1_file}: {str(e)}")
        return False, None, None

def plot_trace_section(ab1_file, match_position, sequence_length, orientation, search_seq):
    # Read the trace data
    record = SeqIO.read(ab1_file, "abi")
    
    # Get trace data from annotations
    channels = ['DATA9', 'DATA10', 'DATA11', 'DATA12']  # A, C, G, T channels
    colors = ['green', 'blue', 'black', 'red'] 
    base_labels = ['A', 'C', 'G', 'T']
    traces = [record.annotations['abif_raw'][c] for c in channels]
    
    # Calculate the region to display either side of match 
    padding = 20  # bases before and after
    bases_per_point = len(traces[0]) / len(record.seq)
    start_point = int((match_position - padding) * bases_per_point)
    end_point = int((match_position + sequence_length + padding) * bases_per_point)
    
    # Ensure we don't go out of bounds
    start_point = max(0, start_point)
    end_point = min(len(traces[0]), end_point)
    
    # Create the plot
    fig = Figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    
    # Calculate sequence positions for x-axis
    seq_positions = np.linspace(match_position - padding, 
                              match_position + sequence_length + padding,
                              end_point - start_point)
    
    # Plot traces with labels for legend
    for trace, color, base in zip(traces, colors, base_labels):
        ax.plot(seq_positions, trace[start_point:end_point], color=color, alpha=0.5, label=base)
    
    # Highlight the matching region w/ yellow
    ax.axvspan(match_position, match_position + sequence_length, 
               color='yellow', alpha=0.3)
    
    max_height = max(max(trace[start_point:end_point]) for trace in traces)
    for i in range(match_position - padding, match_position + sequence_length + padding):
        if 0 <= i < len(record.seq):
            base = record.seq[i]
            if base not in 'ACGT':  # check for ambiguous bases and label purple
                ax.text(i, max_height * 1.05, base, 
                       fontsize=8, ha='center', va='bottom',
                       color='purple', fontweight='bold')
            else:
                ax.text(i, max_height * 1.05, base,
                       fontsize=8, ha='center', va='bottom')
    
    # Add labels
    ax.set_title(f"Trace View - {os.path.basename(ab1_file)}\n"
                 f"Search Sequence: {search_seq} ({orientation})\n"
                 f"Path: {ab1_file}", pad=20)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Signal Intensity")
    
    fig.tight_layout()
    
    return fig

class SequenceFinderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SangerSearch: .ab1 Sequence Finder")
        self.setMinimumSize(1000, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Add author information
        author_label = QLabel("Track down Sanger sequencing files based on a target sequence!\nConnor Woolley 2024")
        author_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
                padding: 5px;
                font-style: italic;
                text-align: center;
            }
        """)
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(author_label)
        
        # Search section
        search_frame = QFrame()
        search_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        search_layout = QVBoxLayout(search_frame)
        
        # Search input
        search_input_layout = QHBoxLayout()
        self.sequence_input = QLineEdit()
        self.sequence_input.setPlaceholderText("Enter sequence to search...")
        self.sequence_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        
        # Directory selection
        self.path_button = QPushButton("Select Directory")
        self.path_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.path_button.clicked.connect(self.select_directory)
        
        search_input_layout.addWidget(self.sequence_input)
        search_input_layout.addWidget(self.path_button)
        
        # Path display
        self.path_label = QLabel("No directory selected")
        self.path_label.setStyleSheet("color: #666; padding: 5px;")
        
        search_layout.addLayout(search_input_layout)
        search_layout.addWidget(self.path_label)
        
        # Add a status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                padding: 5px;
                margin-top: 5px;
            }
        """)
        search_layout.addWidget(self.status_label)
        
        # Search button
        self.search_button = QPushButton("Search")
        self.search_button.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        self.search_button.clicked.connect(self.perform_search)
        search_layout.addWidget(self.search_button)
        
        # Results section
        results_layout = QHBoxLayout()
        
        # Files list
        self.files_list = QScrollArea()
        self.files_list.setWidgetResizable(True)
        self.files_list.setMinimumWidth(400)
        self.files_list.setMaximumWidth(400)
        self.files_list_widget = QWidget()
        self.files_list_layout = QVBoxLayout(self.files_list_widget)
        self.files_list_layout.addStretch()
        self.files_list.setWidget(self.files_list_widget)
        self.files_list.setStyleSheet("""
            QScrollArea {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        
        # Trace viewer
        self.trace_viewer = QWidget()
        self.trace_layout = QVBoxLayout(self.trace_viewer)
        
        results_layout.addWidget(self.files_list)
        results_layout.addWidget(self.trace_viewer, stretch=2)
        
        # Add all sections to main layout
        layout.addWidget(search_frame)
        layout.addLayout(results_layout)
        
        self.current_directory = None
        self.found_files = []

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.current_directory = directory
            self.path_label.setText(f"Selected: {directory}")

    def perform_search(self):
        if not self.current_directory:
            self.status_label.setText("Please select a directory first!")
            self.status_label.setStyleSheet("color: #d32f2f;")  # Red color for error
            return
            
        if not self.sequence_input.text():
            self.status_label.setText("Please enter a sequence to search for!")
            self.status_label.setStyleSheet("color: #d32f2f;")
            return

        # Clear previous results
        for i in reversed(range(self.files_list_layout.count())): 
            widget = self.files_list_layout.itemAt(i).widget()
            if widget is not None:  # Only remove widgets, not stretches
                widget.setParent(None)
        
        # Show searching status
        self.status_label.setText("Searching...")
        self.status_label.setStyleSheet("color: #1976D2;")  # Blue color for info
        QApplication.processEvents()  # Update the UI
        
        # Find files
        ab1_files = find_ab1_files(self.current_directory)
        self.found_files = []
        
        for file in ab1_files:
            found, orientation, position = search_sequence(file, self.sequence_input.text())
            if found:
                self.found_files.append((file, orientation, position))
                self.add_result_item(file, orientation)

        # Update status with results
        if self.found_files:
            self.status_label.setText(f"Found sequence in {len(self.found_files)} files")
            self.status_label.setStyleSheet("color: #388E3C;")  # Green color for success
        else:
            self.status_label.setText("Sequence not found in any files")
            self.status_label.setStyleSheet("color: #d32f2f;")

        # After adding all results, ensure there's a stretch at the bottom
        self.files_list_layout.addStretch()

    def add_result_item(self, file_path, orientation):
        # Create a frame to hold the result
        result_frame = QFrame()
        result_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 5px;
                margin: 2px;
                padding: 5px;
            }
            QFrame:hover {
                background-color: #e9ecef;
            }
        """)
        
        # Create layout for the frame
        result_layout = QVBoxLayout(result_frame)
        result_layout.setSpacing(2)
        
        # Add filename and orientation labels with explicit text color
        filename_label = QLabel(os.path.basename(file_path))
        filename_label.setStyleSheet("font-weight: bold; color: black;")  # Added explicit color
        
        # Add path label (shortened)
        dir_path = os.path.dirname(file_path)
        if len(dir_path) > 50:  # Shorten very long paths
            dir_path = "..." + dir_path[-47:]
        path_label = QLabel(dir_path)
        path_label.setStyleSheet("color: #666; font-size: 10px;")
        path_label.setWordWrap(True)
        
        orientation_label = QLabel(f"Orientation: {orientation}")
        orientation_label.setStyleSheet("color: #666;")
        
        result_layout.addWidget(filename_label)
        result_layout.addWidget(path_label)
        result_layout.addWidget(orientation_label)
        
        # Make the frame clickable
        result_frame.mousePressEvent = lambda e: self.show_trace(file_path, orientation)
        
        self.files_list_layout.addWidget(result_frame)

    def show_trace(self, file_path, orientation):
        try:
            # Clear previous trace
            for i in reversed(range(self.trace_layout.count())): 
                self.trace_layout.itemAt(i).widget().setParent(None)
                
            # Find the position for this file
            file_info = next(f for f in self.found_files if f[0] == file_path)
            
            # Create new trace
            fig = plot_trace_section(file_path, file_info[2], 
                                   len(self.sequence_input.text()),
                                   orientation, self.sequence_input.text())
            
            canvas = FigureCanvas(fig)
            self.trace_layout.addWidget(canvas)
        except Exception as e:
            self.status_label.setText(f"Error displaying trace: {str(e)}")
            self.status_label.setStyleSheet("color: #d32f2f;")

def main():
    app = QApplication(sys.argv)
    window = SequenceFinderWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()