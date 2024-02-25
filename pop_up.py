import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from train_m_single_m_p_pab1_test import main as train_main

class FileUploadDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('File Upload Example')
        self.setGeometry(100, 100, 300, 100)
        
        self.layout = QVBoxLayout()
        
        self.uploadBtn = QPushButton('Upload File')
        self.uploadBtn.clicked.connect(self.openFileDialog)
        
        self.filePathLabel = QLabel('No file selected')
        
        self.layout.addWidget(self.uploadBtn)
        self.layout.addWidget(self.filePathLabel)
        
        self.setLayout(self.layout)
    
    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;CSV Files (*.csv);;Text Files (*.txt);;TSV Files (*.tsv)", 
                                                  options=options)
        if fileName:
            self.filePathLabel.setText(fileName)
            self.processFile(fileName)  # Assuming you have a function to process or pass the file
            self.close()  # Close the pop-up window after processing the file
    
    def processFile(self, filePath):
        train_main(filePath)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileUploadDialog()
    ex.show()
    sys.exit(app.exec_())