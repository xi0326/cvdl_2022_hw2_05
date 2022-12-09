from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import MainWindow as ui
from Q5.Q5 import Question5

class Main(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.imagePath = None

        self.pushButtonLoadImage.clicked.connect(self.showImageOnGUI)
        self.pushButtonShowImages.clicked.connect(Q5Object.showImages)
        self.pushButtonShowDistribution.clicked.connect(Q5Object.showDistribution)
        self.pushButtonShowModelStructure.clicked.connect(Q5Object.showModelStructure)
        self.pushButtonShowComparsion.clicked.connect(Q5Object.showComparison)
        self.pushButtonShowInference.clicked.connect(self.showInference)
        
    def selectImage(self):
        fileName = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(None, caption='Choose a File', directory='C:\\', filter='Image Files (*.png *.jpg *.bmp)')[0])  # get tuple[0] which is file name
        return fileName
    
    def getImagePath(self):
        self.imagePath = self.selectImage()

    def showImageOnGUI(self):
        self.getImagePath()
        self.photo.setPixmap(QtGui.QPixmap(self.imagePath))

    def showInference(self):
        label = Q5Object.showInference(self.imagePath)

        # while label is not None
        if label:
            self.textArea.setText('Prediction Label: ' + label)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Q5Object = Question5()
    
    window = Main()
    window.show()
    sys.exit(app.exec_())