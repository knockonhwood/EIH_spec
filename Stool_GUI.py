import sys
import time
import os
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, QThread, pyqtSignal, pyqtSlot, QRect
import numpy as np
import matplotlib.pyplot as plt
sys.setrecursionlimit(10**6)
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QProgressBar,
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget, QComboBox, QMessageBox, QSizePolicy, QSpacerItem, QGridLayout
)
from PyQt5.QtGui import  QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import matplotlib
import time
import sys
import pandas as pd
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
import warnings
import keras.models 
from statsmodels.robust.scale import huber
warnings.filterwarnings("ignore", category=DeprecationWarning)



class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.lower_limit=500 # lower limit of wavelength in nm (all values below are discarded)
        self.upper_limit=600 # upper limit of wavelength in nm
        self.threshold=0.5
        self.setGeometry(80,80, 1000, 900)
        self.setWindowTitle("Blood in stool predictor")
        self.prediction_label='' #empty label for initiation
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.load_average_NB()
        self.load_model()

        horizontalSpacer = QSpacerItem(40, 20,  QSizePolicy.Maximum, QSizePolicy.Expanding)
        self.gridLayout.addItem(horizontalSpacer, 800, 0, Qt.AlignLeft)
        self.gridLayout.addWidget(self, 1, 2, 1, 1)

        self.display1 = QLabel(self)
        self.display1.setFixedSize(600, 600)
        self.gridLayout.addWidget(self.display1, 2,2, Qt.AlignLeft)

        self.Truth_label = QLabel(self)  # Initialize with the parent widget only
        self.Truth_label.setWordWrap(True)
        font = self.Truth_label.font()
        font.setPointSize(16)
        self.Truth_label.setFont(font)
        self.gridLayout.addWidget(self.Truth_label, 5, 0, 1, 5)  # Assuming you want the QLabel to span across 5 columns
        self.Truth_label.setAlignment(Qt.AlignCenter)  # Align the text to the center horizontally
        self.Truth_label.setGeometry(100, 700, 800, 200) 

        self.load = QPushButton("Load sample data", self)
        self.load.setGeometry(20, 20, 200, 60)
        self.load.show()
        
        self.load.clicked.connect(self.load_new_spectra)
        self.gridLayout.addWidget(self.load, 0, 0, Qt.AlignLeft)

    def load_model(self):
        self.model = keras.models.load_model('C:\\Users\\heather\\Desktop\\EIH Spring\\EIH GUI\\Green_model.keras')

    def load_average_NB(self):
         self.NB_avg=pd.read_csv('C:\\Users\\heather\\Desktop\\EIH Spring\\EIH GUI\\Green_avg_NB.csv', sep='\t')
         

    def load_new_spectra(self):
            print('Button is connected')
            path, _ = QFileDialog.getOpenFileName(self, 'Load sample data', '', 'Text Files (*.txt)')
            if path:  # Check if a file was selected
                data = []
                with open(path, 'r') as file:
                    print(f"Loading file: {path}")  # Debug statement
                    new_row = pd.read_csv(file, sep='\t', header=None, skiprows=13)
                    new_row = new_row.iloc[1:, :]  # Assuming you want to skip the first row after header
                    columns = new_row.iloc[:, 0].tolist()  # Convert the first column to a simple list
                    new_row = new_row.iloc[:, 1]  # Selecting the second column as data
                    row_float = [float(value) for value in new_row] # Convert all values to float
                    data.append(row_float)
                    # Reshape the data to have a single column
                    # data = np.array(row_float).reshape(-1, 1)
                    # print(data.shape)
                    print(len(columns))  # Print the length of the columns list for debugging
                    # Create DataFrame with specified columns
                    self.data = pd.DataFrame(np.array(data), columns=columns)
                    self.shrink_spectra()  # Assuming this is a method you have defined elsewhere
                    self.load.setVisible(False)
                    self.display_graph()
                    self.generate_prediction()

    def shrink_spectra(self):
        print(len(self.data))
        data = self.data.loc[:, self.data.columns]
        wavelengths = data.columns.values[1:].astype(float)
        lower_lim_index = np.argmax(wavelengths >= self.lower_limit)  # Index of first wavelength >= 500
        upper_lim_index = np.argmin(wavelengths <= self.upper_limit)  # Index of first wavelength <= 600
        # Select columns based on indices
        filtered_columns = data.columns[lower_lim_index + 1:upper_lim_index + 1]
        # Create a new DataFrame with selected columns
        self.data = data[filtered_columns]
        self.norm_func()
        print('Length of input data', len(self.data))

    def generate_prediction(self):
        self.prediction=self.model.predict(self.data) 
        self.Display_prediction()

    def Display_prediction(self):
        prediction_value = self.prediction[0][0]
        print(prediction_value)
        if prediction_value > self.threshold:
              self.prediction_label='It is likely this sample has blood. The prediced probability is {} %'.format(round(prediction_value *100,2))
              self.Truth_label.setStyleSheet("color : red")
        else:
            self.prediction_label='It is unlikely this sample has blood. The prediced probability is {} %'.format(round(prediction_value *100,2))
            self.Truth_label.setStyleSheet("color : green")
        self.Truth_label.setText(self.prediction_label)

    def norm_func(self):
        X=self.data.iloc[0, :]
        Hmean, Hstd = huber(X)
        HCentered = X - Hmean
        Hnorm = HCentered / Hstd
        self.data.iloc[0, :] = Hnorm

    def display_graph(self):
        matplotlib.use('Agg')
        fig, ax = plt.subplots()
        new_data=self.data.values[0]
        print(len(new_data))
        plt.plot(self.NB_avg.iloc[:,0], self.NB_avg.iloc[:,1], label='Average Without Blood', color='darkslategrey', linewidth=2.4)
        plt.plot(self.data.columns, new_data,label='Your Sample', color='mediumvioletred',linewidth=1.6)
        plt.title('Normalized green spectra',fontsize='18')
        plt.xlabel('Wavelength (nm)',fontsize='16')
        plt.ylabel('Normalized Reflectance',fontsize='16')
        plt.legend(fontsize='14')
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        image_array = np.frombuffer(buf, np.uint8).copy()
        image_array.shape = int(h), int(w), 4
        image_array = image_array[:, :, :3]
        plt.close(fig)
        height, width, channel = image_array.shape
        img_bytes = image_array.tobytes()
        qImg = QImage(img_bytes, width, height, channel*width, QImage.Format_RGB888)
        self.display1.setPixmap(QPixmap.fromImage(qImg))
                    
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
