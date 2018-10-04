"""Build the tkinter gui root"""
import math
from PyQt5.QtWidgets import *#(QWidget, QToolTip, QDesktopWidget, QPushButton, QApplication)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, pyqtSlot 
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import sys
from GA_system.counting.plot import PlotCanvas
from GA_system.counting.run import CarRunning

THREADS = []

class GuiRoot(QWidget):
    """Root of gui."""
    def __init__(self, dataset, training_data):
        """Create GUI root with datasets dict"""
        super().__init__()
        self.threadpool = QThreadPool()
        self.setFixedSize(1000, 600)
        self.center()
        self.setWindowTitle('GA practice')      
        self.show()
        #read the map and training data
        self.map_datalist = dataset.keys()
        self.map_data = dataset 
        self.training_datalist = training_data.keys()
        self.training_data = training_data

        #creat file choosing area
        self.file_run_creation(self.map_datalist, self.training_datalist)
        
        self.operation_parameter_creation()
        self.ouput_text_creation()
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.file_run)
        vbox.addWidget(self.operation_type)
        vbox.addWidget(self.text_group_box)
        hbox.addLayout(vbox)
        self.m = PlotCanvas(self.map_data)
        hbox.addWidget(self.m)
        self.setLayout(hbox)

    def file_run_creation(self, datalist, training_data):
        self.file_run = QGroupBox("File choose")
        layout = QGridLayout()
        layout.setSpacing(10)
        map_file_label = QLabel("Map file: ")

        self.map_file_choose = QComboBox()
        for i in datalist:
            self.map_file_choose.addItem("{}".format(i))
        self.map_file_choose.currentTextChanged.connect(self.file_changed)

        training_file_label = QLabel("Training file: ")
        self.training_file_choose = QComboBox()
        for i in training_data:
            self.training_file_choose.addItem("{}".format(i))
        self.run_btn = QPushButton("Start", self)
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(map_file_label, 1, 0, 1, 1)
        layout.addWidget(self.map_file_choose, 1, 1, 1, 3)
        layout.addWidget(training_file_label, 2, 0, 1, 1)
        layout.addWidget(self.training_file_choose, 2, 1, 1, 3)
        layout.addWidget(self.run_btn, 3, 0, 1, 4)
        #layout.setContentsMargins(0,0,0,0)
        layout.setVerticalSpacing(0)
        layout.setHorizontalSpacing(0)
        self.file_run.setLayout(layout)
    
    def operation_parameter_creation(self):
        """Operation parameter field"""
        self.operation_type = QGroupBox("Operation parameter setting")
        vbox = QVBoxLayout()
        
        #set Rrproduction selectiopn related widget
        selection_button_group = QButtonGroup(self)
        selec_layout = QVBoxLayout()
        selection_layout = QHBoxLayout()
        selection2_layout = QHBoxLayout()
        selection_label = QLabel("Reproduction Selection :")
        self.radio_wheel = QRadioButton("Roulette Wheel")
        self.radio_tournament = QRadioButton("Tournament")
        selection_button_group.addButton(self.radio_wheel, 11)
        selection_button_group.addButton(self.radio_tournament, 12)
        self.radio_wheel.setChecked(True)
        competitor_setting = QLabel("Competitor number:")
        self.competitor_line = QSpinBox()
        self.competitor_line.setRange(2, 10)
        self.competitor_line.setValue(5)
        self.competitor_line.setMaximumWidth(50)
        self.competitor_line.setDisabled(True)
        self.radio_wheel.toggled.connect(self.set_completitor)
        self.radio_tournament.toggled.connect(self.set_completitor)
        selec_layout.addWidget(selection_label)
        selection_layout.addWidget(self.radio_wheel)
        selection2_layout.addWidget(self.radio_tournament)
        selection2_layout.addWidget(competitor_setting)
        selection2_layout.addWidget(self.competitor_line)
        selection2_layout.insertSpacing(-1,100)
        selec_layout.addLayout(selection_layout)
        selec_layout.addLayout(selection2_layout)

        #Set and operation paremeter region, including iteration times, population number, 
        #mutation probability, crossover probability, network j value
        iteration_layout = QHBoxLayout()
        iteration_setting = QLabel("Iteration times :")
        self.iteration_line = QSpinBox()
        self.iteration_line.setRange(1, 10000)
        self.iteration_line.setValue(400)
        self.iteration_line.setMaximumWidth(100)
        iteration_layout.addWidget(iteration_setting)
        iteration_layout.addWidget(self.iteration_line)
        iteration_layout.insertSpacing(-1,150)

        population_layout = QHBoxLayout()
        population_setting = QLabel("Population number:")
        self.population_line = QSpinBox()
        self.population_line.setRange(1, 10000)
        self.population_line.setValue(200)
        self.population_line.setMaximumWidth(100)
        population_layout.addWidget(population_setting)
        population_layout.addWidget(self.population_line)
        population_layout.insertSpacing(-1,150)

        mutation_layout = QHBoxLayout()
        mutation_setting = QLabel("Mutation probability: ")
        self.mutation_line = QDoubleSpinBox()
        self.mutation_line.setValue(0.6)
        self.mutation_line.setRange(0, 1)
        self.mutation_line.setDecimals(2)
        self.mutation_line.setMaximumWidth(100)
        mutation_layout.addWidget(mutation_setting)
        mutation_layout.addWidget(self.mutation_line)
        mutation_layout.insertSpacing(-1,150)

        crossover_layout = QHBoxLayout()
        crossover_setting = QLabel("Crossover probability: ")
        self.crossover_line = QDoubleSpinBox()
        self.crossover_line.setRange(0, 1)
        self.crossover_line.setDecimals(2)
        self.crossover_line.setValue(0.6)
        self.crossover_line.setMaximumWidth(100)
        crossover_layout.addWidget(crossover_setting)
        crossover_layout.addWidget(self.crossover_line)
        crossover_layout.insertSpacing(-1,150)

        net_j_layout = QHBoxLayout()
        net_j_setting = QLabel("Network neurl number j: ")
        self.net_j_line = QSpinBox()
        self.net_j_line.setRange(1,10)
        self.net_j_line.setValue(6)
        self.net_j_line.setMaximumWidth(100)
        net_j_layout.addWidget(net_j_setting)
        net_j_layout.addWidget(self.net_j_line)
        net_j_layout.insertSpacing(-1,200)

        sd_layout = QHBoxLayout()
        sd_setting = QLabel("Maximum SD: ")
        self.sd_line = QSpinBox()
        self.sd_line.setRange(1,100)
        self.sd_line.setValue(10)
        self.sd_line.setMaximumWidth(80)
        sd_layout.addWidget(sd_setting)
        sd_layout.addWidget(self.sd_line)
        sd_layout.insertSpacing(-1,250)

        vbox.addLayout(iteration_layout)
        vbox.addLayout(population_layout)
        vbox.addLayout(selec_layout)
        vbox.addLayout(crossover_layout)
        vbox.addLayout(mutation_layout)
        vbox.addLayout(net_j_layout)
        vbox.addLayout(sd_layout)
        self.operation_type.setLayout(vbox)
    def ouput_text_creation(self):
        self.text_group_box = QGroupBox("Execution log")
        layout = QVBoxLayout()
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)
        self.text_group_box.setLayout(layout)
    def file_changed(self):
        """print map"""
        self.m.plot_map(self.map_file_choose.currentText())
        self.console.append('Map changed')
    def run(self):
        l = []
        l.append(self.iteration_line.value())
        l.append(self.population_line.value())
        if self.radio_wheel.isChecked():
            l.append('w')
        elif self.radio_tournament.isChecked():
            l.append('t')
        l.append(self.crossover_line.value())
        l.append(self.mutation_line.value())
        l.append(self.net_j_line.value())
        l.append(self.competitor_line.value())
        l.append(self.sd_line.value())
        # disable avoid to touch
        self.disable('yes')
        self.console.append('Training start')
        car = CarRunning(self.map_data, self.map_file_choose.currentText(),
        self.training_data, self.training_file_choose.currentText(), l)
        car.signals.result.connect(self.plot_output)
        self.threadpool.start(car)
    def plot_output(self, s):
        self.console.append('Training complete')
        self.console.append('Best parameters :')
        self.console.append('theta: {}'.format(s[1].theta))
        self.console.append('means: {}'.format(s[1].means))
        self.console.append('weight: {}'.format(s[1].weight))
        self.console.append('SD: {}'.format(s[1].sd))
        self.console.append('Error rate: {}'.format(1/s[1].adapt_value))
        self.m.plot_car(s[0])
        self.disable('no')
    def center(self):
        """Place window in the center"""
        qr = self.frameGeometry()
        central_p = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(central_p)
        self.move(qr.topLeft())
    def disable(self, yes_or_no):
        if yes_or_no == 'yes':
            self.iteration_line.setDisabled(True)
            self.population_line.setDisabled(True)
            self.radio_wheel.setDisabled(True)
            self.radio_tournament.setDisabled(True)
            self.crossover_line.setDisabled(True)
            self.mutation_line.setDisabled(True)
            self.net_j_line.setDisabled(True)
            self.map_file_choose.setDisabled(True)
            self.training_file_choose.setDisabled(True)
            self.run_btn.setDisabled(True)
            self.competitor_line.setDisabled(True)
            self.sd_line.setDisabled(True)
        else:
            self.iteration_line.setDisabled(False)
            self.population_line.setDisabled(False)
            self.radio_wheel.setDisabled(False)
            self.radio_tournament.setDisabled(False)
            self.crossover_line.setDisabled(False)
            self.mutation_line.setDisabled(False)
            self.net_j_line.setDisabled(False)
            self.map_file_choose.setDisabled(False)
            self.training_file_choose.setDisabled(False)
            self.run_btn.setDisabled(False)
            self.sd_line.setDisabled(False)
    def set_completitor(self):
        if self.radio_tournament.isChecked() == True:
            self.competitor_line.setDisabled(False)
        else:
            self.competitor_line.setDisabled(True)
if __name__ == '__main__':
    print("Error: This file can only be imported. Execute 'main.py'")
