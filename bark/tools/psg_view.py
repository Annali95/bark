from PyQt5.QtWidgets import QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QWidget, QComboBox, QLabel, QRadioButton, QCheckBox, QGridLayout, QLineEdit, QScrollArea

 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt

import os
import sys
import string
import yaml
import numpy as np

import bark
from bark.io.eventops import (OpStack, write_stack, read_stack, Update, Merge,
                              Split, Delete, New)
import warnings
warnings.filterwarnings('ignore')  # suppress matplotlib warnings


help_string = '''
Pressing any number or letter (uppercase or lowercase) will mark a segment.

Shortcuts
---------
any letter or number    annotate segment
ctrl+s                  saves the annotation data
ctrl+h                  prints this message
ctrl+o                  zoom out
ctrl+i                  zoom in
ctrl+j,right            next segment
ctrl+i,left             previous segment

ctrl+z                  undo last operation
ctrl+y                  redo
ctrl+w                  close

click on the right of the current label        next segment
click on the left of the current label        previous segment

The top panel is a map of all label locations.
Click on a label to travel to that location.

On close, an operation file and the final event file will be written.
Do not kill from terminal unless you want to prevent a save.

To create custom label, input the label name in the right toolbar.


''' 


color = 'green'
fontsize = 12
# kill all the shorcuts
def kill_shortcuts(plt):
    plt.rcParams['keymap.all_axes'] = ''
    plt.rcParams['keymap.back'] = ''
    plt.rcParams['keymap.forward'] = ''
    plt.rcParams['keymap.fullscreen'] = ''
    plt.rcParams['keymap.grid'] = ''
    plt.rcParams['keymap.home'] = ''
    plt.rcParams['keymap.pan'] = ''
    #plt.rcParams['keymap.quit'] = ''
    plt.rcParams['keymap.save'] = ''
    plt.rcParams['keymap.xscale'] = ''
    plt.rcParams['keymap.yscale'] = ''
    plt.rcParams['keymap.zoom'] = ''


def labels_to_scatter_coords(labels):
    times = [x['start'] for x in labels]
    values = []
    for record in labels:
        name = record['name']
        if not isinstance(name, str) or name == '':
            v = 0
        elif name.isdigit():
            v = int(name)
        elif name[0].isalpha():
            # alphabet in range 11-36
            v = 133 - ord(name[0].lower())
        else:
            v = 37
        values.append(v)
    return times, values


def nearest_label(labels, xdata):
    return np.argmin(np.abs(xdata - np.array([x['start'] for x in labels])))


def write_metadata(path, meta='.meta.yaml'):
    import codecs
    params = {'columns': {'name': {'units': 'null'},
                          'start': {'units': 's'},
                          'stop': {'units': 's'}},
                          'datatype': 2002}
    bark.write_metadata(path, meta, **params)


def build_shortcut_map(mapfile=None):
    allkeys = string.digits + string.ascii_letters
    shortcut_map = {x: x for x in allkeys}
    # load keys from file
    if mapfile:
        custom = {str(key): value
                  for key, value in yaml.load(open(mapfile, 'r')).items()}
        print('custom keymaps:', custom)
        shortcut_map.update(custom)
    return shortcut_map


def to_seconds(dset):
    'TODO Converts bark EventData object to units of seconds.'
    if 'offset' in dset.attrs and dset.attrs['offset'] != 0:
        raise Exception('offsets are not yet supported in event file')
    if dset.attrs['columns']['start']['units'] == 's':
        pass
    elif 'units' in dset.attrs and dset.attrs['units'] == 's':
        pass
    else:
        raise Exception('only units of s are supported in event file')
    return dset


def load_opstack(opsfile, labelfile, labeldata, use_ops):
    load_ops = os.path.exists(opsfile) and use_ops
    if load_ops:
        opstack = read_stack(opsfile)
        print('Reading operations from {}.'.format(opsfile))
        if len(opstack.original_events) != len(labeldata):
            print("The number of segments in autosave file is incorrect.")
            sys.exit(0)
        for stack_event, true_event in zip(opstack.original_events, labeldata):
            if (stack_event['name'] != true_event['name'] or
                    not np.allclose(stack_event['start'], true_event['start'])
                    or
                    not np.allclose(stack_event['stop'], true_event['stop'])):
                print("Warning! Autosave:\n {}\n Original:\n{}"
                      .format(stack_event, true_event))
    else:
        opstack = OpStack(labeldata)
    return opstack



def createlabel(name,start,end,interval):
    import pandas as pd
    data = []
    while start+interval < end:
        x=start
        y=start+interval
        dict = {"start":x,"stop":y,"name":""}
        data.append(dict)
        start += interval    
    df = pd.DataFrame(data) 
    df.to_csv(name,index=False) 

def getfiles():
    file = FileDialog()   
    files = file.openFileNamesDialog() 
    if not files:
        sys.exit(app.exec_())
    sampled = [bark.read_sampled(file) for file in files]   
    readonlylabelfile = file.openFileNameDialog()
    if not readonlylabelfile:
        import pandas as pd
        origin_labels = pd.DataFrame()    
    else:
        origin_labels = bark.read_events(readonlylabelfile).data   
    return files, sampled, sampled



def readfiles(outfile=None, shortcutfile=None, use_ops=True):
    gap = 0
    file = FileDialog()   
    files = file.openFileNamesDialog() 
    if not files:
        sys.exit(app.exec_())
    sampled = [bark.read_sampled(file) for file in files]   
    readonlylabelfile = file.openFileNameDialog()
    if not readonlylabelfile:
        import pandas as pd
        origin_labels = pd.DataFrame()    
    else:
        origin_labels = bark.read_events(readonlylabelfile).data   
    trace_num = len(files)
    dat = files[0]
    labelfile = os.path.splitext(dat)[0] + '_split.csv' 
    exist = os.path.exists(labelfile)
    kill_shortcuts(plt)
    opsfile = labelfile + '.ops.json'  
    metadata = labelfile + '.meta.yaml'
    if not os.path.exists(labelfile):
        write_metadata(labelfile)

    if not os.path.exists(labelfile):
        showDia = Input()
        gap = int(showDia.showDialog())
        start = 0
        end = int(round(len(sampled[0].data)/sampled[0].attrs["sampling_rate"]))
        trace_num = len(sampled)
        createlabel(labelfile,start,end,gap)    


    labels = bark.read_events(labelfile)
    labeldata = to_seconds(labels).data.to_dict('records')
    if len(labeldata) == 0:
        print('{} contains no intervals.'.format(labelfile))
        return
    opstack = load_opstack(opsfile, labelfile, labeldata, use_ops)
    if not gap:
        if len(opstack.events) == 0:
            print('opstack is empty. Please delete {}.'.format(opstack))
            return
        gap = opstack.events[0]['stop'] - opstack.events[0]['start']

    shortcuts = build_shortcut_map(shortcutfile)
    #create a new outfile  
    if not outfile:
        outfile = os.path.splitext(labelfile)[0] + '_edit.csv'

    return origin_labels,trace_num, gap, sampled, opstack, shortcuts, outfile, labels.attrs, opsfile


class Plot:
    def __init__(self,ax, x_visible=True, y_visible=True, gap=3):
        self.ax = ax
        self.ax.set_axis_bgcolor('k')
        self.boundary_start = self.ax.axvline(color=color)
        self.boundary_stop = self.ax.axvline(color=color)
        self.gap = gap
        self.label = self.ax.text(0,0,'',fontsize=fontsize,color=color)
        self.label.set_visible(False)
        self.ax.figure.tight_layout()
        self.ax.get_xaxis().set_visible(x_visible)
        self.ax.get_yaxis().set_visible(y_visible)

    def update_x_axis(self,start,stop):
        self.ax.set_xlim(start,stop)

    def clear_plot(self):
        self.ax.cla()
   
    def update_one_label(self,start,stop,name):
        self.boundary_start.set_xdata((start, start))
        self.boundary_stop.set_xdata((stop, stop))
        ymin, ymax = self.ax.get_ylim()
        y = (ymin+ymax)/4
        self.label.set_text(name)
        self.label.set_x((start+stop)/2)
        self.label.set_y(y)
        self.label.set_visible(True)

class Psg_Plot(Plot):
    def __init__(self,ax,trace_num,data,sr, N_points, x_visible=True, y_visible=True, yrange=0.5, gap=3):
       

        Plot.__init__(self,ax,x_visible,y_visible,gap)
        self.init = False
        self.data = data
        self.sr = sr
        self.yrange = yrange
        self.N_points = N_points
        self.psg_line = []
        self.trace_num = trace_num
        self.display = []

        for i in range(self.trace_num):
            line, = self.ax.plot(
                np.arange(self.N_points),
                np.zeros(self.N_points),
                color='gray')
            self.psg_line.append(line)
            self.display.append(True)


    def update_y_axis(self,yrange):
        self.yrange = yrange

    def update_boundary(self,start,stop):
        self.boundary_start.set_xdata((start, start))
        self.boundary_stop.set_xdata((stop, stop))

    def get_yrange(self,buffer_start_samp,buffer_stop_samp):
        x = self.data[0][buffer_start_samp:buffer_stop_samp]   
        return max(x) - min(x)

    def update_psgillograms(self,buffer_start_samp,buffer_stop_samp):

        offset = self.yrange/2
        buf_start = buffer_start_samp / self.sr
        buf_stop = buffer_stop_samp / self.sr

        for i in range(self.trace_num):
            if self.display[i]: 
                self.psg_line[i].set_visible(True)
            else:
                self.psg_line[i].set_visible(False)
            
            x = self.data[i][buffer_start_samp:buffer_stop_samp]   
            t = np.arange(len(x)) / self.sr + buf_start
            if len(x) > 10000:
                t_interp = np.linspace(buf_start, buf_stop, 10000)
                x_interp = np.interp(t_interp, t, x)
            else:
                t_interp = t
                x_interp = x
            x_interp = list(map(lambda num: num+offset, x_interp))
            self.psg_line[i].set_data(t_interp, x_interp)
            offset += self.yrange

        self.update_x_axis(buf_start, buf_stop)    
        self.ax.yaxis.set_ticks(np.arange(0, offset, self.yrange/4))
        self.ax.set_ylim(0,offset)
        self.set_y()
        self.init = True
    
    def set_y(self):
        labels = [item.get_text() for item in self.ax.get_yticklabels()]
        
        for i in range(len(labels)-1):
            if i%4 == 1:
                labels[i] = -self.yrange/4
            elif i%4 == 2:
                labels[i] = "track %d"%((len(labels) - i)/4)
            elif i%4 == 3:
                labels[i] = self.yrange/4
            else :
                labels[i] = ""

        self.ax.set_yticklabels(labels)


class Label_Plot(Plot):
    def __init__(self,ax, x_visible=True, y_visible=True,gap=3):
        Plot.__init__(self,ax,x_visible,y_visible,gap)
        self.labels = [self.ax.text(0,0,'',fontsize=fontsize,color=color) for _ in range(20)]
        self.boundaries_start =  [self.ax.axvline(color=color) for _ in range(20)]
        self.boundaries_stop =  [self.ax.axvline(color=color) for _ in range(20)]
    def update_y_axis(self,yrange):
        self.yrange = yrange

    def clear_plot(self):
        for a,b,c in zip(self.labels,self.boundaries_start,self.boundaries_stop):
            a.set_visible(False)
            b.set_visible(False)
            c.set_visible(False)


    def update_labels(self,current,data,origin_data=False): 
        if origin_data == False:
            self.update_opstack_labels(current,data)
        else:
            self.update_origin_labels(current,data)

    def update_opstack_labels(self,current,opstack,y_pos = 4):

        'labels for current syl and two on either side'
        for i in range(current -8, 11 + current):

            label_i = i - current
            if i >= 0 and i < len(opstack.events):
                text = self.labels[label_i]    
                start_line = self.boundaries_start[label_i]
                start_line.set_visible(True)
                stop_line = self.boundaries_stop[label_i]
                stop_line.set_visible(True)

                start = opstack.events[i]['start']
                stop = opstack.events[i]['stop']
                x = (start+stop) / 2                
                ymin, ymax = self.ax.get_ylim()
                y = (ymin+ymax)/y_pos
                name = opstack.events[i]['name']

                if isinstance(name, str):
                    text.set_x(x)
                    text.set_visible(True)
                    text.set_text(name)
                    text.set_y(y)

                else:
                    text.set_visible(False)

                start_line.set_xdata((start, start))
                stop_line.set_xdata((stop, stop))


            else:
                self.labels[label_i].set_visible(False)

    def update_origin_labels(self,current,origin_data):
        xmin, xmax = self.ax.get_xlim()
        for i in range(0,len(origin_data)):
            if(origin_data["start"][i]>=xmin):
                break
        while origin_data["stop"][i] < xmax:     
            name = origin_data["name"][i]
            self.ax.axvline(x=origin_data["start"][i],color='pink')
            self.ax.axvline(x=origin_data["stop"][i],color='pink')  
            ymin, ymax = self.ax.get_ylim()
            pos_y = (ymin+ymax)/3
            pos_x = origin_data["start"][i] + (origin_data["stop"][i]-origin_data["start"][i])/2
            self.ax.text(pos_x, pos_y, name, fontsize=12, color = 'pink')
            i += 1



class PlotCanvas(FigureCanvas):
 
    def __init__(self, 
                 origin_data, 
                 trace_num, 
                 gap, 
                 sampled, 
                 opstack, 
                 keymap, 
                 outfile, 
                 out_attrs, 
                 opsfile=None, 
                 parent=None, 
                 width=5, 
                 height=10, 
                 dpi=100):
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        
        self.maxpoint = 20

        self.axes = fig.add_subplot(height+1,1,2)
        self.axes_1 = fig.add_subplot(height+1,1,3)
        self.axes_2 = fig.add_subplot(height+1,1,4)
        self.axes_3 = fig.add_subplot(height+1,1,5)        
        pos = self.axes_3.get_position()        
        pos_psg = [pos.x0, 0.05 ,pos.width, pos.y0 - 0.1]        
        self.axes_4 = fig.add_axes(pos_psg)

        self.origin_data = origin_data
        self.trace_num = trace_num
        self.data = []
        self.sr = 0
        self.gap = gap
        self.yrange = 4000
        for dataset in sampled:
            self.data.append(dataset.data.ravel())
            self.sr = dataset.sampling_rate
        self.maxpoint = 15        
        self.N_points = int(round(self.sr*self.gap*self.maxpoint)) 
        self.label_attrs = out_attrs
        self.opstack = opstack
        self.opsfile = opsfile
        self.outfile = outfile
        self.keymap = keymap
        self.y_init = False

     
        if opstack.ops:
            self.label_index = opstack.ops[-1].index
        else:
            self.label_index = 0       

        self.psg_ax = Psg_Plot(ax= self.axes_4,
                               gap=gap,
                               trace_num=trace_num,
                               N_points = self.N_points, 
                               data=self.data,
                               sr = self.sr
                               )
        if not self.origin_data.empty:
            self.label_ax = Label_Plot(ax=self.axes_1,x_visible=False, y_visible=False,gap=gap)
        
        self.label_ax_2 = Label_Plot(ax=self.axes_2,x_visible=False, y_visible=False,gap=gap)
        self.current_ax = Plot(ax=self.axes_3,x_visible=False, y_visible=False,gap=gap)
        self.map_ax = self.axes
        self.initialize_minimap()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.update_plot_data()
 
 
    def update_plot_data(self):

        if not self.opstack.events:
            print('no segments')
            plt.close("all")
            return
                
        i = self.label_index       
        sr = self.sr
        start = self.opstack.events[i]['start']
        start_samp = int(start * sr)
        stop = self.opstack.events[i]['stop']
        stop_samp = int(stop * sr)
        syl_samps = stop_samp - start_samp     
        buffer_start_samp = start_samp - (self.N_points - syl_samps) // 2
        
        if buffer_start_samp < 0:
            buffer_start_samp = 0
        
        buffer_stop_samp = buffer_start_samp + self.N_points
        
        if buffer_stop_samp >= self.data[0].shape[0]:
            buffer_stop_samp = self.data[0].shape[0] - 1
        
        buf_start = buffer_start_samp / sr
        buf_stop = buffer_stop_samp / sr

        name = self.opstack.events[i]['name']
        #fix me ax line covered by the grey line
        # if self.y_init == False:
        #     self.yrange = self.psg_ax.get_yrange(buffer_start_samp,buffer_stop_samp)
        #     self.y_init = True

        self.psg_ax.yrange = self.yrange
        self.psg_ax.update_psgillograms(buffer_start_samp,buffer_stop_samp)
        self.psg_ax.update_boundary(start,stop)

        if self.N_points > int(round(self.sr*self.gap*self.maxpoint)):
            if not self.origin_data.empty:           
                self.label_ax.clear_plot()
            self.label_ax_2.clear_plot()
        else:
            
            if not self.origin_data.empty:            
                self.label_ax.update_x_axis(buf_start,buf_stop)   
                self.label_ax.update_labels(current = i,data = self.origin_data,origin_data=True) 
           
            self.label_ax_2.update_x_axis(buf_start,buf_stop)       
            self.label_ax_2.update_labels(current = i,data = self.opstack,origin_data=False)
        self.current_ax.update_one_label(start,stop,name)
        self.current_ax.update_x_axis(buf_start,buf_stop)     

        self.update_minimap()
        
        if self.opstack.ops:
            last_command = str(self.opstack.ops[-1])
        else:
            last_command = 'none'
        if i == 0:
            self.map_ax.set_title('ctrl+h for help, prints to terminal')
        else:
            self.map_ax.set_title('{}/ {} {}'.format(i + 1, len(
                self.opstack.events), last_command))

        self.draw()



    def initialize_minimap(self):
        times, values = labels_to_scatter_coords(self.opstack.events)
        self.map_ax.set_axis_bgcolor('k')
        self.map_ax.scatter(times,
                            values,
                            c=values,
                            vmin=0,
                            vmax=37,
                            cmap=plt.get_cmap('hsv'),
                            edgecolors='none')
        self.map_ax.vlines(self.opstack.events[self.label_index]['start'],
                           -1,
                           38,
                           zorder=0.5,
                           color='w',
                           linewidth=1)
        self.map_ax.tick_params(axis='y',
                                which='both',
                                left='off',
                                right='off',
                                labelleft='off')
        self.map_ax.set_ylim(-1, 38)

    def update_minimap(self):
        # If perfomance lags, may need to adjust plot elements instead of
        # clearing everything and starting over.
        self.map_ax.clear()
        self.initialize_minimap()


     
    def connect(self):
        'creates all the event connections'

        self.cid_key_press = self.mpl_connect('key_press_event',
                                                     self.on_key_press)
        self.cid_mouse_press = self.mpl_connect('button_press_event',
                                                       self.on_mouse_press)


    def on_mouse_press(self, event):
        start_pos = self.psg_ax.boundary_start.get_xdata()[0]
        stop_pos = self.psg_ax.boundary_stop.get_xdata()[0]

        # jump to syllable from map click
        if event.inaxes == self.map_ax:
            i = nearest_label(self.opstack.events, float(event.xdata))
            self.label_index = i
            self.update_plot_data()

        elif event.inaxes == self.axes_4:
            if event.xdata < start_pos:
                self.dec_i()
            elif event.xdata > stop_pos: 
                self.inc_i()        



    def inc_i(self):
        'Go to next syllable.'
        if self.label_index < len(self.opstack.events) - 1:
            self.label_index += 1
        self.update_plot_data()

    def dec_i(self):
        'Go to previous syllable.'
        if self.label_index > 0:
            self.label_index -= 1
        self.update_plot_data()

    def on_key_press(self, event):
                
        if event.key()  == Qt.Key_Right:
            self.inc_i()
        elif event.key() == Qt.Key_Left:
            self.dec_i()
        elif event.key() <= Qt.Key_Z and event.key() >= Qt.Key_A:
            if self.N_points > int(round(self.sr*self.gap*self.maxpoint)):
                return
            newlabel = chr(event.key())
            self.opstack.push(Update(self.label_index, 'name', newlabel))
            self.inc_i()

    def addlabel(self,str):
        if self.N_points > int(round(self.sr*self.gap*self.maxpoint)):
                return
        self.opstack.push(Update(self.label_index, 'name', str))
        self.inc_i()
    
    def deletelabel(self):
        if self.N_points > int(round(self.sr*self.gap*self.maxpoint)):
                return
        self.opstack.push(Update(self.label_index, '', str))

    def zoom_in_x(self):
        self.N_points += int(self.sr * self.gap)*20
        self.update_plot_data()
    
    def zoom_out_x(self):
        if self.N_points > int(self.sr * self.gap)*20:
            self.N_points -= int(self.sr * self.gap)*20
        self.update_plot_data()

    def zoom_in_y(self):
        self.yrange *= 2 
        self.update_plot_data()
    
    def zoom_out_y(self):
        self.yrange /= 2 
        self.update_plot_data()

    def delete(self):
        self.opstack.push(Delete(self.label_index))
        if self.label_index >= len(self.opstack.events):
            self.label_index = len(self.opstack.events) - 1
        self.update_plot_data()       

    def redo(self):
        if self.opstack.undo_ops:
            self.opstack.redo()
            self.label_index = self.opstack.ops[-1].index
            self.update_plot_data()

    def undo(self):
        if self.opstack.ops:
            self.opstack.undo()
            self.label_index = self.opstack.undo_ops[-1].index
            self.update_plot_data()

    def save(self):
        'Writes out labels to file.'
        from pandas import DataFrame
        label_data = DataFrame(self.opstack.events)
        bark.write_events(self.outfile, label_data, **self.label_attrs)
        print(self.outfile, 'written')
        if self.opsfile:
            write_stack(self.opsfile, self.opstack)
            print(self.opsfile, 'written')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.save()



class FileDialog(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height) 
        self.show()
 
    def openFileNamesDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"Choose all the sound files", "","dat Files (*.dat);;Python Files (*.py)", options=options)
        if files:
            return files
        else:
            self.close()

    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Choose a label file", "","csv Files (*.csv);;Python Files (*.py)", options=options)
        if fileName:
            return fileName


class Input(QWidget):
    
    def __init__(self):
        super().__init__()     
      
    def showDialog(self):       
        gap, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the gap of each label:')        
        if ok:
            return gap 


class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        
        self.left = 0
        self.top = 20
        self.title = 'PsgView'
        app = QtWidgets.QApplication(sys.argv)
        screen = app.primaryScreen()
        size = screen.size()
        rect = screen.availableGeometry()
        self.width = rect.width()/2
        self.height = rect.height()

        self.setWindowTitle("PsgView")
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)
        self.initUI()
        self.initToolbar()

        self.show()

    def initToolbar(self):

        self.file_menu = QtWidgets.QMenu('&File', self)
        # Quit
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        self.file_menu.addAction('&Save', self.reviewer.save, QtCore.Qt.CTRL + QtCore.Qt.Key_S)
        self.file_menu.addAction('&Help', self.help, QtCore.Qt.CTRL + QtCore.Qt.Key_H)
        self.file_menu.addAction('&Redo', self.reviewer.redo, QtCore.Qt.CTRL + QtCore.Qt.Key_Y)
        self.file_menu.addAction('&Undo', self.reviewer.undo, QtCore.Qt.CTRL + QtCore.Qt.Key_Z)
        # control        
        self.control_menu = QtWidgets.QMenu('&Control', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.control_menu)
        
        self.control_menu.addAction('&zoom in x', self.reviewer.zoom_in_x, QtCore.Qt.CTRL + QtCore.Qt.Key_A)
        self.control_menu.addAction('&zoom out x',self.reviewer.zoom_out_x, QtCore.Qt.CTRL + QtCore.Qt.Key_D)
        self.control_menu.addAction('&zoom in y', self.reviewer.zoom_in_y, QtCore.Qt.CTRL + QtCore.Qt.Key_W)
        self.control_menu.addAction('&zoom out y',self.reviewer.zoom_out_y, QtCore.Qt.CTRL + QtCore.Qt.Key_E)
        self.control_menu.addAction('&next',self.reviewer.inc_i, QtCore.Qt.CTRL + QtCore.Qt.Key_J)
        self.control_menu.addAction('&previous',self.reviewer.dec_i,QtCore.Qt.CTRL + QtCore.Qt.Key_F)
        self.control_menu.addAction('&delete',self.reviewer.deletelabel,QtCore.Qt.Key_Backspace)



    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        origin_labels,trace_num, gap, sampled, opstack, shortcuts, outfile, attrs, opsfile = readfiles()    
        height = trace_num*2 + 4
        
        self.reviewer = PlotCanvas(origin_labels,trace_num, gap, sampled, 
            opstack, shortcuts, outfile, attrs, opsfile, parent = self, width=8, height=height)
        self.reviewer.connect()
        self.reviewer.move(20,20)

        self.scroll = QScrollArea(self.widget)
        self.scroll.setWidget(self.reviewer)
        self.widget.layout().addWidget(self.scroll)
        self.checkboxes = []

        lbl = QLabel(self)
        lbl.setText("Customize Label")
        self.label = QLineEdit(self)
        label_button = QPushButton("Add Label",self)
        label_button.setToolTip('This is an example button')

        lbl_2 = QLabel(self)
        lbl_2.setText("Number of Label to display")
        self.label_2 = QLineEdit(self)
        label_2_button = QPushButton("Ok",self)
        label_2_button.setToolTip('Change the number of label to display')
        start_x = 820
        lbl.move(start_x, 30)
        self.label.move(start_x, 60)
        label_button.move(start_x,90)
        label_button.clicked.connect(self.add_label)

        lbl_2.move(start_x, 120)
        self.label_2.move(start_x, 150)
        label_2_button.move(start_x,180)
        label_2_button.clicked.connect(self.change_label)

        for i in range(0,trace_num):
            box = QCheckBox('Track'+ str(i+1), self)
            box.move(start_x,310+20*i)
            box.ind = trace_num-1-i 
            box.stateChanged.connect(self.state_changed)
            box.setChecked(True)
            self.checkboxes.append(box)

    def add_label(self):
        text = self.label.text()
        self.reviewer.addlabel(text)

    def change_label(self):
        number = self.label_2.text()
        if number.isdigit():
            self.reviewer.N_points = int(round(self.reviewer.sr*self.reviewer.gap*int(number))) 
            self.reviewer.update_plot_data()
    
    def change_label_max(self):
        number = int(self.label_3.text())
        self.reviewer.maxpoint = number
        self.reviewer.update_plot_data()


    def change_ylim(self):
        inputNumber = self.ylim.text()
        if inputNumber.isdigit():
            self.reviewer.yrange = float(inputNumber)
            self.reviewer.update_plot_data()
           
    def change_gap(self):
        inputNumber = self.gap_input.text()
        if inputNumber.isdigit():
            self.reviewer.gap = int(inputNumber)
            self.reviewer.update_plot_data()
        else:
            info = "Please select a number, `{0}` isn't valid!"
    
    def fileQuit(self):
        self.close()
    
    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """This is bark psg viewer"""
                                )
    def help(self):
        QtWidgets.QMessageBox.about(self, "Help", help_string)
    
    def keyPressEvent(self,e):
        self.reviewer.on_key_press(e)


    def state_changed(self, state):
        target = self.sender()
        index = target.ind 
        if state == Qt.Checked:
            self.reviewer.psg_ax.display[index] = True
        else:
            self.reviewer.psg_ax.display[index] = False

        self.reviewer.update_plot_data()


def _main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

if __name__ == '__main__':
    _main()


