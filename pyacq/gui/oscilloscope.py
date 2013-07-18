# -*- coding: utf-8 -*-

from PyQt4 import QtCore,QtGui
import pyqtgraph as pg
import zmq

from .tools import RecvPosThread
from .guiutil import *
from .multichannelparam import MultiChannelParam

from matplotlib.cm import get_cmap
from matplotlib.colors import ColorConverter


param_global = [
    #~ {'name': 'xsize', 'type': 'logfloat', 'value': 10., 'step': 0.1},
    {'name': 'xsize', 'type': 'logfloat', 'value': 1., 'step': 0.1},
    {'name': 'ylims', 'type': 'range', 'value': [-10., 10.] },
    {'name': 'background_color', 'type': 'color', 'value': 'k' },
    {'name': 'refresh_interval', 'type': 'int', 'value': 100 , 'limits':[5, 1000]},
    ]

param_by_channel = [ 
    #~ {'name': 'channel_name', 'type': 'str', 'value': '','readonly' : True},
    #~ {'name': 'channel_index', 'type': 'str', 'value': '','readonly' : True},
    {'name': 'color', 'type': 'color', 'value': "FF0"},
    #~ {'name': 'width', 'type': 'float', 'value': 1. , 'step': 0.1},
    #~ {'name': 'style', 'type': 'list', 
                #~ 'values': OrderedDict([ ('SolidLine', Qt.SolidLine), ('DotLine', Qt.DotLine), ('DashLine', Qt.DashLine),]), 
                #~ 'value': Qt.SolidLine},
    {'name': 'gain', 'type': 'float', 'value': 1, 'step': 0.1},
    {'name': 'offset', 'type': 'float', 'value': 0., 'step': 0.1},
    {'name': 'visible', 'type': 'bool', 'value': True},
    ]


class MyViewBox(pg.ViewBox):
    doubleclicked = QtCore.pyqtSignal()
    zoom = QtCore.pyqtSignal(float)
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
    def mouseClickEvent(self, ev):
        ev.accept()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.ignore()
    def wheelEvent(self, ev):
        if ev.modifiers() ==  Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.zoom.emit(z)
        ev.accept()

class Oscilloscope(QtGui.QWidget):
    def __init__(self, stream = None, parent = None,):
        QtGui.QWidget.__init__(self, parent)
        
        assert stream['type'] == 'signals_stream'
        
        self.stream = stream
        

        self.mainlayout = QtGui.QVBoxLayout()
        self.setLayout(self.mainlayout)
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_configure_dialog)
        self.graphicsview  = pg.GraphicsView()#useOpenGL = True)
        self.mainlayout.addWidget(self.graphicsview)
        self.plot = pg.PlotItem(viewBox = self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE,'')
        self.socket.connect("tcp://localhost:{}".format(self.stream['port']))
        
        self.thread = RecvPosThread(socket = self.socket, port = self.stream['port'])
        self.thread.start()
        
        self.timer = QtCore.QTimer(interval = 100)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        
        # Create parameters
        n = stream['nb_channel']
        self.np_array = self.stream['shared_array'].to_numpy_array()
        sr = self.stream['sampling_rate']
        
        all = [ ]
        for i, channel_index, channel_name in zip(range(n), stream['channel_indexes'], stream['channel_names']):
            name = 'Signal{} name={} channel_index={}'.format(i, channel_name,channel_index)
            all.append({ 'name': name, 'type' : 'group', 'children' : param_by_channel})
        self.paramSignals = pg.parametertree.Parameter.create(name='AnalogSignals', type='group', children=all)
        self.paramGlobal = pg.parametertree.Parameter.create( name='Global options',
                                                    type='group', children =param_global)
        self.allParams = pg.parametertree.Parameter.create(name = 'all param', type = 'group', children = [self.paramGlobal,self.paramSignals  ])
        
        self.allParams.sigTreeStateChanged.connect(self.on_param_change)
        self.paramGlobal.param('xsize').setLimits([2./sr, self.np_array.shape[1]/sr*.9])
        
        
        self.paramControler = OscilloscopeControler(parent = self)
        self.paramControler.setWindowFlags(Qt.Window)
        self.viewBox.zoom.connect(self.paramControler.gain_zoom)

        
        # Create curve items
        self.curves = [ ]
        for i, channel_index, channel_name in zip(range(n), stream['channel_indexes'], stream['channel_names']):
            color = self.paramSignals.children()[i].param('color').value()
            curve = self.plot.plot([np.nan], [np.nan], pen = color)
            self.curves.append(curve)            
        
        self.paramGlobal.param('xsize').setValue(3)

    def open_configure_dialog(self):
        self.paramControler.show()    
    
    def autoestimate_scales(self):
        if self.thread.pos is None: return None, None
        pos =self.thread.pos
        n = self.stream['nb_channel']
        #~ self.all_mean =  np.array([ np.mean(self.np_array[i,:pos]) for i in range(n) ])
        #~ self.all_sd = np.array([ np.std(self.np_array[i,:pos]) for i in range(n) ])
        # better than std and mean
        self.all_mean = np.array([ np.median(self.np_array[i,:pos]) for i in range(n) ])
        self.all_sd=  np.array([ np.median(np.abs(self.np_array[i,:pos]-self.all_mean[i])/.6745) for i in range(n) ])
        return self.all_mean, self.all_sd
    
    def on_param_change(self, params, changes):
        #~ print params, changes
        for param, change, data in changes:
            if change != 'value': continue
            if param.name() in ['gain', 'offset', 'visible', 'ylims']: continue # done in refresh
            if param.name()=='color':
                i = self.paramSignals.children().index(param.parent())
                pen = pg.mkPen(color = data)
                self.curves[i].setPen(pen)
            if param.name()=='background_color':
                self.graphicsview.setBackground(data)
            if param.name()=='xsize':
                xsize = data
                sr = self.stream['sampling_rate']
                self.intsize = int(xsize*sr)
                self.t_vect = np.arange(self.intsize, dtype = np.float64)/sr
                self.t_vect -= self.t_vect[-1]
            if param.name()=='refresh_interval':
                self.timer.setInterval(data)
        #~ self.refresh()


    
    def refresh(self):
        if self.thread.pos is None: return
        head = self.thread.pos%self.np_array.shape[1]
        
        #~ color = self.paramGlobal.param('background_color').value()
        #~ self.graphicsview.setBackground(color)
        

        
        tail = head-self.intsize
        if tail<0:
            # FIXME : 2 curves and no concatenate here
            #~ print 'need concatenate'
            #~ print self.np_array.shape
            #~ print tail, head, size
            np_arr = np.concatenate([self.np_array[:,self.np_array.shape[1]+tail:], self.np_array[:,0:head], ], axis = 1)
            #~ print np_arr.shape
        else:
            np_arr = self.np_array[:,tail:head]
        
        for c, curve in enumerate(self.curves):
            p = self.paramSignals.children()[c]
            if not p.param('visible').value():
                curve.setData([np.nan], [np.nan])
                continue
            
            g = p.param('gain').value()
            o = p.param('offset').value()
            curve.setData(self.t_vect, np_arr[c,:]*g+o)
        
        self.plot.setXRange( self.t_vect[0], self.t_vect[-1])
        ylims  = self.paramGlobal.param('ylims').value()
        self.plot.setYRange( *ylims )
        








class OscilloscopeControler(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        
        self.oscilloscope = parent

        #layout
        self.mainlayout = QtGui.QVBoxLayout()
        self.setLayout(self.mainlayout)
        t = u'Options for signals - {}'.format(self.oscilloscope.stream['name'])
        self.setWindowTitle(t)
        self.mainlayout.addWidget(QLabel('<b>'+t+'<\b>'))
        
        h = QtGui.QHBoxLayout()
        self.mainlayout.addLayout(h)
        
        self.treeParamSignal = pg.parametertree.ParameterTree()
        self.treeParamSignal.header().hide()
        h.addWidget(self.treeParamSignal)
        self.treeParamSignal.setParameters(self.oscilloscope.paramSignals, showTop=True)
        
        if self.oscilloscope.stream['nb_channel']>1:
            self.multi = MultiChannelParam( all_params = self.oscilloscope.paramSignals, param_by_channel = param_by_channel)
            h.addWidget(self.multi)
        
        v = QtGui.QVBoxLayout()
        h.addLayout(v)
        
        self.treeParamGlobal = pg.parametertree.ParameterTree()
        self.treeParamGlobal.header().hide()
        v.addWidget(self.treeParamGlobal)
        self.treeParamGlobal.setParameters(self.oscilloscope.paramGlobal, showTop=True)

        # Gain and offset
        v.addWidget(QLabel(u'<b>Automatic gain and offset on selection:<\b>'))
        for i,text in enumerate(['Real scale (gain = 1, offset = 0)',
                            'Fake scale (same gain for all)',
                            'Fake scale (gain per channel)',]):
            but = QPushButton(text)
            v.addWidget(but)
            but.mode = i
            but.clicked.connect(self.auto_gain_and_offset)

        v.addWidget(QLabel(self.tr('<b>Automatic color on selection:<\b>'),self))
        but = QPushButton('Progressive')
        but.clicked.connect(lambda : self.automatic_color(cmap_name = None))
        v.addWidget(but)

        v.addWidget(QLabel(self.tr('<b>Gain zoom (mouse wheel on graph):<\b>'),self))
        h = QHBoxLayout()
        v.addLayout(h)
        for label, factor in [ ('--', 1./10.), ('-', 1./1.3), ('+', 1.3), ('++', 10.),]:
            but = QPushButton(label)
            but.factor = factor
            but.clicked.connect(self.gain_zoom)
            h.addWidget(but)
    
    def auto_gain_and_offset(self):
        mode = self.sender().mode

        nb_channel = self.oscilloscope.stream['nb_channel']
        selected = self.multi.selected()
        n = np.sum(selected)
        if n==0: return
        
        av, sd = self.oscilloscope.autoestimate_scales()
        if mode==0:
            ylims = [np.min(av[selected]-3*sd[selected]), np.max(av[selected]+3*sd[selected]) ]
            gains = np.ones(nb_channel, dtype = float)
            offsets = np.zeros(nb_channel, dtype = float)
        elif mode in [1, 2]:
            ylims  = [-.5, n-.5 ]
            gains = np.zeros(nb_channel, dtype = float)
            if mode==1:
                gains = np.ones(nb_channel, dtype = float) * 1./n/max(sd[selected])
            elif mode==2:
                gains = .6/n/sd
            offsets = np.zeros(nb_channel, dtype = float)
            offsets[selected] = range(n)[::-1] - av[selected]*gains[selected]
        
        # apply
        for i, p in enumerate(self.oscilloscope.paramSignals.children()):
            p.param('gain').setValue(gains[i])
            p.param('offset').setValue(offsets[i])
            p.param('visible').setValue(selected[i])
        self.oscilloscope.paramGlobal.param('ylims').setValue(ylims)
    
    
    def automatic_color(self, cmap_name = None):
        if cmap_name is None:
            cmap_name = 'jet'
        #~ nb_channel = self.oscilloscope.stream['nb_channel']
        selected = self.multi.selected()
        n = np.sum(selected)
        if n==0: return
        cmap = get_cmap(cmap_name , n)
        s=0
        for i, p in enumerate(self.oscilloscope.paramSignals.children()):
            if selected[i]:
                color = [ int(c*255) for c in ColorConverter().to_rgb(cmap(s)) ] 
                p.param('color').setValue(color)
                s += 1
    
    def gain_zoom(self, factor):
        if type(factor) is bool:# button
            factor = self.sender().factor
        for i, p in enumerate(self.oscilloscope.paramSignals.children()):
            p.param('gain').setValue(p.param('gain').value()*factor)
            #~ p.param('offset').setValue(p.param('gain').value()/factor)

