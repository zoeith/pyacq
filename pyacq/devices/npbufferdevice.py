# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import numpy as np

from ..core import Node, register_node_type
from pyqtgraph.Qt import QtCore, QtGui


class NumpyDeviceBuffer(Node):
    """A fake analogsignal device.

    This node streams data from a predefined buffer in an endless loop.
    """
    _output_specs = {'signals': dict(streamtype='analogsignal',
                                                shape=(-1, 16), compression ='', sample_rate =30.
                                                )}

    def __init__(self, **kargs):
        Node.__init__(self, **kargs)

    def configure(self, *args, **kwargs):
        """
        Parameters
        ----------
        nb_channel: int
            Number of output channels.
        sample_interval: float
            Time duration of a single data sample. This determines the rate at
            which data is sent.
        chunksize: int
            Length of chunks to send.
        buffer: array
            Data to send. Must have `buffer.shape[0] == nb_channel`.
        cycle_times: int (Default:0)
            Number of times to play the buffer. 0 for infinite.
        cycled_callback: callback function (Default: None)
            After sending cycle_times, optionally call this function.
        cycle_end_set: float (Default: 0.0)
            After cycle_times is reached, fill the buffer with this value.
        """
        return Node.configure(self, *args, **kwargs)

    def _configure(self, nb_channel=16, sample_interval=0.001, chunksize=256,
                    buffer=None, channel_names=None, cycle_times=0,
                    cycled_callback=None, cycle_end_set = 0.0):
        self.nb_channel = nb_channel
        self.sample_interval = sample_interval
        self.chunksize = chunksize
        self.cycle_times = cycle_times
        self.cycle_count = 0
        self.cycled_callback =cycled_callback
        self.cycle_end_set = cycle_end_set

        if channel_names is None:
            channel_names = [ 'chan{}'.format(c) for c in range(self.nb_channel) ]
        self.channel_names = channel_names

        self.output.spec['shape'] = (-1, nb_channel)
        self.output.spec['sample_rate'] = 1. / sample_interval

        if buffer is None:
            nloop = 40
            self.length = nloop * self.chunksize
            t = np.arange(self.length) * sample_interval
            self.buffer = np.random.rand(self.length, nb_channel) * .05
            self.buffer += np.sin(2 * np.pi * 440. * t)[:, None] * .5
            self.buffer = self.buffer.astype('float32')
        else:
            assert buffer.shape[1] == self.nb_channel, 'Wrong nb_channel'
            #see new version send_data
            #assert buffer.shape[0] % chunksize == 0, 'Wrong buffer.shape[0] not multiple chunksize'
            self.buffer = buffer
            self.length = buffer.shape[0]

        self.output.spec['dtype'] = buffer.dtype.name

    def after_output_configure(self, outputname):
        if outputname == 'signals':
            channel_info = [ {'name': self.channel_names[c]} for c in range(self.nb_channel) ]
            self.outputs[outputname].params['channel_info'] = channel_info

    def _initialize(self):
        self.head = 0
        ival = int(self.chunksize * self.sample_interval * 1000)
        self.timer = QtCore.QTimer(singleShot=False, interval=ival)
        self.timer.timeout.connect(self.send_data)

    def _start(self):
        self.head = 0
        self.timer.start()

    def _stop(self):
        self.timer.stop()

    def _close(self):
        pass

    def reset(self):
        """
        send 0 to output and stop.
        reset cycle_count to 0
        """

        super().stop()
        self.cycle_count=0
        data = np.ones((self.chunksize,self.nb_channel))*self.cycle_end_set
        self.send_data(data=data)
        if self.cycled_callback is not None:
            self.cycled_callback()

    def send_data(self,data=None):
        if data is None:
            i1 = self.head
            i2 = i1 + self.chunksize
            data=np.take(self.buffer,np.arange(i1,i2),mode='wrap',axis=0)
            if i2 >= self.length:
                self.head = np.remainder(i2,self.chunksize)
                self.cycle_count += i2//self.length
                #self.cycle_count += np.divmod(i2-i1,self.length)[0]
            else:
                self.head = i2
            #check if now over number of cycle times
            if (self.cycle_times>0) and (self.cycle_count>=self.cycle_times):
                #make data just 1 cycle and fill with zeros
                #call reset
                times_over = self.cycle_count - self.cycle_times
                final_values = self.cycle_end_set*np.ones(((times_over*self.length+self.head),self.nb_channel))
                data[:-(times_over*self.length+self.head),:] = final_values
                self.output.send(data,index=self.head)
                self.reset() # stop things then send 0s.
            else:
                self.output.send(data,index=self.head)

        else:
            self.output.send(data,index=self.head)

register_node_type(NumpyDeviceBuffer)
