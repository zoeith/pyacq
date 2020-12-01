# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

"""
A simple device node using python time to send a trigger signal.
"""
import time
import sys
import subprocess
import os
import re
from io import BytesIO,StringIO

import numpy as np

from ..core import Node, register_node_type, ThreadPollInput
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.util.mutex import Mutex

class Timer(Node):
    """
    Simple timer/delay device using a qthread.

    Functions in 3 modes: a continuous timer (metronome), an on demand timer,
    a data delay.

    - If configured for a metronome, it starts either when `run` is called or
    or when the first input to time stream shows up. And emits a `True` on the output
    time stream when time is up.

    - In on demand timer mode, a bool signal sent to the timer input will cause
    the timer to start and result in a `True` emitted on the output.

    - As a delay, data sent to the input data stream is delayed for the configured
    time then passed to a new 'data' output stream. A `True` is also emitted on
    the time output stream.

    """
    _output_specs = {'time': dict(streamtype='digitalsignal',dtype=np.bool,
                                                shape=(1, ), compression ='',
                                                sample_rate = 1.),
                                                }

    _input_specs = {'time': dict(streamtype='digitalsignal',dtype=np.bool,
                                                shape=(1, ), compression ='',
                                                sample_rate = 1.),
                                                }

    def __init__(self, datastream=None, **kargs):

        """
        datastream argument should be a dictionary with at least the
        readonly_params = ['protocol', 'transfermode', 'shape', 'dtype']
        for the input and output datastream to act as a delay or
        on demand timer for that data stream.
        """
        self.datastream=datastream
        if self.datastream is None:
            self.metronome = True
        else:
            self.metronome = False
            Timer._input_specs['data']=self.datastream
            Timer._output_specs['data']=self.datastream


        Node.__init__(self, **kargs)


    def _configure(self, waittime, **options):
        """
        Set a waittime in seconds.

        Here we finish configuring the timer to act as a metronome (default),
        on demand timer, or delay.

        ondemand: default `False`, `True` to act as an ondemand timer.
        pass_data: default `False`, `True` to act as a data delay timer.
        This might be duplicated functionality of the delay mode. Just ignore
        the output data stream.

        From InputStream, certain parameters must be predefined:
        readonly_params = ['protocol', 'transfermode', 'shape', 'dtype']

        """
        self.waittime = waittime
        self.options = options


    def _initialize(self):
        # metronome: timer runs on signal to time input or run.
        if self.datastream is None:
            self.poller = ThreadPollInput(input_stream=self.inputs['time'])
        #ondemand and delay: timer runs on signal to data input.
        else:
            print('Setting up data delay')
            self.poller = ThreadPollInput(input_stream=self.inputs['data'], return_data=True)
        if self.metronome:
            self.poller.new_data.connect(self.tick_tock)
        else:
            self.poller.new_data.connect(self.delay_data)

    def _start(self):
        self.poller.start()

    def _stop(self):
        self.poller.stop()

    def tick_tock(self, pos, data):
        while self.running():
            time.sleep(self.waittime)
            self.outputs['time'].send(np.array([True]))

    def delay_data(self, pos, data):
        if self.running():
            print("Sleeping before sending data: ", data)
            time.sleep(self.waittime)
            self.outputs['time'].send(np.array([True]))
            if (self.outputs.get('data') is not None) and (data is not None):
                self.outputs['data'].send(data)


register_node_type(Timer)
