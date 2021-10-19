#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:48:08 2021

@author: dominic
"""
from typing import Dict, List, Optional, Text, Tuple
from copy import copy

import numpy as np
from numpy import ndarray
from scipy.integrate import cumtrapz

from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.process_tensor import ProcessTensor
from time_evolving_mpo.bath import Bath
from time_evolving_mpo.system import BaseSystem
from time_evolving_mpo.config import NpDtype, NpDtypeReal
from time_evolving_mpo.file_formats import assert_tempo_dynamics_dict
from time_evolving_mpo.util import save_object, load_object


class TwoTimeBathCorrelations(BaseAPIClass):
    r"""
    Parameters
    ----------
    system: BaseSystem
        The system.
    bath: Bath
        The bath object containing all coupling information and temperature.
    process_tensor: ProcessTensor
        The corresponding process tensor calculated for the given bath.
    name: str (default = None)
        An optional name for the bath.
    description: str (default = None)
        An optional description of the bath.
    description_dict: dict (default = None)
        An optional dictionary with descriptive data.
    """
    def _init_(
            self,
            system: BaseSystem,
            bath: Bath,
            process_tensor: ProcessTensor,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None
            ) -> None:
        self._system = system
        self._bath = bath
        self._process_tensor = process_tensor
        
        super().__init__(name, description, description_dict)
    
    @property
    def system(self):
        return self._system
    
    @property
    def bath(self):
        return self._bath
    
    @property
    def process_tensor(self):
        return self._process_tensor
    
    

    def correlation(self,
                    freq_1: float,
                    time_1: float,
                    freq_2: Optional[float] = None,
                    time_2: Optional[float] = None,
                    dagg: Optional[tuple] = (1,0)):
        r'''
        

        Parameters
        ----------
        freq_1 : float
            Frequency of the later time operator.
        time_1 : float
            Time the later operator acts.
        freq_2 : float (default = None)
            Frequency of the earlier time operator. If set to None will default
            to freq_2=freq_1.
        time_2 : float (default = None)
            Time the earlier operator acts. If set to None will default to
            time_2=time_1.
        dagg : Optional[tuple] (default = (1,0))
            Determines whether each operator is daggered or not e.g. (1,0) 
            would correspond to < a^\dagger a >

        Returns
        -------
        times : tuple
            Pair of times of each operation.
        correlation : complex
            Bath correlation function <a^{dagg[0]}_{freq_1} (time_1) a^{dagg[1]}_{freq_2} (time_2)>
        '''
        if time_2 is None:
            time_2 = time_1
        assert time_2 <= time_1, \
            "The argument time_1 must be greater than or equal to time_2"
        if freq_2 is None:
            freq_2 = freq_1
        correlation_set = [(y,x) for y in range(int(np.floor(time_1/self.process_tensor._parameters.dt))) for x in range(y+1)]
        re_kernel,im_kernel = self._calc_kernel(freq_1,time_1,freq_2,time_2,dagg)
        
        sys_correlations = self.process_tensor.calc_correlations(self.bath.coupling_operator,correlation_set)
        
        correlation = sys_correlations.real*re_kernel+1j*sys_correlations.imag*im_kernel
        return correlation

    
    def _calc_kernel(self,
                    freq_1: float,
                    time_1: float,
                    freq_2: float,
                    time_2: float,
                    dagg: tuple):
        r'''
        Function to calculate the corresponding kernel for the desired 
        correlation function.
        
        Parameters
        ----------
        freq_1 : float
            Frequency of the later time operator.
        time_1 : float
            Time the later operator acts.
        freq_2 : float
            Frequency of the earlier time operator.
        time_2 : float
            Time the earlier operator acts.
        dagg : tuple
            Determines whether each operator is daggered or not e.g. (1,0) 
            would correspond to < a^\dagger a >

        Returns
        -------
        re_kernel : ndarray
            An array that multiplies the real part of the system correlation 
            functions before being summed.
        im_kernel : ndarray
            An array that multiplies the imaginary part of the system
            correlation functions before being summed.

        '''
        dt = self.process_tensor._parameters.dt
        
        def bose_einstein(w,T):
            if T == 0:
                return 0
            return np.exp(-w/T)/(1-np.exp(-w/T))
        def phase(i,j):
            return np.exp(-1j*((2*dagg[0]-1)*freq_1*i+(2*dagg[1]-1)*freq_2*j)*dt)
        ker_dim = int(np.floor(time_1/dt))
        switch = int(np.floor(time_2/dt))
        re_kernel = np.zeros((ker_dim,ker_dim),dtype=NpDtype)
        im_kernel = np.zeros((ker_dim,ker_dim),dtype=NpDtype)
        tpp_index,tp_index = np.meshgrid(np.arange(ker_dim),np.arange(ker_dim),
                                      indexing='ij')
        n_1,n_2 = bose_einstein(freq_1,self.bath.correlations.temperature),\
                      bose_einstein(freq_2,self.bath.correlations.temperature)
        if dagg == (0,1):
            re_kernel[:switch,:] = -phase(tp_index[:switch,:],tpp_index[:switch,:])
            re_kernel[:switch,switch:] -= phase(tpp_index[:switch,switch:],tp_index[:switch,switch:])
            
            im_kernel[:switch,:switch] = -(2*n_1+1)*phase(tpp_index[:switch,:switch],tp_index[:switch,:switch])
            im_kernel[:switch,:] += (2*n_2+1)*phase(tp_index[:switch,:],tpp_index[:switch,:])
            im_kernel[switch:,switch:] = 2*(n_2+1)*phase(tp_index[switch:,switch:],tpp_index[switch:,switch:])
            
        elif dagg == (1,0):
            re_kernel[:switch,:] = -phase(tp_index[:switch,:],tpp_index[:switch,:])
            re_kernel[:switch,switch:] -= phase(tpp_index[:switch,switch:],tp_index[:switch,switch:])
            
            im_kernel[:switch,:switch] = (2*n_1+1)*phase(tpp_index[:switch,:switch],tp_index[:switch,:switch])
            im_kernel[:switch,:] -= (2*n_2+1)*phase(tp_index[:switch,:],tpp_index[:switch,:])
            im_kernel[switch:,switch:] = -2*(n_2)*phase(tp_index[switch:,switch:],tpp_index[switch:,switch:])
            
        elif dagg == (1,1):
            re_kernel[:switch,:] = phase(tp_index[:switch,:],tpp_index[:switch,:])
            re_kernel[:switch,switch:] += phase(tpp_index[:switch,switch:],tp_index[:switch,switch:])
            
            im_kernel[:switch,:switch] = -(2*n_1+1)*phase(tpp_index[:switch,:switch],tp_index[:switch,:switch])
            im_kernel[:switch,:] -= (2*n_2+1)*phase(tp_index[:switch,:],tpp_index[:switch,:])
            im_kernel[switch:,switch:] = -2*(n_2+1)*phase(tp_index[switch:,switch:],tpp_index[switch:,switch:])
        
        elif dagg == (0,0):
            re_kernel[:switch,:] = phase(tp_index[:switch,:],tpp_index[:switch,:])
            re_kernel[:switch,switch:] += phase(tpp_index[:switch,switch:],tp_index[:switch,switch:])
            
            im_kernel[:switch,:switch] = (2*n_1+1)*phase(tpp_index[:switch,:switch],tp_index[:switch,:switch])
            im_kernel[:switch,:] += (2*n_2+1)*phase(tp_index[:switch,:],tpp_index[:switch,:])
            im_kernel[switch:,switch:] = 2*(n_2)*phase(tp_index[switch:,switch:],tpp_index[switch:,switch:])
            
        re_kernel = np.triu(re_kernel)
        im_kernel = np.triu(im_kernel)
        return re_kernel,im_kernel
        
    