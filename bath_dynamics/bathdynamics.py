# Copyright 2021 The TEMPO Collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module for calculating bath dynamics as outlined in [Gribben2021].

**[Gribben2021]**
D. Gribben, A. Strathearn, G. E. Fux, P. Kirton, and B. W. Lovett,
*Using the Environment to Understand non-Markovian Open Quantum Systems*, 
arXiv:2106.04212 [quant-ph] (2021).
"""
from typing import Dict, Optional, Text
import numpy as np
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.process_tensor import ProcessTensor
from time_evolving_mpo.bath import Bath
from time_evolving_mpo.system import BaseSystem
from time_evolving_mpo.config import NpDtype


class TwoTimeBathCorrelations(BaseAPIClass):
    """
    Class to facilitate calculation of two-time bath correlations.
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
    def __init__(
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
        self._system_correlations = np.array([[]],dtype=NpDtype)
        super().__init__(name, description, description_dict)
    @property
    def system(self):
        """
        System Hamiltonian
        """
        return self._system
    @property
    def bath(self):
        """
        Bath properties
        """
        return self._bath
    def correlation(self,
                    freq_1: float,
                    time_1: float,
                    freq_2: Optional[float] = None,
                    time_2: Optional[float] = None,
                    dagg: Optional[tuple] = (1,0)):
        r"""
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
            Bath correlation function
            <a^{dagg[0]}_{freq_1} (time_1) a^{dagg[1]}_{freq_2} (time_2)>
        """
        dt = self._process_tensor.times[1]-self._process_tensor.times[0]
        corr_mat_dim = time_1/dt
        current_corr_dim = self._system_correlations.shape[0]
        if time_2 is None:
            time_2 = time_1
        assert time_2 <= time_1, \
            "The argument time_1 must be greater than or equal to time_2"
        if freq_2 is None:
            freq_2 = freq_1
        correlation_set = [(x,y) for y in range(current_corr_dim,corr_mat_dim)\
                           for x in range(y+1)]
        re_kernel,im_kernel = self._calc_kernel(freq_1,time_1,
                                                freq_2,time_2,dagg)
        if len(correlation_set>0):
            dim_diff = corr_mat_dim-self._system_correlations.shape[0]
            coup_op = self.bath.coupling_operator
            _new_sys_correlations = \
                self._process_tensor.calc_correlations(coup_op,correlation_set)
            self._system_correlations = np.pad(self._system_correlations,
                                               ((0,dim_diff),
                                                (0,dim_diff)))
            for n,i in enumerate(correlation_set):
                self._system_correlations[i] = _new_sys_correlations[n]

        _sys_correlations = self._system_correlations[:corr_mat_dim,
                                                      :corr_mat_dim]
        correlation = np.sum(_sys_correlations.real*re_kernel+\
                             +1j*_sys_correlations.imag*im_kernel)*dt**2
        return correlation

    def _calc_kernel(self,
                    freq_1: float,
                    time_1: float,
                    freq_2: float,
                    time_2: float,
                    dagg: tuple):
        r"""
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

        """
        dt = self._process_tensor.times[1]-self._process_tensor.times[0]
        def bose_einstein(w,temp):
            if temp == 0:
                return 0
            return np.exp(-w/temp)/(1-np.exp(-w/temp))
        def phase(i,j):
            ph = np.exp(-1j*((2*dagg[0]-1)*freq_1*i+(2*dagg[1]-1)*freq_2*j)*dt)
            return ph
        ker_dim = int(np.floor(time_1/dt))
        switch = int(np.floor(time_2/dt))
        re_kernel = np.zeros((ker_dim,ker_dim),dtype=NpDtype)
        im_kernel = np.zeros((ker_dim,ker_dim),dtype=NpDtype)
        tpp_index,tp_index = np.meshgrid(np.arange(ker_dim),np.arange(ker_dim),
                                      indexing='ij')
        n_1,n_2 = bose_einstein(freq_1,self.bath.correlations.temperature),\
                      bose_einstein(freq_2,self.bath.correlations.temperature)
        if dagg == (0,1):
            re_kernel[:switch,:] = -phase(tp_index[:switch,:],
                                          tpp_index[:switch,:])
            re_kernel[:switch,switch:] -= phase(tpp_index[:switch,switch:],
                                                tp_index[:switch,switch:])
            im_kernel[:switch,:switch] = -(2*n_1+1)*phase(tpp_index[:switch,
                                                                    :switch],
                                                          tp_index[:switch,
                                                                   :switch])
            im_kernel[:switch,:] += (2*n_2+1)*phase(tp_index[:switch,:],
                                                    tpp_index[:switch,:])
            im_kernel[switch:,switch:] = 2*(n_2+1)*phase(tp_index[switch:,
                                                                  switch:],
                                                         tpp_index[switch:,
                                                                   switch:])
        elif dagg == (1,0):
            re_kernel[:switch,:] = -phase(tp_index[:switch,:],
                                          tpp_index[:switch,:])
            re_kernel[:switch,switch:] -= phase(tpp_index[:switch,switch:],
                                                tp_index[:switch,switch:])
            im_kernel[:switch,:switch] = (2*n_1+1)*phase(tpp_index[:switch,
                                                                   :switch],
                                                         tp_index[:switch,
                                                                  :switch])
            im_kernel[:switch,:] -= (2*n_2+1)*phase(tp_index[:switch,:],
                                                    tpp_index[:switch,:])
            im_kernel[switch:,switch:] = -2*(n_2)*phase(tp_index[switch:,
                                                                 switch:],
                                                        tpp_index[switch:,
                                                                  switch:])
        elif dagg == (1,1):
            re_kernel[:switch,:] = phase(tp_index[:switch,:],
                                         tpp_index[:switch,:])
            re_kernel[:switch,switch:] += phase(tpp_index[:switch,switch:],
                                                tp_index[:switch,switch:])
            im_kernel[:switch,:switch] = -(2*n_1+1)*phase(tpp_index[:switch,
                                                                    :switch],
                                                          tp_index[:switch,
                                                                   :switch])
            im_kernel[:switch,:] -= (2*n_2+1)*phase(tp_index[:switch,:],
                                                    tpp_index[:switch,:])
            im_kernel[switch:,switch:] = -2*(n_2+1)*phase(tp_index[switch:,
                                                                   switch:],
                                                          tpp_index[switch:,
                                                                    switch:])
        elif dagg == (0,0):
            re_kernel[:switch,:] = phase(tp_index[:switch,:],
                                         tpp_index[:switch,:])
            re_kernel[:switch,switch:] += phase(tpp_index[:switch,switch:],
                                                tp_index[:switch,switch:])
            im_kernel[:switch,:switch] = (2*n_1+1)*phase(tpp_index[:switch,
                                                                   :switch],
                                                         tp_index[:switch,
                                                                  :switch])
            im_kernel[:switch,:] += (2*n_2+1)*phase(tp_index[:switch,:],
                                                    tpp_index[:switch,:])
            im_kernel[switch:,switch:] = 2*(n_2)*phase(tp_index[switch:,
                                                                switch:],
                                                       tpp_index[switch:,
                                                                 switch:])
        re_kernel = np.triu(re_kernel)
        im_kernel = np.triu(im_kernel)
        return re_kernel,im_kernel
    