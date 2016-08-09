NCSA: A simple program to control the Elektor:
Network Connected Signal Analyzer with Python
==============================================================================

https://www.elektormagazine.de/news/ncsa-network-connected-signal-analyzer-von-elektor

==============================================================================

The NCSA can be used as a signal generator for sine, square or triangular waves.
More important is use of the NCSA as input to sample data. The maximum sample
frequency is 1MHz and the sample resolution can be 10Bit or 12Bit.

The NCSA is connected via Ethernet and can be controlled using a simple
open source protocol. This protocol was implemented in Python. The Python
open source software is provided as a command line tool:

usage: ncsa.py [-h] --c
               {sin,square,triangle,noise,stop,plot,fft,plot_fft,wave,plot_wave}
               [--fs FS] [--mode {10,12}]
               [--window {rectangular,hanning,hamming,kaiser,bartlett,blackman}]
               [--N {256,512,1024,2048,4096,8192}] [--ws WS]

User interface for NCSA

optional arguments:
  -h, --help            show this help message and exit
  --c                   {sin,square,triangle,noise,stop,plot,
  			 fft,plot_fft,wave,plot_wave}
                        main generator functions and analyze functions
  --fs FS               Sample frequency
  --mode                {10,12}        Mode: [10|12] default=12 Bit
  --window              {rectangular,hanning,hamming,kaiser,bartlett,blackman}
                        window function for FFT
  --N                   {256,512,1024,2048,4096,8192}
                        Sample size N: 1024,2048,4095,8192
  --ws WS               frequency range for wavelet transformation 10:100

It should be remarked that the software runs under Linux (Ubuntu 16.04 LTS but
should run under other Linux systems too).