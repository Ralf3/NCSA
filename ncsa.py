#!/usr/bin/env python

import sys
import argparse
from scipy import signal
from struct import pack
import socket
import struct
import numpy as np
import pylab as plt

class connection(object):
    "define the ip addr and the port "
    def __init__(self,ip,port):
        self.ip=ip
        self.port=port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.ip, self.port))
        
    def mysend(self, msg):
        totalsent = 0
        sent = self.s.send(msg)
        if sent == 0:
            raise RuntimeError("socket connection broken")
        return True

    def myreceive(self,MSGLEN):
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.s.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == '':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return ''.join(chunks)
    
    def close(self):
        self.s.shutdown(1)
	self.s.close()
	return True

class device(object):
    def __init__(self,ip='192.168.1.123',port=4000):
        self.s=connection(ip,port)
        self.AD12B=1  # 12 bit
        self.ADCS=6
        self.scale=4095.0
        self.FS=504201 # default FS
        self.M=60
        self.N1=2
        self.N2=2
        self.F=60000000
    def command(self,N=4096,FS=10000,Mode=12):
        """ N=4096 data points
            FS=10kHz
            Mode=12bit
        """
        if(Mode==12):
            self.AD12B=1
            self.ADCS=6
            K=int(self.F / FS) / (self.ADCS + 1) - 17
            if(K==-17):
                F=self.FS
            else:
                F=int(self.F / (17 + K) / (self.ADCS + 1))
        else:
            self.AD12B=0
            self.ADCS=4
            self.scale=1023.0
            K=int(self.F / FS) / (self.ADCS + 1) - 14
            if(K==-14):
                F=self.FS
            else:
                F=int(self.F / (14 + K) / (self.ADCS + 1))
        self.FS=F
        fx1=F       & 0xff
        fx2=(F>>8)  & 0xff
        fx3=(F>>16) & 0xff
        N1=N & 0xff
        N2=(N>>8) & 0xff
        command=pack('8B',1,N1,N2,fx1,fx2,fx3,self.AD12B,self.ADCS)
        print N1,N2,fx1,fx2,fx3,self.AD12B,self.ADCS
        self.s.mysend(command)
        data=self.s.myreceive(2*N)   # 12 Bit splittet in two bytes
        print 'received: ',len(data)
        return data
    def convert(self,data):
        """ converts the bytes to an numpy array
        """
        N=int(len(data)/2)
        d=np.zeros(N)
        x='%dB' % (2*N)
        data=struct.unpack(x,data)
        for i in range(N):
            d[i]=float(int(data[2*i]) | int(data[2*i+1])<<8)*3.3/self.scale
        return d
    def freq(self,fx,T=1):
        """ T: sin=1, square=2, triangle=3"""
        fx5=(fx>>16) & 0xff
        fx4=(fx>>8)  & 0xff
        fx3=fx & 0xff
        #print fx,fx5,fx4,fx3
        command=pack('8B',2,0,0,fx3,fx4,fx5,0,T)
        self.s.mysend(command)
        # print command
        return True
    def stop_gen(self):
        command=pack('8B',3,0,0,0,0,0,0,0)
        self.s.mysend(command)
        print command
        return True
    def noise(self):
        command=pack('8B',4,0,0,0,0,0,0,0)
        self.s.mysend(command)
        print command
        return True
    def plot(self,data):
        N=len(data)
        print N,self.FS,(float(N)/self.FS)
        #print data[0:100]
        x=np.linspace(0,(float(N)/self.FS),N) # step=N*T T=1/FS
        plt.title('FS: '+str(self.FS)+' N: '+str(N))
        plt.plot(x,data[0:N])
        plt.xlabel("t [s]")
        plt.ylabel("U [V]")
        plt.grid(True)
        plt.show()
    def plot_fft(self,data,window=0):
        """ window for fft:
            0 : without
            1 : hanning
            2 : hamming
            3 : kaiser
            4 : bartlett
            5 : blackmann
        """
        data-=np.mean(data)
        N=len(data)
        h=np.ones(N)
        if(window==1):
            h=np.hanning(N)
        if(window==2):
            h=np.hamming(N)
        if(window==3):
            h=np.kaiser(N,14)
        if(window==4):
            h=np.bartlett(N)
        if(window==5):
            h=np.blackman(N)
        data=data*h
        mag = np.abs(np.fft.fft(data))
        print len(mag),
        step=self.FS/float(N)
        print step, N/2*step
        freq = np.arange(0,N/2*step,step)
        response=20*np.log10(mag)
        response-=np.max(response)
        response = np.clip(response, -100, 100)
        #print response[0:100]
        plt.title('FS: '+str(self.FS)+' N: '+str(N))
        plt.plot(freq, response[0:N/2])
        plt.xlabel('Freq')
        plt.ylabel('M [db]')
        plt.grid(True)
        plt.show()
    def plot_fft1(self,data,window=0):
        """ window for fft:
        0 : without
        1 : hanning
        2 : hamming
        3 : kaiser
        4 : bartlett
        5 : blackmann
        """
        N=len(data)
        print N,self.FS,(float(N)/self.FS)
        x=np.linspace(0,(float(N)/self.FS),N) # step=N*T T=1/FS
        # define the multiple plot
        plt.subplot(2,1,1)
        plt.title('FS: '+str(self.FS)+' N: '+str(N))
        plt.plot(x,data[0:N])
        plt.xlabel("t [s]")
        plt.ylabel("U [V]")
        plt.grid(True)
        data-=np.mean(data)
        N=len(data)
        h=np.ones(N)
        if(window==1):
            h=np.hanning(N)
        if(window==2):
            h=np.hamming(N)
        if(window==3):
            h=np.kaiser(N,14)
        if(window==4):
            h=np.bartlett(N)
        if(window==5):
            h=np.blackman(N)
        data=data*h
        mag = np.abs(np.fft.fft(data))
        print len(mag),
        step=self.FS/float(N)
        print step, N/2*step
        freq = np.arange(0,N/2*step,step)
        response=20*np.log10(mag)
        response-=np.max(response)
        response = np.clip(response, -100, 100)
        plt.subplot(2,1,2)
        plt.plot(freq, response[0:N/2])
        plt.xlabel('Freq')
        plt.ylabel('M [db]')
        plt.grid(True)
        plt.show()
    def plot_wave(self,data,ws=32):
        N=len(data)
        dt=1.0/self.FS/8
        T=N*dt
        t = np.linspace(T, -dt, 0, endpoint=False)
        df=self.FS/(2*2) # fs/2 and ricker 8
        print t
        widths = np.arange(1, ws)
        cwtmatr = signal.cwt(data, signal.ricker, widths)
        mean=np.mean(cwtmatr)
        # cwtmatr[cwtmatr[:,:]>3*mean]=0
        cax=plt.imshow(cwtmatr,aspect='auto')
        cbar = plt.colorbar(cax)
        plt.title('FS: '+str(self.FS)+' N: '+str(N))
        plt.xlabel('dt [%fs]' %(dt))
        plt.ylabel('dF [%dHz]' %(int(np.round(df))))
	plt.grid(True)
        plt.show()
    def plot_wave1(self,data,ws=32):
        N=len(data)
        print N,self.FS,(float(N)/self.FS)
        x=np.linspace(0,(float(N)/self.FS),N) # step=N*T T=1/FS
        # define the multiple plot
        plt.subplot(2,1,1)
        plt.title('FS: '+str(self.FS)+' N: '+str(N))
        plt.plot(x,data[0:N])
        plt.xlabel("t [s]")
        plt.ylabel("U [V]")
        plt.grid(True)
        # wavlet part
        dt=1.0/self.FS
        T=N*dt
        df=self.FS/(2*2)
        t = np.linspace(T, -dt, 0, endpoint=False)
        widths = np.arange(1, ws)
        cwtmatr = signal.cwt(data, signal.ricker, widths)
        mean=np.mean(cwtmatr)
        # cwtmatr[cwtmatr[:,:]>3*mean]=0
        # print cwtmatr
        plt.subplot(2,1,2)
        cax=plt.imshow(cwtmatr,aspect='auto')
        cbar = plt.colorbar(cax)
        plt.xlabel('dt [%fs]' %(dt))
        plt.ylabel('dF [%dHz]' %(int(np.round(df))))
	plt.grid(True)
        plt.show()


def main():
    parser=argparse.ArgumentParser(description='User interface for NCSA')
    parser.add_argument('--c', choices=['sin', 'square', 'triangle',
                                        'noise', 'stop',
                                        'plot','fft', 'plot_fft',
                                        'wave','plot_wave'],
                        required=True,
                        help='main generator functions and analyze functions')
    parser.add_argument('--fs', type=int, default=1000, help='Sample frequency')
    parser.add_argument('--mode', choices=[10,12], default=12,
                        help='Mode: [10|12] default=12 Bit')
    parser.add_argument('--window',choices=['rectangular','hanning','hamming',
                                            'kaiser','bartlett','blackman'],
                        default='hanning',
                        help='window function for FFT')
    parser.add_argument('--N',type=int, choices=[256,512,1024,2048,4096,8192],
                        default=1024,
                        help='Sample size N: 1024,2048,4095,8192')
    parser.add_argument('--ws', type=int, default=32,
                        help='frequency range for wavelet transformation 10:100')
    
    args=parser.parse_args()
    print args
    # transformation between names and controls
    # transformation between names and controls
    controls={'rectangular' : 0,  
              'hanning' : 1,
              'hamming' : 2,
              'kaiser'  : 3,
              'bartlett': 4,
              'blackman': 5}
    # main loop to start the program
    r=device() # open a connection to the NCSA
    # select the generator functions to call
    if(args.c=='sin'):
        r.freq(args.fs,1) # sin generator
        return            # end the process
    if(args.c=='square'):
        r.freq(args.fs,2) # square generator
        return            # end the process
    if(args.c=='triangle'):
        r.freq(args.fs,3) # triangular generator
        return            # end the process
    if(args.c=='noise'):
        r.noise()         # start the noise generator
        return            # end the process
    if(args.c=='stop'):
        r.stop_gen()      # stop the generator
        return            # end the process
    # select the anlyze functions
    if(args.c=='plot'):
        data=r.command(N=args.N,FS=args.fs,Mode=args.mode)
        data=r.convert(data)
        r.plot(data)
        return
    if(args.c=='fft'):
        data=r.command(N=args.N,FS=args.fs,Mode=args.mode)
        data=r.convert(data)
        r.plot_fft(data,window=controls[args.window])
        return
    if(args.c=='plot_fft'):
        data=r.command(N=args.N,FS=args.fs,Mode=args.mode)
        data=r.convert(data)
        r.plot_fft1(data,window=controls[args.window])
        return
    if(args.c=='wave'):
        data=r.command(N=args.N,FS=args.fs,Mode=args.mode)
        data=r.convert(data)
        r.plot_wave(data,ws=args.ws)
        return
    if(args.c=='plot_wave'):
        data=r.command(N=args.N,FS=args.fs,Mode=args.mode)
        data=r.convert(data)
        r.plot_wave1(data,ws=args.ws)
        return
main()

    

      
