def np_dfft(n):
    if n == 1:
        return( '''# Generating the DFT from scratch Function
        # Sample Signal
        import numpy as np
        import matplotlib.pyplot as plt

        # Defining the parameters for the signal
        a1, a2, a3 = 1.1, 1.28, 2.56
        theta_1, theta_2, theta_3 = 10, 20, 30
        f1, f2, f3 = 100, 160, 50

        # Creating Time for DFT, the number of time samples should be sufficient to capture all frequencies
        t = np.linspace(0, 1, 400)

        x1 = a1 * np.sin(2 * np.pi * f1 * t + theta_1)
        x2 = a2 * np.sin(2 * np.pi * f2 * t + theta_2)
        x3 = a3 * np.sin(2 * np.pi * f3 * t + theta_3)

        # Getting all values into a single signal
        x = x1 + x2 + x3

        # Plotting the obtained signals and the final combined signal
        fig, axs = plt.subplots(4, 1, figsize=(10, 6))

        axs[0].plot(t, x1)
        axs[0].set_title("Signal 1 (f = " + str(f1) + " Hz)")
        axs[0].grid(True)

        axs[1].plot(t, x2)
        axs[1].set_title("Signal 2 (f = " + str(f2) + " Hz)")
        axs[1].grid(True)

        axs[2].plot(t, x3)
        axs[2].set_title("Signal 3 (f = " + str(f3) + " Hz)")
        axs[2].grid(True)

        axs[3].plot(t, x)
        axs[3].set_title("Final Signal")
        axs[3].grid(True)

        plt.tight_layout()'''
        )
    if n==2:
        return(
            ''' 
        # DFT from Scratch -- > 
        import time



        def DFT(signal):

        N = len(signal)
        result = np.zeros(len(signal),dtype=complex)
        for k in range(N):
            X_w=0
            for n  in range(N):
            theta = -2j * np.pi* k *  (n /N)
            result[k]+= np.exp(theta)*signal[n]

        return result


        feq2 = DFT(x)
        # feq2 = np.fft.fft(x)
        # Plotting the graph
        plt.figure(figsize=(15,6))
        plt.grid(True)
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.title("DFT of the given signal ")
        start = time.time()
        plt.plot(np.fft.fftfreq(len(x),1/400),np.abs(feq2))
        end = time.time()
        print(f"This Operation Takes : {end - start} sec ")
            
        '''
        )
    if n==3:
        return(


            '''
    # Circular Convolution

    def circular_convolution(x, h):
    x1,x2,x3 = pad(x,h)

    for n in range(len(x3)):
        f = 0
        for k in range(len(x3)):
        f += x1[k]*x2[n-k]
        x3[n] = f
    return x3


    def pad(x,h):
    x1,x2 = np.array(x),np.array(h)
    Nx , Nh = len(x1),len(x2)
    # Zero Padding In Case we got inequal size
    if Nx>Nh:
        x2=np.pad(x2,(0,Nx-Nh))
        x3 = np.zeros(Nx,dtype=int)
    if Nh>Nx:
        x1=np.pad(x1,(0,Nh-Nx))
        x3 = np.zeros(Nx,dtype=int)
    else :
        x3 = np.zeros(Nx,dtype=int)
    return x1,x2,x3

    x = [1,1,1,1,0,0,0,0]
    h = [5,4,5]

    x = np.array(x)


    import time

    start = time.time()
    k = circular_convolution(x,h)
    end = time.time()
    print(f"This Operation Takes : {end - start} sec ")
    plt.scatter(np.arange(len(k)),k)
    plt.title("From Scratch Circular Convolution")

'''
        )

    if n==4:
        return (

            ''' 
import numpy as np
import matplotlib.pyplot as plt
import time
fs = 20
t = np.arange(0,500000/fs,1/fs)

# x = np.sign(np.sin(2*np.pi*t))
x = [1,-2,3,0,-1,2]


plt.figure(figsize=(15,6))
plt.xlim(0,100)
plt.grid(True)
plt.title("Squrewave")
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.plot(x)

# h1 = np.loadtxt("h.txt")
h1 = [1,1]
L= 3
M =len(h1)
x = np.array_split(x, len(x)/L)
len(h1)

h = list(h1) + list(np.zeros(L-1))
h = np.array(h)
print(x)


stt2 = time.time()
dft_h = np.fft.fft(h)
print(dft_h)
# X processing
x[0] = list(np.zeros(M-1)) + list(x[0])
# ---
for i in range(1,len(x)):
  print("runn")
  x[i] = list(list(x[i-1][len(x[i-1])-(M-1):])) + list(x[i])

print(x)


#----
dft_x = np.fft.fft(x)
Y = dft_x*dft_h
Y = np.real(Y)
Y = np.fft.ifft(Y)
Y_ = []
for i in Y:
  Y_.append(i[M-1:])
  #---
print(Y_)
end2 = time.time()
print(f"This Operation Takes : {end2 - stt2} sec ")

K = []
for i in Y_:
  K += list(i)


plt.figure(figsize=(15,6))
plt.grid(True)
plt.title("Convolution using Overlap Save ")
plt.ylabel("magnitude ")
plt.xlabel("samples")
plt.plot(K)


# FEQ
dfft = np.fft.fft(K)
freq = np.fft.fftfreq(len(K),1/20)
plt.figure(figsize=(15,6))
plt.grid(True)
plt.title("OverLap save Impulse Response")
plt.ylabel("Frequency ")
plt.xlabel("Frequency Respone of both (-ve / +ve )harmonics")
plt.plot(freq,np.abs(dfft))
print(np.fft.ifft(dfft))


# Linear Convolution
import numpy as np
import matplotlib.pyplot as plt
sample = 20
t = np.arange(0,500000/fs,1/fs)
x = np.sign(np.sin(2*np.pi*t))

plt.figure(figsize=(15,6))
plt.xlim(0,100)
plt.grid(True)
plt.title("Squrewave")
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.plot(x)

h1 = np.loadtxt("h.txt")

import time


start3 = time.time()

M = len(h1)
L = 200
fq = 20
x = np.array_split(x,len(x)/L)

N=M+L-1

# -----
pdx = np.zeros(M-1)
for i in range(len(x)):
  x[i] = list(x[i])+list(pdx)


pdh = np.zeros(L-1)
h1 = np.array(list(h1)+list(pdh))


#---

dft_x = np.fft.fft(x)
dft_h = np.fft.fft(h1)
Y = dft_x*dft_h
Y = np.real(Y)

# --
y = np.fft.ifft(Y)

# ---
end3 = time.time()
print(f"This Operation Takes {start3 - end3 } secs")
k = []
for i in y:
  k += list(i)



plt.figure(figsize=(15,6))
plt.xlim(0,100)
plt.grid(True)
plt.title("Linear Convolution")
plt.xlabel("Sample")
plt.ylabel("Magnitude")
plt.plot(k)
plt.xlim(0,500000)
'''
        )
    
    if n ==5:
        return (
            ''' 
import numpy as np
import matplotlib.pyplot as plt
import time

x = [2,2,2,2,0,0,0,0]
L = 2
M = 4

# M,N,L = len(x),len(x)+len(h)-1,len(h)

# Step 1 :  arranging the xn column wise
x = np.array(x)
x = x.reshape((L,M),order='F')
x


# Step 2 : M point DFT -->


# Making W matrix
def w(N):
  return np.round(np.exp(-1j*2*np.pi/N))
W = []

for i in range(0,4):
  for j in range(0,4):
    W.append(w(4)**(j*i))


W = np.array(W)
W = W.reshape(4,4)
print(W)
dft_x = []


# Applying DFT via manual multiplication
dft_x.append(np.dot(W,x[0]))
dft_x.append(np.dot(W,x[1]))


# Step 3 : Multipying Each Element by Wn

# Making 8x8 Wn
K= []
for i in range(0,8):
  for j in range(0,8):
    K.append(w(8)**(j*i))
K= np.array(K)
K = K.reshape(8,8)
Y = np.array(list(list(K[0][:4])+list(K[1][:4])))
Y = Y.reshape(2,4)
output = dft_x*Y
output = output.T



output


# Step 4 : Taking L point DFT
# Getting w2
w2 = []
for i in range(0,2):
  for j in range(0,2):
    w2.append(w(2)**(j*i))

w2= np.array(w2)
w2 = w2.reshape(2,2)
dft_w2  = []
dft_w2.append(np.dot(w2,output[0]))
dft_w2.append(np.dot(w2,output[1]))
dft_w2.append(np.dot(w2,output[2]))
dft_w2.append(np.dot(w2,output[3]))

dft_w2 = np.array(dft_w2)

final = dft_w2.T
final


pr =final.flatten(order='C')
apr = np.array([1,2,3,4,5,6,7,8])
print(np.fft.fft(x))


plt.figure(figsize=(15,6))
plt.stem(apr,np.abs(pr))
plt.grid(True)
plt.xlabel("Points")
plt.ylabel("Magnitude")


plt.figure(figsize=(15,6))
plt.stem(apr,np.angle(pr))
plt.grid(True)
plt.title("Phase Plot")
plt.xlabel("Points")
plt.ylabel("Phase")

'''
        )
    
    if n==6:
        return (
            ''' 


# DIT using Radix 2
# Calculating 8 point fft using RADIX-2

import numpy as np
import matplotlib.pyplot as plt
# k = input("Enter Your value space seperated : ")
# k = k.split()
# k = k[:7]
x = [1, 1, 1, 1, 2, 2, 2, 2]
x = np.array(x)
G1 = np.zeros(len(x)//2, dtype=complex)
G2 = np.zeros(len(x)//2, dtype=complex)
A = np.zeros(len(x)//4, dtype=complex)
B = np.zeros(len(x)//4, dtype=complex)
C = np.zeros(len(x)//4, dtype=complex)
D = np.zeros(len(x)//4, dtype=complex)
X = np.zeros(len(x), dtype=complex)

# Define the twiddle factor function
def w(k, m, N):
    return np.exp((-1j*2*np.pi*k*m)/N)

# Stage 1
G1[0] = x[0] + x[4]*w(0, 0, 8)#
G1[1] = x[1] + x[5]*w(0, 0, 8)#
G1[2] = x[2] + x[6]*w(0, 0, 8)#
G1[3] = x[3] + x[7]*w(0, 0, 8)#

G2[0] = x[0] - x[4]*w(0, 0, 8)#
G2[1] = (x[1] - x[5])*w(1, 1, 8)#
G2[2] = (x[2] - x[6])*w(1, 2, 8)#
G2[3] = (x[3] - x[7])*w(1, 3, 8)#

# Stage 2
A[0] = G1[2]+G1[0]
A[1] = G1[1]+G1[3]
B[0] = G1[0]-G1[2]
B[1] = (G1[1]-G1[3])*w(1, 2, 8)#


C[0] = G2[2]+G2[0]
C[1] = G2[1]+G2[3]
D[0] = (G2[0]-G2[2])*w(0, 0, 8)#
D[1] = (G2[1]-G2[3])*w(1, 2, 8)#

# Stage 3
X[0] = A[0] + A[1]
X[1] = C[0] + C[1]
X[2] = B[0] + B[1]
X[3] = D[0] + D[1]
X[4] = A[0] - A[1]
X[5] = C[0] - C[1]
X[6] = B[0] - B[1]
X[7] = D[0] - D[1]


print(X)

axies = [1,2,3,4,5,6,7,8]
# Plotting the magnitude and phase of the DFT
plt.figure(figsize=(12, 6))

plt.grid(True)
plt.stem(axies,np.abs(X))
plt.title(' DIT Magnitude Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.figure(figsize=(12, 6))
plt.grid(True)


plt.stem(axies,np.angle(X))
plt.title('Phase Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')



'''
        )
    
    if n==7:
        return (

            ''' 





# Calculating 8 point fft DIF using RADIX-2

import numpy as np
import matplotlib.pyplot as plt
# k = input("Enter Your value space seperated : ")
# k = k.split()
# k = k[:7]
x = [1, 1, 1, 1, 2, 2, 2, 2]
x = np.array(x)
G1 = np.zeros(len(x)//4, dtype=complex)
G2 = np.zeros(len(x)//4, dtype=complex)
H1 = np.zeros(len(x)//4, dtype=complex)
H2 = np.zeros(len(x)//4, dtype=complex)
F1 = np.zeros(len(x)//2, dtype=complex)
F2 = np.zeros(len(x)//2, dtype=complex)
X = np.zeros(len(x), dtype=complex)

# Define the twiddle factor function
def w(k, m, N):
    return np.exp((-1j*2*np.pi*k*m)/N)

# Stage 1
G1[0] = x[0] + x[4]*w(0, 0, 8)#
G1[1] = x[0] - x[4]*w(0, 0, 8)#
G2[0] = x[2] + x[6]*w(0, 0, 8)#
G2[1] = x[2] - x[6]*w(0, 0, 8)#
H1[0] = x[1] + x[5]*w(0, 1, 8)#
H1[1] = x[1] - x[5]*w(0, 1, 8)#
H2[0] = x[3] + x[7]*w(0, 1, 8)#
H2[1] = x[3] - x[7]*w(0, 1, 8)

# Stage 2
F1[0] = G1[0] + G2[0]*w(0, 0, 8)#
F1[1] = G1[1] + G2[1]*w(2, 1, 8)#

F1[2] = G1[0] - G2[0]*w(2, 0, 8)#
F1[3] = G1[1] - G2[1]*w(2, 1, 8)#

F2[0] = H1[0] + H2[0]*w(0, 0, 8)#
F2[1] = H1[1] + H2[1]*w(2, 1, 8)#

F2[2] = H2[0] - H2[0]*w(2, 0, 8)#
F2[3] = H2[1] - H2[1]*w(2, 1, 8)#


# Stage 3
X[0] = F1[0] + F2[0]*w(0, 0, 8)#
X[1] = F1[1] + F2[1]*w(1, 1, 8)#
X[2] = F1[2] + F2[2]*w(1, 2, 8)#
X[3] = F1[3] + F2[3]*w(3, 1, 8)#

X[4] = F1[0] - F2[0]*w(0, 4, 8)#
X[5] = F1[1] - F2[1]*w(1, 1, 8)#
X[6] = F1[2] - F2[2]*w(1, 2, 8)#
X[7] = F1[3] - F2[3]*w(3, 1, 8)#


print(X)

axies = [1,2,3,4,5,6,7,8]
# Plotting the magnitude and phase of the DFT
plt.figure(figsize=(12, 6))

plt.grid(True)
plt.stem(axies,np.abs(X))
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')


plt.figure(figsize=(12, 6))
plt.grid(True)


plt.stem(axies,np.angle(X))
plt.title('Phase Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
'''
        )
    if n==8:
        return(
            ''' 

# For testing purposes, I kept default values
f1, f2, fs, N = 100, 900, 1200, 70 # Optimal Settings
# f1, f2, fs, N = 100, 800, 1000, 111
t = np.linspace(0, 200, 260, endpoint=False)
x = np.sin(2 * np.pi * 100 * t) + 1.4*(np.sin(2*np.pi*500*t))
plt.figure(figsize=(15,6))
plt.plot(t, x)
plt.title('Original Signal')
plt.show()



def Ideal_Impulse_Response(f, x, N):
    h_i = np.sinc(2 * f * (x - (N / 2)))
    return h_i

def LOW_PASS(f1=f1, fs=fs, N=N, x=x):

    fn = 2 * (f1 / fs)
    h_l = Ideal_Impulse_Response(fn, np.arange(N), N)
    return h_l

def HIGH_PASS(f1=f1, fs=fs, N=N, x=x):


    h_l = LOW_PASS(f1=f1, fs=fs, N=N, x=x)
    h_hp = -h_l
    h_hp[N // 2] += 1

    return h_hp

def BAND_PASS(f1=f1, f2=f2, fs=fs, N=N, x=x):
    h_l = LOW_PASS(f1=f1, fs=fs, N=N, x=x)
    h_h = HIGH_PASS(f1=f2, fs=fs, N=N, x=x)
    return h_l + h_h

def BAND_REJECT(f1=f1, f2=f2, fs=fs, N=N, x=x):
    h_l = LOW_PASS(f1=f1, fs=fs, N=N, x=x)
    h_h = HIGH_PASS(f1=f2, fs=fs, N=N, x=x)
    return h_l -  h_h

def APPLY_FILTER(x,fitler_keranal):
    return np.convolve(x,fitler_keranal,mode="same")



lp = LOW_PASS()
hp = HIGH_PASS()
bp = BAND_PASS()
br = BAND_REJECT()

lp_signal = APPLY_FILTER(x, lp)
hp_signal = APPLY_FILTER(x, hp)
bp_signal = APPLY_FILTER(x, bp)
br_signal = APPLY_FILTER(x, br)

N=len(x)
def RECT_WINDOW(N):
    return np.ones(N)

def HANNING_WINDOW(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

def HAMMING_WINDOW(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

def BLACKMAN_WINDOW(N):
    return 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

b_r = RECT_WINDOW(N) * br_signal
b_h = HANNING_WINDOW(N) * br_signal
b_ha = HAMMING_WINDOW(N) * br_signal
b_b = BLACKMAN_WINDOW(N) * br_signal


def gain_conversion(signal,fs=fs):
  fft = np.fft.fft(signal,2024)
  fft = np.abs(fft)
  fft_n = fft/max(fft)
  fft_freq = freq = np.fft.fftfreq(len(fft_n), 1/fs)
  gain = 20*np.log10(np.abs(fft_n)+ 1e-6)

  return gain,fft_freq

lpfg , lpff = gain_conversion(lp)
hpfg , hpf = gain_conversion(hp)
bpfg , bpf = gain_conversion(bp)
brfg , brf = gain_conversion(br)





'''
        )


            




