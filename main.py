# import wfdb
# record = wfdb.rdrecord('100', pb_dir='mitdb/')
# wfdb.plot_wfdb(record=record, title='Record s25047-2704-05-04-10-44') 
# print(record.__dict__)
import csv
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter


# Processing data 
rawdata = np.loadtxt('samples.csv', str ,skiprows=2, delimiter = ',')
ecg2 = rawdata[:, 1]
timestampinit = rawdata[:, 0]
timestamp = np.ndarray(np.size(timestampinit))
for time in range(np.size(timestampinit)):
    top= timestampinit[time].strip("'")
    h,r=top.split(":")
    timeinmseconds = 60000*int(h)
    m,s=[int(j) for j in r.split(".")]
    timeinmseconds += m * 1000 
    timeinmseconds += s
    timestamp[time] = timeinmseconds

    
    


ecg = np.ndarray(np.size(ecg2))
for i in range(np.shape(ecg)[0]):
    ecg[i] = float(ecg2[i])

plt.subplot(511)
plt.title("Original ECG Signal")
plt.plot(ecg)

# Putting a band pass filter
signal_freq = 360
nyquist_freq = 0.5 * 360
lowcut = 1
highcut = 15
low = lowcut / nyquist_freq
high = highcut / nyquist_freq
b,a = butter(1, [low, high], btype="band")

data = ecg
filtered_ecg = lfilter(b,a,data)
filtered_ecg[:5] = filtered_ecg[5]
plt.subplot(512)
plt.title("Filtered Signal Output")
plt.plot(filtered_ecg)
#Differentiating
differentiated_ecg = np.ediff1d(filtered_ecg)
plt.subplot(513)
plt.title("Filtered+Differentiated Signal Output")
plt.plot(differentiated_ecg)

#Sqaurring
squarred_ecg = differentiated_ecg ** 2
plt.subplot(514)
plt.title("Filtered+Differentiated+Squared Signal Output")
plt.plot(squarred_ecg)

#moving window: convolution
convolved_ecg = np.convolve(squarred_ecg, np.ones(21))
plt.subplot(515)
plt.title("Filtered+Differentiated+Squared+Widow Integrated Signal Output")
plt.plot(convolved_ecg)
plt.tight_layout()
plt.gca().axes.get_yaxis().set_visible(False)
plt.show()
#Peak detection

data = convolved_ecg
len = data.size
spacing = 150
x = np.zeros(len + 2*spacing)
x[:spacing] = data[0] - 1.e-6
x[-spacing:] = data[-1] - 1.e-6
x[spacing: spacing+len] = data
peak_candidate = np.zeros(len)
peak_candidate[:] = True

for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

ind = np.argwhere(peak_candidate)
ind = ind.reshape(ind.size)
#threashold deciding

xn = (ecg - filtered_ecg)/(36*32)
x5 = convolved_ecg
peaki = convolved_ecg[0]
spki = 0
npki = 0
c=0
peak = [0]
threshold1 = spki
pk = []
for i in range(1,np.size(x5)-320):
    if x5[i]>peaki:
        peaki = x5[i]
                     
    npki = ((npki*(i-1))+xn[i])/i
    spki = ((spki*(i-1))+x5[i])/i
    spki = 0.875*spki + 0.125*peaki
    npki = 0.875*npki + 0.125*peaki
    
    threshold1 = npki + 0.25*(spki-npki)
    threshold2 = 0.5 * threshold1

    if(x5[i]>=threshold2):
        if(peak[-1]+24<i):
            peak.append(i)
            pk.append(x5[i])
            


xu = data[peak]
ind = ind[data[ind] >( threshold2+threshold2)/2]
detected_peak_indices = ind
detected_peaks = convolved_ecg[detected_peak_indices]
time = timestamp[detected_peak_indices]
#plt.plot(detected_peaks)
#plt.show()

heart_rate = np.ndarray(np.size(time)-1)
for i in range(1,np.size(time)):
    heart_rate[i-1] = (time[i]-time[i-1])/1000


heart_beat = 60/heart_rate

avg1 = np.average(heart_beat[-8:])
avg2 = 0
a = 8
for i in range(1,np.size(heart_beat)):
    if heart_beat[-1*i] > 55 and heart_beat[-1*i] < 105:
        a = a-1
        avg2 +=  heart_beat[-1*i]
    if a == 0:
        break
avg2 = avg2/8

low_limit = 0.92 * avg2
high_limit = 1.16*avg2
if avg1<=high_limit and avg1 >= low_limit:
    print("Normal sinus with recent average heart beat", avg1)
else:
    print("Not a Normal sinus with average heart beat", avg1)
