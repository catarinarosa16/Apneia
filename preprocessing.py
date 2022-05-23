import wfdb
import numpy as np
import biosppy.signals.tools as stools
import biosppy.signals.ecg as ecg
import scipy.signal as ss
from tqdm import tqdm
import matplotlib.pyplot as plt

def getECG(filename):
    signal = wfdb.rdrecord(filename).p_signal[:,0]
    return signal

def getLabels(filename):
    labels = wfdb.rdann(filename, extension="apn").symbol
    return labels

def baseline_correct(signal):
    signal_correct = ss.medfilt(signal, kernel_size=3)
    return signal_correct
    
def find_rpeaks(signal, fs):
    rpeaks, = ecg.hamilton_segmenter(signal, sampling_rate=fs)
    rpeaks, = ecg.correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
    return rpeaks
    
def rri_calculation(rpeaks, fs):
    rri = np.diff(rpeaks) / float(fs)
    return rri

def processingData(filename, sample, fs, labels=None):
    
    if labels is None:
        labels = getLabels(filename)
    ecg_raw = getECG(filename)
    ecg_filtered, _, _= stools.filter_signal(ecg_raw, ftype='FIR', band='bandpass', order=int(0.3 * fs), frequency=[3, 45], sampling_rate=fs)
    ecg_filtered = baseline_correct(ecg_filtered)
    ecg = []
    apn = []
    r_peaks = []
    rri = []
    hr = []
    
    for i in tqdm(range(len(labels)), desc=filename):
        signal_min = ecg_filtered[i * sample: (i + 1) * sample]
        ecg.append(signal_min)
        #signal_filtered = baseline_correct(signal_filtered)
        
        #finding R peaks and correction
        rpeaks = find_rpeaks(signal_min, fs)
        if 40 <= len(rpeaks) <= 200:    
            r_peaks.append(rpeaks)
            apn.append(0 if labels[i] == "N" else 1)
            rri_temp = rri_calculation(rpeaks, fs)
            rri.append(rri_temp)
        
        #for elem in rri_temp:
        #    rri.append(elem)
        #    hr.append(60. / elem)
    result = { 'ecg': ecg, "rpeaks": r_peaks, "rri": rri, 'apn': apn}
    return result
        
        
"""
         print("i:", i)
        value1 = int((i - before) * sample)
        value2 = int((i + 1 + after) * sample)
        print("Value: ", value1, value2)
        print("B: ", before)
        print("F: ", after)
        #print("Size: ", signal.size)
        time = np.arange(signal.size) / 100
        plt.figure(figsize=(30,4))
        plt.plot(time, signal)
        plt.xlabel("time in s")
        plt.ylabel("ECG in mV")
        plt.show()
        
        plt.figure(figsize=(30,4))
        time = np.arange(signal_filtered.size) / 100
        plt.plot(time, signal_filtered)
        plt.xlabel("time in s")
        plt.ylabel("ECG in mV")
        plt.show()
"""

"""
#print("Size: ", signal.size)
time = np.arange(unpickled_df['ecg_raw'].size) / 100
plt.figure(figsize=(30,4))
plt.plot(time, unpickled_df['ecg_raw'])
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.show()

time = np.arange(unpickled_df['ecg_filtered'].size) / 100
plt.figure(figsize=(30,4))
plt.plot(time, unpickled_df['ecg_filtered'])
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.show()
"""