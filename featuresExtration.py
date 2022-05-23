import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import warnings
from sklearn.decomposition import PCA
from hrv.rri import RRi
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_poincare_plot_features, get_csi_cvi_features, get_nn_intervals
import neurokit2 as nk
from scipy.signal import detrend, find_peaks


warnings.filterwarnings(action="ignore")

def EDR_all(ecgs, rpeaks_idxs, peak_count_nz):
    EDR_list = []
    for i in range(rpeaks_idxs.shape[0]):
        
        cs_rpeak = interp1d(rpeaks_idxs[i], ecgs[i][rpeaks_idxs[i]])
        min_x = np.min(rpeaks_idxs[i])
        max_x = np.max(rpeaks_idxs[i])
        n = np.max(peak_count_nz)
        t = np.linspace(min_x, max_x, n)
        edr_ = cs_rpeak(t)
        edr_ -= np.mean(edr_) #center the signal
        #edr_, _ = edr(ecgs[i], rpeaks_idxs[i])
        EDR_list.append(edr_)
    EDR = np.array(EDR_list)
    return EDR
def iqr(x):
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    return iqr

def mad(x):
    mean_x = np.mean(x)
    mean_adj = np.abs(x - mean_x)
    return np.mean(mean_adj)
def sample_entropy(a):
    return np.abs(a[2] - a).max(axis=0)


def _calculate_features(file, filename):
    df = pd.DataFrame()
    
    features = {}
    rri = file["rri"]
    
    for i in tqdm(range(len(rri)), desc=filename, file=sys.stdout):
        rri_min = RRi(rri[i])
        hr_min = rri_min.to_hr()

        
        if np.all(np.logical_and(hr_min >= 20, hr_min <= 200)):
            ecg = file["ecg"][i]
            #EDR calculo
            edr = detrend(hr_min)
            #edr = (edr - edr.mean()) / edr.std()
            edr_peaks,_ = find_peaks(edr, height=0, distance=4)
            edr_peaks = edr_peaks
            resp_peaks_diff = np.diff(edr_peaks) / 4
            
            mean_resp = edr_peaks.size / 60

            #rri_min = get_nn_intervals(rri[i],ectopic_beats_removal_method='malik',verbose=True)
            apn = file["apn"][i]
            rri_time_features, rri_frequency_features, = get_time_domain_features(rri_min), get_frequency_domain_features(rri_min)
            non_linear_features1, non_linear_features2 = get_poincare_plot_features(rri_min), get_csi_cvi_features(rri_min)
            features.update(
                {
                    #time
                    "mean_nni": rri_time_features["mean_nni"],
                    "rmssd": rri_time_features["rmssd"],
                    "sdnn": rri_time_features["sdnn"],
                    "sdsd": rri_time_features["sdsd"],
                    "nni_50": rri_time_features["nni_50"],
                    "pnni_50": rri_time_features["pnni_50"],
                    "nni_20": rri_time_features["nni_20"],
                    "pnni_20": rri_time_features["pnni_20"],
                    "range_nni": rri_time_features["range_nni"],
                    "median_nni": rri_time_features["median_nni"],
                    "mean_hr": rri_time_features["mean_hr"],
                    "cvsd": rri_time_features["cvsd"],
                    "cvnni": rri_time_features["cvnni"],
                    "max_hr": rri_time_features["max_hr"],
                    "min_hr": rri_time_features["min_hr"],
                    "std_hr": rri_time_features["std_hr"],
                    #frequency
                    "vlf": rri_frequency_features["vlf"] / rri_frequency_features["total_power"],
                    "lf": rri_frequency_features["lf"] / rri_frequency_features["total_power"],
                    "hf": rri_frequency_features["hf"] / rri_frequency_features["total_power"],
                    "lf_hf": rri_frequency_features["lf_hf_ratio"],
                    "lfnu": rri_frequency_features["lfnu"],
                    "hfnu": rri_frequency_features["hfnu"],
                    #non-linear
                    "sd1": non_linear_features1["sd1"],
                    "sd2": non_linear_features1["sd2"],
                    "sd1sd2": non_linear_features1["ratio_sd2_sd1"],
                    "csi": non_linear_features2["csi"],
                    "cvi": non_linear_features2["cvi"],
                    "Modified_csi":non_linear_features2["Modified_csi"],
                    #EDR feactures
                    "edr_mean_rate": mean_resp,
                    "edr_mean_period": (1 / mean_resp),
                    "resp_RMS": np.sqrt(np.mean(resp_peaks_diff**2)),
                    "resp_STD": np.std(resp_peaks_diff),
                    "apn": apn
                }
            )
            df = df.append(features, ignore_index=True)
    return df


"""
from hrv.classical import time_domain, frequency_domain, non_linear
from hrv.rri import RRi

                #"md": md,
                "rmssd": rri_time_features["rmssd"],
                "sdnn": rri_time_features["sdnn"],
                "nn50": rri_time_features["nn50"],
                "pnn50": rri_time_features["pnn50"],
                "mrri": rri_time_features["mrri"],
                "mhr": rri_time_features["mhr"],
                "vlf": rri_frequency_features["vlf"] / rri_frequency_features["total_power"],
                "lf": rri_frequency_features["lf"] / rri_frequency_features["total_power"],
                "hf": rri_frequency_features["hf"] / rri_frequency_features["total_power"],
                "lf_hf": rri_frequency_features["lf_hf"],
                "lfnu": rri_frequency_features["lfnu"],
                "hfnu": rri_frequency_features["hfnu"], 
                "apn": apn
                
                #
                "mean_nni": rri_time_features["mean_nni"],
                "rmssd": rri_time_features["rmssd"],
                "sdnn": rri_time_features["sdnn"],
                "sdsd": rri_time_features["sdsd"],
                "nni_50": rri_time_features["nni_50"],
                "pnni_50": rri_time_features["pnni_50"],
                "nni_20": rri_time_features["nni_20"],
                "pnni_20": rri_time_features["pnni_20"],
                "range_nni": rri_time_features["range_nni"],
                "median_nni": rri_time_features["median_nni"],
                "mean_hr": rri_time_features["mean_hr"],
                "cvsd": rri_time_features["cvsd"],
                "cvnni": rri_time_features["cvnni"],
                "max_hr": rri_time_features["max_hr"],
                "min_hr": rri_time_features["min_hr"],
                "std_hr": rri_time_features["std_hr"],
                #frequency
                "vlf": rri_frequency_features["vlf"] / rri_frequency_features["total_power"],
                "lf": rri_frequency_features["lf"] / rri_frequency_features["total_power"],
                "hf": rri_frequency_features["hf"] / rri_frequency_features["total_power"],
                "lf_hf": rri_frequency_features["lf_hf_ratio"],
                "lfnu": rri_frequency_features["lfnu"],
                "hfnu": rri_frequency_features["hfnu"],
                "apn": apn
                
                
                "HRV_RMSSD": hrv_indices["HRV_RMSSD"],
                "HRV_MeanNN": hrv_indices["HRV_MeanNN"],
                "HRV_SDNN": hrv_indices["HRV_SDNN"],
                "HRV_SDSD": hrv_indices["HRV_SDSD"],
                "HRV_CVNN": hrv_indices["HRV_CVNN"],
                "HRV_CVSD": hrv_indices["HRV_CVSD"],
                "HRV_MedianNN": hrv_indices["HRV_MedianNN"],
                "HRV_MadNN": hrv_indices["HRV_MadNN"],
                "HRV_MCVNN": hrv_indices["HRV_MCVNN"],
                "HRV_pNN50": hrv_indices["HRV_pNN50"],
                "HRV_LFn": hrv_indices["HRV_LFn"],
                "HRV_HFn": hrv_indices["HRV_HFn"],
                "HRV_LnHF": hrv_indices["HRV_LnHF"],
                "HRV_SD1": hrv_indices["HRV_SD1"],
                "HRV_SD2": hrv_indices["HRV_SD2"],
                "SD1SD2": hrv_indices["HRV_SD2SD1"],
                "HRV_CSI": hrv_indices["HRV_CSI"],
                "HRV_CVI": hrv_indices["HRV_CVI"],
                "HRV_CSI_Modified": hrv_indices["HRV_CSI_Modified"],
                "HRV_CVI_Modified": hrv_indices["HRV_CVI_Modified"],
                "HRV_SampEn": hrv_indices["HRV_SampEn"],
"""