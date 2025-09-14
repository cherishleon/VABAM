import numpy as np
from scipy.stats import mode
import itertools
from tqdm import trange, tqdm
import gc
import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utilities.Utilities import GenBatches

from scipy import signal
from scipy.fftpack import dct

def compute_snr(signal, noisy_signal):
    """Computes SNR in dB for a 2D batch signal."""
    noise = noisy_signal - signal  # Extract noise
    signal_power = np.mean(signal ** 2, axis=1)
    noise_power = np.mean(noise ** 2, axis=1)
    return 10 * np.log10(signal_power / noise_power)  # SNR in dB

def scale_and_normalize(data, sigma, mean, min_x, max_x):
    """
    Scales and normalizes the input data.
      """
    # Apply scaling
    scaled_data = data * sigma + mean
    
    # Apply normalization
    normalized_data = (scaled_data - min_x) / (max_x - min_x)
    
    return normalized_data
    
# For the dimensional Kullback-Leibler Divergence of the Z distribution
def LogNormalDensity(LatSamp, LatMean, LogSquaScale):
    Norm = tf.math.log(2. * tf.constant(np.pi))
    InvSigma = tf.math.exp(-LogSquaScale)
    MeanSampDiff = (LatSamp - LatMean)
    return -0.5 * (MeanSampDiff * MeanSampDiff * InvSigma + LogSquaScale + Norm)


# For Factor-VAE
def SplitBatch (Vec, HalfBatchIdx1, HalfBatchIdx2, mode='Both'):
    
    HalfBatch1 = tf.nn.embedding_lookup(Vec, HalfBatchIdx1)
    HalfBatch2 = tf.nn.embedding_lookup(Vec, HalfBatchIdx2)
    
    if mode=='Both':
        return  HalfBatch1, HalfBatch2
    elif mode=='D1':
        return  HalfBatch1
    elif mode=='D2':
        return  HalfBatch2
 

# Power spectral density 
def FFT_PSD(Data, ReducedAxis, MinFreq=1, MaxFreq=51, 
           method='matching_pursuit', nperseg=None, window='hann', 
           preserve_dims=False, return_phase=False):
    """
    Power Spectral Density calculation with configurable methods.

    Parameters
    ----------
    Data : array_like
        Input data tensor (1D to 4D supported).
    ReducedAxis : str
        'None', 'All', or 'Batch' - how to reduce dimensions.
    MinFreq : int, default=1
        Minimum frequency index (inclusive).
    MaxFreq : int, default=51
        Maximum frequency index (exclusive).
    method : str, default='matching_pursuit'
        'fft' (original), 'welch' (variance-reduced), 'matching_pursuit' (sparse DCT),
        or 'welch_evo' (time-varying STFT power averaged over time).
    nperseg : int, optional
        Segment length for Welch/STFT-based methods.
    window : str, default='hann'
        Window function for Welch/STFT-based methods.
    preserve_dims : bool, default=False
        If True, maintain input dimensions when reducing.
    return_phase : bool, default=False
        If True, also return phase. For 'fft' the phase is from the complex FFT.
        For 'welch_evo' the phase is the circular mean of STFT phases over time.
        For 'welch' and 'matching_pursuit', phase is computed from the complex FFT
        of the input in the selected frequency band.

    Returns
    -------
    AggPSPDF : array_like
        Normalized power spectral density (PDF along the last axis).
    phase_result : array_like, optional
        Phase array corresponding to the selected band and reduction rule when
        return_phase=True.
    """
    original_shape = Data.shape
    Data = Data[:, None] if len(Data.shape) < 3 else Data

    # Utility: circular mean of phase
    def _circ_mean(ph, axis=None, keepdims=False):
        return np.angle(np.mean(np.exp(1j * ph), axis=axis, keepdims=keepdims))

    phase_raw = None  # will be set only if return_phase is True

    if method in ("sparse_dct", "matching_pursuit"):
        # Matching Pursuit (MP) is a greedy algorithm: at each iteration it selects
        # the atom with maximal inner product with the current residual and updates the residual.
        # When the dictionary is orthonormal (e.g., the DCT-II basis), k-step MP is
        # equivalent to keeping the k largest transform coefficients (hard-thresholding)
        # in that basis. Thus, this branch implements DCT hard-thresholding (sparse DCT),
        # which can be viewed as a restricted special case of MP under an orthonormal dictionary.
        # It is NOT general MP with redundant/overcomplete dictionaries.
        # Reference: Mallat, S. G., & Zhang, Z. (1993). “Matching Pursuits with Time-Frequency
        # Dictionaries.” IEEE Transactions on Signal Processing, 41(12), 3397–3415.
        # DOI: 10.1109/78.258082
        sig_len = Data.shape[-1]
    
        X = Data.reshape(-1, sig_len)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    
        C = dct(X, type=2, norm="ortho", axis=-1)
    
        band_width = max(1, int(MaxFreq) - int(MinFreq))
        k = min(sig_len, max(128, 2 * band_width))
    
        num_coeff = C.shape[1]
        if k < num_coeff:
            kill = np.argpartition(np.abs(C), num_coeff - k, axis=1)[:, : num_coeff - k]
            C[np.arange(C.shape[0])[:, None], kill] = 0.0
    
        HalfLen = sig_len // 2
        m0 = max(0, int(MinFreq))
        m1 = min(int(MaxFreq), HalfLen)
        if m1 <= m0:
            m1 = min(m0 + 1, HalfLen)
    
        psd_slice = (C[:, :HalfLen] ** 2)[:, m0:m1]
        psd_slice = np.nan_to_num(psd_slice, nan=0.0, posinf=0.0, neginf=0.0)
        psd_slice = np.maximum(psd_slice, 0.0)
        PSD = psd_slice.reshape(Data.shape[:-1] + (-1,))

    elif method == 'welch':
        sig_len = Data.shape[-1]
        if nperseg is None:
            nperseg = min(256, sig_len // 4)
        freqs, PSD = signal.welch(Data, axis=-1, nperseg=nperseg, window=window)
        PSD = PSD[..., MinFreq:MaxFreq]

        if return_phase:
            HalfLen = Data.shape[-1] // 2
            m0 = max(0, int(MinFreq))
            m1 = min(int(MaxFreq), HalfLen)
            if m1 <= m0:
                m1 = min(m0 + 1, HalfLen)
            FFTComplex_any = np.fft.fft(Data, axis=-1)[..., :HalfLen][..., m0:m1]
            phase_raw = np.angle(FFTComplex_any)

    elif method == 'welch_evo':
        sig_len = Data.shape[-1]
        if nperseg is None:
            nperseg = min(256, sig_len // 4)
        noverlap = nperseg // 2

        f, t, Zxx = signal.stft(
            Data, axis=-1, nperseg=nperseg, noverlap=noverlap, window=window, detrend='constant', 
            return_onesided=True, boundary=None, padded=False
        )
        P = (np.abs(Zxx) ** 2)

        freq_bins = P.shape[-2]
        m0 = max(0, int(MinFreq))
        m1 = min(int(MaxFreq), freq_bins)
        if m1 <= m0:
            m1 = min(m0 + 1, freq_bins)

        P = P[..., m0:m1, :]
        PSD = P.mean(axis=-1)

        if return_phase:
            phase_evo = np.angle(Zxx[..., m0:m1, :])
            phase_raw = _circ_mean(phase_evo, axis=-1, keepdims=False)

    else:  # 'fft'
        HalfLen = Data.shape[-1] // 2
        FFTRes = np.abs(np.fft.fft(Data, axis=-1)[..., :HalfLen])[..., MinFreq:MaxFreq]
        PSD = (FFTRes ** 2) / Data.shape[-1]
        if return_phase:
            FFTComplex = np.fft.fft(Data, axis=-1)[..., :HalfLen][..., MinFreq:MaxFreq]
            phase_raw = np.angle(FFTComplex)

    if ReducedAxis == 'All':
        if preserve_dims:
            AggPSD = np.mean(PSD, axis=(0, 1), keepdims=True)
            if len(original_shape) < 3:
                AggPSD = np.squeeze(AggPSD, axis=1)
        else:
            AggPSD = np.mean(PSD, axis=(0, 1))
        _eps = 1e-12
        AggPSD = np.nan_to_num(AggPSD, nan=0.0, posinf=0.0, neginf=0.0)
        AggPSD = np.maximum(AggPSD, 0.0)
        AggPSPDF = (AggPSD + _eps) / np.sum(AggPSD + _eps, axis=(-1), keepdims=True)

    elif ReducedAxis == 'None':
        _eps = 1e-12
        PSD_safe = np.nan_to_num(PSD, nan=0.0, posinf=0.0, neginf=0.0)
        PSD_safe = np.maximum(PSD_safe, 0.0)
        AggPSPDF = (PSD_safe + _eps) / np.sum(PSD_safe + _eps, axis=(-1), keepdims=True)
        if preserve_dims and len(original_shape) < 3:
            AggPSPDF = np.squeeze(AggPSPDF, axis=1)

    else:
        _eps = 1e-12
        PSD_safe = np.nan_to_num(PSD, nan=0.0, posinf=0.0, neginf=0.0)
        PSD_safe = np.maximum(PSD_safe, 0.0)
        AggPSPDF = (PSD_safe + _eps) / np.sum(PSD_safe + _eps, axis=(-1), keepdims=True)

    if return_phase and phase_raw is not None:
        if ReducedAxis == 'All':
            if preserve_dims:
                phase_result = _circ_mean(phase_raw, axis=(0, 1), keepdims=True)
                if len(original_shape) < 3:
                    phase_result = np.squeeze(phase_result, axis=1)
            else:
                phase_result = _circ_mean(phase_raw, axis=(0, 1), keepdims=False)
        else:
            phase_result = phase_raw
            if preserve_dims and len(original_shape) < 3:
                phase_result = np.squeeze(phase_result, axis=1)
        return AggPSPDF, phase_result

    return AggPSPDF


# Permutation given PSD over each generation
def ProbPermutation(Data, WindowSize=3):
    # To make the data have the shape: (NMiniBat, N_frequency, NGen)
    Data = np.transpose(Data, (0,2,1))
    
    # For the M generation vectors, Data shape: (NMiniBat, N_frequency, NGen)
    # For the true PSD, Data shape: (1, N_frequency, NMiniBat)
    
    # Generating true permutation cases
    TruePerms = np.concatenate(list(itertools.permutations(np.arange(WindowSize)))).reshape(-1, WindowSize)

    # Getting all permutation cases
    Data_Ext = tf.signal.frame(Data, frame_length=WindowSize, frame_step=1, axis=-1)
    PermsTable =  np.argsort(Data_Ext, axis=-1)

    CountPerms = 1- (TruePerms[None,None,None] == PermsTable[:,:,:, None])
    CountPerms = 1-np.sum(CountPerms, axis=-1).astype('bool')

    # Reducing the window axis
    CountPerms = np.sum(CountPerms, axis=(2))
    
    # Data shape: (NMiniBat, N_frequency, N_permutation_cases)
    ProbCountPerms = CountPerms / np.sum(CountPerms, axis=(1,2), keepdims=True)
    
    return np.maximum(ProbCountPerms, 1e-7)    



def MeanKLD(P,Q):
    return np.mean(np.sum(P*np.log(P/Q), axis=-1))



# The 'predict' function in TensorFlow version 2.10 may cause memory leak issues.
def Sampler (Data, SampModel,BatchSize=100, GPU=True):
    if GPU==False:
        with tf.device('/CPU:0'):
            PredVal = SampModel.predict(Data, batch_size=BatchSize, verbose=1)   
    else:
        PredVal = SampModel.predict(Data, batch_size=BatchSize, verbose=1)   

    return PredVal


def SamplingZ (Data, SampModel, NMiniBat, NParts, NSubGen, BatchSize = 1000, GPU=True, SampZType='Modelbd',  SecDataType=None, ReparaStdZj=1.):
    
    '''
    Sampling Samp_Z 
    - Return shape of Samp_Z: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
    - NGen = NParts * NSubGen
    
    - Modelbd: The predicted values are repeated NGen times after the prediction. 
    - Modelbr: The data is repeated NParts times before the prediction. Then, The predicted values are repeated NSubGen times.
    - Gaussbr: The data sampled (NMiniBat, NParts, LatDim) from the Gaussian distribution is repeated NSubGen times. 
    '''
    
    assert SampZType in ['Modelbd','Modelbdr', 'Gaussbr'], "Please verify the value of 'SampZType'. Only 'Modelbd','Modelbdr', and 'Gaussbr' are valid."
    NGen = NParts * NSubGen
    
    # Sampling Samp_Z
    if SampZType =='Modelbd': # Z ~ N(Zμ|y, σ) or N(Zμ|y, cond, σ) 
        # Shape of UniqSamp_Z: (NMiniBat, LatDim) 
        UniqSamp_Z = Sampler(Data, SampModel, BatchSize=BatchSize, GPU=GPU)
        Samp_Z =  np.broadcast_to(UniqSamp_Z[:, None], (NMiniBat, NGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])
    
    elif SampZType =='Modelbdr':
        if SecDataType == 'CONDIN' : # For the CondVAE
            DataExt = [np.repeat(arr, NParts, axis=0) for arr in Data]
        else:
            DataExt = np.repeat(Data, NParts, axis=0)
        # Shape of UniqSamp_Z: (NMiniBat, NParts, LatDim) 
        UniqSamp_Z = Sampler(DataExt, SampModel, BatchSize=BatchSize, GPU=GPU).reshape(NMiniBat, NParts, -1)
        Samp_Z =  np.broadcast_to(UniqSamp_Z[:, :, None], (NMiniBat, NParts, NSubGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])
    
    elif SampZType =='Gaussbr': # Z ~ N(0, ReparaStdZj)
        # Shape of UniqSamp_Z: (NMiniBat, NParts, LatDim) 
        UniqSamp_Z = np.random.normal(0, ReparaStdZj, (NMiniBat, NParts , SampModel.output.shape[-1]))
        Samp_Z =  np.broadcast_to(UniqSamp_Z[:, :, None], (NMiniBat, NParts, NSubGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])

    # Return shape of Samp_Z: (NMiniBat*NParts*NSubGen, LatDim)
    return Samp_Z


def SamplingZj (Samp_Z, NMiniBat,  NParts, NSubGen, LatDim, NSelZ, ZjType='bd' ):
    
    '''
     Sampling Samp_Zj 
    - Return shape of Samp_Zj: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
    - Masking is applied to select Samp_Zj from Samp_Z 
      by assuming that the Samp_Z with indices other than j have a fixed mean value of '0' following a Gaussian distribution.
    - Samp_Zj ~ N(Zμj|y, σj), j∼U(1,LatDim)
    - In the expression j∼U(1,LatDim), j corresponds to LatDim and all js are selected randomly.
    '''
    NGen = NParts * NSubGen
    
    # Masking for selecting Samp_Zj from Samp_Z 
    if ZjType =='bd': 
        Mask_Z = np.zeros((NMiniBat, NGen, LatDim))
        for i in range(NMiniBat):
            Mask_Z[i, :, np.random.choice(LatDim, NSelZ,replace=False )] = 1
    
    # Selecting Samp_Zj from Samp_Z 
    Mask_Z = Mask_Z.reshape(NMiniBat*NGen, LatDim)
    Samp_Zj = Samp_Z * Mask_Z

    # Return shape of Samp_Zj: (NMiniBat*NGen, LatDim)
    return Samp_Zj


def SamplingFCs (Data, SampModel, NMiniBat, NParts, NSubGen, BatchSize = 1000, GPU=True, SampFCType='Modelbdrm', FcLimit= [0, 1.]):

    # Check for valid SampFCType values
    assert SampFCType in ['Modelbdrm', 'Modelbdm'], "Please verify the value of 'SampFCType'. Only 'Modelbdrm', and 'Modelbdm' are valid."
    
    # Sampling FCs
    if SampFCType =='Modelbdrm':
        DataExt = np.repeat(Data, NParts*NSubGen, axis=0)
        ## Return shape of Samp_FC: (NMiniBat*NParts*NSubGen, NFCs) for optimal use of GPU
        Samp_FC = Sampler(DataExt, SampModel, BatchSize=BatchSize, GPU=GPU).reshape(-1, SampModel.output.shape[-1])

    elif SampFCType =='Modelbdm':
        DataExt = np.repeat(Data, NSubGen, axis=0)
        # Shape of UniqSamp_FC: (NMiniBat, 1, NSubGen, LatDim) 
        UniqSamp_FC = Sampler(DataExt, SampModel, BatchSize=BatchSize, GPU=GPU).reshape(NMiniBat, 1, NSubGen, -1)
        Samp_FC =  np.broadcast_to(UniqSamp_FC, (NMiniBat, NParts, NSubGen, UniqSamp_FC.shape[-1])).reshape(-1, UniqSamp_FC.shape[-1])

    # Return shape of Samp_FC: (NMiniBat*NParts*NSubGen, LatDim)
    return FcLimit[0] + Samp_FC * (FcLimit[1] - FcLimit[0])
    
    

def Partition3D(Mat, NParts):
    B, M, F = Mat.shape
    PartSize = M // NParts
    Remainder = M % NParts

    NewMat = np.zeros_like(Mat)
    # ReturnIDX will store the partition ID and local position for each element
    ReturnIDX = np.zeros((B, M, 2), dtype=int)  # Adding an extra dimension for partition ID and local position

    for b in range(B):
        CumulativeIndex = 0
        for i in range(NParts):
            PartSizeAdjusted = PartSize + (1 if i < Remainder else 0)
            Slice = Mat[b, CumulativeIndex:CumulativeIndex + PartSizeAdjusted, :]
            SortedIndices = np.argsort(Slice, axis=0)
            SortedSlice = np.take_along_axis(Slice, SortedIndices, axis=0)

            NewMat[b, CumulativeIndex:CumulativeIndex + PartSizeAdjusted, :] = SortedSlice
            # Store partition ID and local position
            ReturnIDX[b, CumulativeIndex:CumulativeIndex + PartSizeAdjusted, 0] = i
            for f in range(F):
                ReturnIDX[b, CumulativeIndex:CumulativeIndex + PartSizeAdjusted, 1] = np.arange(PartSizeAdjusted)

            CumulativeIndex += PartSizeAdjusted

    return NewMat, ReturnIDX[:,:,0]


def GenConArange (ConData, NGen):
    # Processing Conditional information 
    ## Finding the column index of the max value in each row of ConData and sort the indices
    ArgMaxP_PSPDF = np.argmax(ConData, axis=-1)
    SortIDX = np.column_stack((np.argsort(ArgMaxP_PSPDF), ArgMaxP_PSPDF[np.argsort(ArgMaxP_PSPDF)]))

    # Computing the number of iterations
    UniqPSPDF = np.unique(ArgMaxP_PSPDF)
    NIter = NGen // len(UniqPSPDF)

    # Selecting one row index for each unique value, repeated for NIter times and ensure the total number of selected indices matches NGen
    SelIDX = np.concatenate([np.random.permutation(SortIDX[SortIDX[:, 1] == psd])[:1] for psd in UniqPSPDF for _ in range(NIter)], axis=0)
    SelIDX = np.vstack((SelIDX, np.random.permutation(SortIDX)[:NGen - len(SelIDX)]))

    # Sorting IDX based on the max values
    SelIDX = SelIDX[np.argsort(SelIDX[:, 1])]

    ## Generating CON_Arange
    return ConData[SelIDX[:,0]]


def GenConArangeSimple (ConData, NGen, seed=1):

    ### Selecting Conditional dataset randomly
    np.random.seed(seed)
    ConData = np.random.permutation(ConData)[:NGen]
    
    ### Identifying the index of maximum frequency for each selected condition
    MaxFreqSelCond = np.argmax(ConData, axis=-1)
    ### Sorting the selected conditions by their maximum frequency
    Idx_MaxFreqSelCond = np.argsort(MaxFreqSelCond)
    CONbm_Sort = ConData[Idx_MaxFreqSelCond]

    return CONbm_Sort



def Denorm (NormX, MaxX, MinX):
    return NormX * (MaxX - MinX) + MinX 


def MAPECal (TrueData, PredSigRec, MaxX, MinX):
    # Denormalization
    DenormTrueData = Denorm(TrueData, MaxX, MinX).copy()
    DenormPredSigRec = Denorm(PredSigRec, MaxX, MinX).copy()
   
    # MAPE
    MAPEdenorm = np.mean(np.abs((DenormTrueData - DenormPredSigRec) / DenormTrueData)) * 100
    MAPEnorm = np.mean(np.abs(((TrueData+1e-7) - PredSigRec) / (TrueData+1e-7))) * 100

    return MAPEnorm, MAPEdenorm


def MSECal (TrueData, PredSigRec, MaxX, MinX):
    # Denormalization
    DenormTrueData = Denorm(TrueData, MaxX, MinX).copy()
    DenormPredSigRec = Denorm(PredSigRec, MaxX, MinX).copy()
   
    # MSE
    MSEdenorm = np.mean((DenormTrueData - DenormPredSigRec)**2)
    MSEnorm = np.mean((TrueData - PredSigRec)**2)

    # R-squared (coefficient of determination)
    ybar_denorm = np.mean(DenormTrueData)
    sst_denorm  = np.mean((DenormTrueData - ybar_denorm)**2)  # Var(y) in original units
    R2denorm   = 1.0 - (MSEdenorm / sst_denorm)
    
    return MSEnorm, MSEdenorm, R2denorm



def mu_law_encode(audio, quantization_channels=256, input_range='0to1'):
    """
    Mu-law encoding for audio.
    
    Args:
        audio: Input audio signal.
               By default, assumed to be in the range [-1, 1] or [0, 1].
        quantization_channels: Number of quantization levels (commonly 256).
        input_range:
            - '-1to1': audio is already in the range [-1, 1]
            - '0to1': audio is in the range [0, 1], so it will be rescaled to [-1, 1]

    Returns:
        Encoded audio signal as int in the range [0, quantization_channels - 1].
    """
    if input_range == '0to1':
        # Rescale [0, 1] to [-1, 1]
        audio = 2.0 * audio - 1.0
    
    mu = quantization_channels - 1
    
    # Ensure audio is within [-1, 1]
    audio = np.clip(audio, -1, 1)
    
    # Apply the mu-law formula
    encoded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    
    # Convert from [-1, 1] to [0, mu], then round
    encoded = ((encoded + 1) / 2 * mu + 0.5).astype(np.int32)
    return encoded

def mu_law_decode(encoded, quantization_channels=256, output_range='0to1'):
    """
    Mu-law decoding for audio.
    
    Args:
        encoded: Encoded audio signal (int), in the range [0, quantization_channels - 1].
        quantization_channels: Number of quantization levels (commonly 256).
        output_range:
            - '-1to1': returns the decoded audio in the range [-1, 1]
            - '0to1': rescales the decoded audio to [0, 1]

    Returns:
        Decoded audio signal (float).
        By default in the range [-1, 1], or in [0, 1] if specified.
    """
    mu = quantization_channels - 1
    
    # Convert from [0, mu] to [-1, 1]
    encoded = 2.0 * (encoded.astype(np.float32) / mu) - 1.0
    
    # Apply the mu-law decoding formula
    decoded = np.sign(encoded) * (np.exp(np.abs(encoded) * np.log1p(mu)) - 1) / mu
    
    if output_range == '0to1':
        # Rescale [-1, 1] back to [0, 1]
        decoded = (decoded + 1.0) / 2.0
    
    return decoded

        
