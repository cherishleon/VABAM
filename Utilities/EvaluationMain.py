import os
import numpy as np
from scipy.stats import mode
import itertools
import pickle
from tqdm import trange, tqdm

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utilities.AncillaryFunctions64 import FFT_PSD, ProbPermutation, MeanKLD, Sampler, SamplingZ, SamplingZj, SamplingFCs, compute_snr
from Utilities.Utilities import CompResource

import sys
sys.path.append("..")
from Benchmarks.Models.DiffWave64 import DiffWAVE_Restoration
from Benchmarks.Models.VDiffWave64 import VDiffWAVE_Restoration


def find_t(instance, Xbdr_tmp_copy, Iter, GenSteps, SNR_cutoff=10.0):
    """
    Finds the time step t where the SNR is just above the cutoff threshold.
    That is, it returns the last  where SNR is above the SNR_cutoff.
    
    Parameters:
    - instance: an object containing GenModel and sampling functions
    - Xbdr_tmp_copy: A true, writable copy of Xbdr_tmp to avoid modifying the original
    - Iter: Number of iterations (from cfg)
    - GenSteps: Number of generation steps (from cfg)
    - SNR_cutoff: The SNR threshold
    
    Returns:
    - t_tmp and the corresponding iteration index
    """
    # Create a copy of the original to preserve it
    Xbdr_tmp_raw = Xbdr_tmp_copy.copy()
    
    # Variables to store the last t and index where SNR is above the cutoff
    last_t = None
    last_i = None

    for i in range(Iter):
        if 'VDWave' in instance.Name:
            t_tmp = i / float(Iter - 1)  
            Xbdr_tmp_iter, _, _ = instance.GenModel.sample_q_t_0(Xbdr_tmp_copy.copy(), t_tmp, None, gamma_t=None)
        elif 'DiffWave' in instance.Name:
            t_tmp = i  # Directly use the index
            Noise = tf.random.normal(tf.shape(Xbdr_tmp_raw), 0, instance.GenModel.config['GaussSigma'])
            Xbdr_tmp_iter, _ = instance.GenModel.diffusion(Xbdr_tmp_copy.copy(), instance.GenModel.alpha_bar[t_tmp].item(), Noise)
        else:
            raise ValueError(f"Unsupported model type: {instance.Name}")
            
        # Compute the SNR
        snr_val = np.mean(compute_snr(Xbdr_tmp_raw, Xbdr_tmp_iter))
        print(f"Iteration {i}, t={t_tmp}, SNR={snr_val}")
        
        # Save the t and index if SNR is above the cutoff
        if snr_val > SNR_cutoff:
            last_t = t_tmp
            last_i = i
        else:
            # When SNR first drops below the cutoff, return the last t and index where SNR was above the cutoff
            if last_t is not None:
                print(f"SNR dropped below {SNR_cutoff} at iteration {i}. Returning t={last_t} (iteration {last_i}) where SNR was still above cutoff.")
                return last_t, last_i
            else:
                # If SNR is below the cutoff from the beginning, return the current values
                print(f"At iteration {i}, SNR is below {SNR_cutoff} without any prior SNR above cutoff. Returning current values.")
                return t_tmp, i

    # If SNR is above the cutoff for all iterations, return the default value
    if 'VDWave' in instance.Name:
        return float(GenSteps - 1) / float(Iter - 1), GenSteps
    elif 'DiffWave' in instance.Name:
        return GenSteps, GenSteps


# FFT_PSD function to support multi-method processing
def FFT_PSD_MultiMethod(Data, ReducedAxis, MinFreq=1, MaxFreq=51, 
                       methods=['fft', 'welch', 'matching_pursuit', 'welch_evo'],
                       nperseg=None, window='hann', preserve_dims=False, return_phase=False):
    """
    Multi-method Power Spectral Density calculation.
    
    Returns
    -------
    results : dict
        Dictionary with method names as keys and their respective PSD results as values.
        Format: {'method_name': (AggPSPDF, phase_result)}
    """
    results = {}
    
    for method in methods:
        try:
            if return_phase:
                psd_result, phase_result = FFT_PSD(
                    Data, ReducedAxis, MinFreq, MaxFreq, method, 
                    nperseg, window, preserve_dims, return_phase
                )
                results[method] = (psd_result, phase_result)
            else:
                psd_result = FFT_PSD(
                    Data, ReducedAxis, MinFreq, MaxFreq, method, 
                    nperseg, window, preserve_dims, return_phase
                )
                results[method] = (psd_result, None)
        except Exception as e:
            print(f"Warning: Method {method} failed with error: {e}")
            results[method] = (None, None)
    
    return results




class Evaluator ():
    
    def __init__ (self, MinFreq=1, MaxFreq=51,  SimSize = 1, NMiniBat=100,  NSubGen=100, NParts=5, ReparaStdZj = 1, NSelZ = 1, 
                  SampBatchSize = 1000, GenBatchSize = 1000, SelMetricCut = 1., SelMetricType = 'KLD', GPU=False, Name=None,
                  fft_methods=['fft']):

        
        # Optional parameters with default values
        self.MinFreq = MinFreq               # The minimum frequency value within the analysis range (default = 1).
        self.MaxFreq = MaxFreq               # The maximum frequency value within the analysis range (default = 51).
        self.SimSize = SimSize               # Then umber of simulation repetitions for aggregating metrics (default: 1)
        self.NMiniBat = NMiniBat             # The size of the mini-batch, splitting the task into N pieces of size NMiniBat.
        self.NSubGen = NSubGen               # The number of generations (i.e., samplings) within a sample.
        self.NParts = NParts                 # The number of partitions (i.e., samplings) in generations within a sample.
        self.ReparaStdZj = ReparaStdZj       # The size of the standard deviation when sampling Zj (Samp_Zjb ~ N(0, ReparaStdZj)).
        self.NSelZ = NSelZ                   # The size of js to be selected at the same time (default: 1).
        self.SampBatchSize = SampBatchSize   # The batch size during prediction of the sampling model.
        self.GenBatchSize= GenBatchSize      # The batch size during prediction of the generation model.
        self.GPU = GPU                       # GPU vs CPU during model predictions (i.e., for SampModel and GenModel). "The CPU is strongly recommended for optimal precision."
        self.SelMetricCut = SelMetricCut     # The threshold for Zs and ancillary data where the metric value is below SelMetricCut.
        self.SelMetricType = SelMetricType   # The type of metric used for selecting Zs and ancillary data. 
        self.Name = Name                     # Model name.
        self.NGen = NSubGen * NParts         # The number of generations (i.e., samplings) within the mini-batch.
        
        # Method parameters
        self.fft_methods = fft_methods
        
        # Iteration counters
        self.sim, self.mini, self.iter = 0, 0, 0

        # Generation of checkpoint folders
        if not os.path.exists('./Data/Checkpoints/') and Name is not None:
            os.makedirs('./Data/Checkpoints/')
        self.CheckpointPath = './Data/Checkpoints/' +Name+ '.pkl'
        
        # Initialize method-specific trackers
        self._initialize_method_trackers()
    
    ''' --------------------------------------------------------- Ancillary Functions ---------------------------------------------------------'''
    
    def _initialize_method_trackers(self):
        """Initialize tracking structures for all methods."""
        self.SubResDic = {}
        self.AggResDic = {}
        
        for method in self.fft_methods:
            self.SubResDic[f'I_V_ZjZ_{method}'] = []
            self.SubResDic[f'I_V_FCsZj_{method}'] = []
            self.SubResDic[f'I_S_FCsZj_{method}'] = []
            
            self.AggResDic[f'I_V_ZjZ_{method}'] = []
            self.AggResDic[f'I_V_FCsZj_{method}'] = []
            self.AggResDic[f'I_S_FCsZj_{method}'] = []
            
            setattr(self, f'I_V_ZjZ_{method}', 0)
            setattr(self, f'I_V_FCsZj_{method}', 0)
            setattr(self, f'I_S_FCsZj_{method}', 0)

    def save_checkpoint(self, ):
        """
        Saves (pickles) all relevant fields, including optional metrics if present.
        """
        checkpoint_data = {
            "sim": self.sim,
            "mini": self.mini,
            "iter": self.iter,
            "SubResDic": self.SubResDic,
            "AggResDic": self.AggResDic,
            "BestZsMetrics": self.BestZsMetrics,
            "TrackerCand": self.TrackerCand,
            "TrackerCand_Temp": self.TrackerCand_Temp,
            "Name": self.Name,
            "NMiniBat": self.NMiniBat,
            "SimSize": self.SimSize,
            "fft_methods": self.fft_methods }

        # Save method-specific metrics
        for method in self.fft_methods:
            for metric_type in ['I_V_ZjZ', 'I_V_FCsZj', 'I_S_FCsZj']:
                attr_name = f'{metric_type}_{method}'
                if hasattr(self, attr_name):
                    checkpoint_data[attr_name] = getattr(self, attr_name)

        # Optional attributes that may or may not exist yet
        optional_attrs = ["I_V_CONsZj", "I_S_CONsZj", "I_V_CONsX", "I_S_CONsX"]
        
        for attr in optional_attrs:
            if hasattr(self, attr):
                checkpoint_data[attr] = getattr(self, attr)

        # Write to disk
        with open(self.CheckpointPath, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Evaluator] Checkpoint saved -> {self.CheckpointPath}")

    
    def load_checkpoint(self, ):
        """
        Loads state from 'path' into the current Evaluator instance.
        Tries to assign optional fields if present in the file.
        """
        if not os.path.isfile(self.CheckpointPath):
            raise FileNotFoundError(f"No checkpoint file found at: {self.CheckpointPath}")

        with open(self.CheckpointPath, 'rb') as f:
            checkpoint_data = pickle.load(f)

        # Mandatory fields
        self.sim = checkpoint_data.get("sim", 0)
        self.mini = checkpoint_data.get("mini", 0)
        self.iter = checkpoint_data.get("iter", 0)
        self.SubResDic = checkpoint_data.get("SubResDic", {})
        self.AggResDic = checkpoint_data.get("AggResDic", {})
        self.BestZsMetrics = checkpoint_data.get("BestZsMetrics", {})
        self.TrackerCand = checkpoint_data.get("TrackerCand", {})
        self.TrackerCand_Temp = checkpoint_data.get("TrackerCand_Temp", {})
        self.Name = checkpoint_data.get("Name", None)
        self.NMiniBat = checkpoint_data.get("NMiniBat", self.NMiniBat)
        self.SimSize = checkpoint_data.get("SimSize", self.SimSize)
        self.fft_methods = checkpoint_data.get("fft_methods", self.fft_methods)

        # Load method-specific metrics
        for method in self.fft_methods:
            for metric_type in ['I_V_ZjZ', 'I_V_FCsZj', 'I_S_FCsZj']:
                attr_name = f'{metric_type}_{method}'
                setattr(self, attr_name, checkpoint_data.get(attr_name, 0))

        # Load optional attributes
        for attr in ["I_V_CONsZj", "I_S_CONsZj", "I_V_CONsX", "I_S_CONsX"]:
            setattr(self, attr, checkpoint_data.get(attr, None))
            
        print(f"[Evaluator] Checkpoint loaded <- {self.CheckpointPath}")
        print("iter =", self.iter)
        
    ''' --------------------------------------------------------- Candidate Z Selection Functions ---------------------------------------------------------'''
        
    def LocCandZsMaxFreq(self, CandQV_results, Samp_Z, SecData=None):
        """Multi-method candidate Z location with method-specific tracking."""
        
        for method, (CandQV, _) in CandQV_results.items():
            if CandQV is None:
                continue
                
            # Calculate scores based on SelMetricType
            if self.SelMetricType == 'Entropy': 
                Score = -np.sum(CandQV * np.log(CandQV), axis=1).ravel()
            elif self.SelMetricType == 'KLD':
                # Check if QV_Batch_method exists, if not skip this method
                qv_batch_attr = f'QV_Batch_{method}'
                if not hasattr(self, qv_batch_attr):
                    print(f"Warning: {qv_batch_attr} not found, skipping method {method}")
                    continue
                    
                # Use method-specific QV_Batch
                QV_Batch_method = getattr(self, qv_batch_attr)
                CandQV_T = CandQV.transpose(0,2,1).reshape(self.NMiniBat*self.NGen, -1)[:,:,None]
                KLD_BatGen = np.sum(QV_Batch_method * np.log(QV_Batch_method / CandQV_T), axis=1)
                Score = np.min(KLD_BatGen, axis=-1)
    
            # Get maximum frequency
            MaxFreq = np.argmax(CandQV, axis=1).ravel() + 1
    
            # Check if BestZsMetrics exists for this method
            if method not in self.BestZsMetrics:
                print(f"Warning: BestZsMetrics not found for method {method}, skipping")
                continue
    
            for Freq, _ in self.BestZsMetrics[method].items():
                FreqIdx = np.where(MaxFreq == Freq)[0]
                if len(FreqIdx) < 1: 
                    continue
    
                # Find minimum score and candidate Z values
                MinScoreIdx = np.argmin(Score[FreqIdx]) 
                MinScore = np.min(Score[FreqIdx]) 
                CandZs = Samp_Z[[FreqIdx[MinScoreIdx]]]
                
                # Track results for specific method
                self.TrackerCand_Temp[method][Freq]['TrackZX'].append(CandZs[None])
                self.TrackerCand_Temp[method][Freq]['TrackMetrics'].append(MinScore[None])
                
                if SecData is not None:
                    CandSecData = SecData[[FreqIdx[MinScoreIdx]]]
                    self.TrackerCand_Temp[method][Freq]['TrackSecData'].append(CandSecData[None])
                else:
                    CandSecData = None
    
                # Update best Z metrics for specific method
                if MinScore < self.BestZsMetrics[method][Freq][0]:
                    self.BestZsMetrics[method][Freq] = [MinScore, CandZs, CandSecData]
                    print(f'Candidate Z updated! Method: {method}, Freq: {Freq}, Score: {np.round(MinScore, 4)}')
        
    def SubNestedZFix(self, SubTrackerCand):
                
        ''' Constructing the dictionary: {'KeyID': { 'TrackZX' : Zs or Xs, 'TrackSecData' : Secondary-data }}
           
           - Outer Dictionary:
             - Key (KeyID): A unique, sequentially increasing integer from 'Cnt'; That is, the sequence number in each frequency domain.
             - Value: An inner dictionary (explained below)
        
           - Inner Dictionary:
             - Key (TrackZX) : Value (Tracked Z or X data)
             - Key (TrackSecData) : Values (Tracked secondary data matrix)
        '''
        
        Cnt = itertools.count()
        if self.SecDataType == False:
            Results = {next(Cnt):{ 'TrackZX' : TrackZX} 
                        for TrackZX, TrackMetrics 
                        in zip(SubTrackerCand['TrackZX'], SubTrackerCand['TrackMetrics'])
                        if TrackMetrics < self.SelMetricCut }

        else:
            Results = {next(Cnt):{ 'TrackZX' : TrackZX, 'TrackSecData' : TrackSecData} 
                        for TrackZX, TrackSecData, TrackMetrics 
                        in zip(SubTrackerCand['TrackZX'], SubTrackerCand['TrackSecData'], SubTrackerCand['TrackMetrics'])
                        if TrackMetrics < self.SelMetricCut }
            
        return Results
    
    ''' --------------------------------------------------------- Task Iteration Functions ---------------------------------------------------------'''
    
    def Iteration (self, TaskLogic, SaveInterval=1, Continue=False):
        """
        If Continue=True, try to load from 'SavePath' first.
        Then every 'SaveInterval' iterations, save a checkpoint.
        """
        
        # If resuming from a checkpoint
        if Continue and os.path.isfile(self.CheckpointPath):
            self.load_checkpoint()
            
        # Just functional code for setting the initial position of the progress bar 
        with trange(self.TotalIterSize, initial=self.iter, leave=False) as pbar:

            for sim in range(self.sim, self.SimSize):
                self.sim = sim

                # Check the types of ancillary data fed into the sampler model and define the pipeline accordingly.
                if self.SecDataType == 'CONDIN' : 
                    SplitData = [np.array_split(sub, self.SubIterSize) for sub in (self.AnalSig, self.TrueCond)] 

                else: # For models with a single input such as VAE and TCVAE.
                    SplitData = np.array_split(self.AnalSig, self.SubIterSize) 

                for mini in range(self.mini, self.SubIterSize):
                    self.mini = mini
                    self.iter += 1
                    print()

                    # Core part; the task logic as the function
                    if self.SecDataType == 'CONDIN':
                        TaskLogic([subs[mini] for subs in SplitData])
                    else:
                        TaskLogic(SplitData[mini])

                    if self.iter % SaveInterval == 0:
                        self.save_checkpoint()

                    pbar.update(1)
    
    ''' --------------------------------------------------------- Post-Sampling Selection Functions ---------------------------------------------------------'''
    
    def SelPostSamp(self, SelMetricCut=np.inf, BestZsMetrics=None, TrackerCand=None, SavePath=None):
        """Post-sampling selection with multi-method support."""
        
        self.SelMetricCut = SelMetricCut
        BestZsMetrics = self.BestZsMetrics if BestZsMetrics is None else BestZsMetrics
        TrackerCand = self.TrackerCand if TrackerCand is None else TrackerCand
        
        # Method-specific post-sampling results
        self.PostSamp = {}
        
        # Store CandFreqIDs for all methods 
        self.CandFreqIDs = {}
        
        for method in self.fft_methods:
            # Get candidate frequency IDs for current method
            if method in BestZsMetrics:
                CandFreqIDs_method = [item[0] for item in BestZsMetrics[method].items() if item[1][0] != np.inf]
                self.CandFreqIDs[method] = CandFreqIDs_method 
                
                # Select nested Z values for current method
                method_PostSamp = {FreqID: self.SubNestedZFix(TrackerCand[method][FreqID]) 
                                  for FreqID in CandFreqIDs_method if FreqID in TrackerCand.get(method, {})}
                
                self.PostSamp[method] = method_PostSamp
                
                # Count observations for current method
                NPostZs = sum(len(item[1]) for item in method_PostSamp.items())
                print(f'Method {method} - Total number of sets in NestedZs: {NPostZs}')
        
        # Save results
        if SavePath is not None:
            with open(SavePath, 'wb') as handle:
                pickle.dump(self.PostSamp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.PostSamp
        
    ''' --------------------------------------------------------- Quality Evaluation Functions ---------------------------------------------------------'''
        
    def KLD_TrueGen(self, PostSamp=None, AnalSig=None, SecDataType=None, PlotDist=True):
        """KLD evaluation with multi-method support."""
        
        PostSamp = self.PostSamp if PostSamp is None else PostSamp
        AnalSig = self.AnalSig if AnalSig is None else AnalSig
        SecDataType = self.SecDataType if SecDataType is None else SecDataType
        
        # Extract post-sampled data
        PostZsList = []
        PostSecDataList = []

        for method_data in PostSamp.values():
            for Freq, Subkeys in method_data.items():
                for Subkeys, Values in Subkeys.items():
                    PostZsList.append(np.array(Values['TrackZX']))
                    if 'TrackSecData' in Values.keys(): 
                        PostSecDataList.append(np.array(Values['TrackSecData']))
        
        PostZsList = np.concatenate(PostZsList)
        if SecDataType is not False:
            PostSecDataList = np.concatenate(PostSecDataList)
        
        # Data binding for model input
        if SecDataType == 'FCIN':
            Data = [PostSecDataList[:, :self.GenModel.input[0].shape[-1]], PostSecDataList[:, self.GenModel.input[0].shape[-1]:], PostZsList]            
        elif SecDataType == 'CONDIN':  
             Data = [PostZsList, PostSecDataList]
        elif SecDataType == False :
             Data = PostZsList
            
        # Generate signals
        self.GenSamp = CompResource(self.GenModel, Data, BatchSize=self.GenBatchSize, GPU=self.GPU)
            
        # Multi-method KLD calculation
        self.method_results = {}
        
        for method in self.fft_methods:
            try:
                # Calculate PSD for generated and true data using current method
                PSDGenSamp_results = FFT_PSD_MultiMethod(self.GenSamp, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                PSDTrueData_results = FFT_PSD_MultiMethod(AnalSig, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                
                PSDGenSamp, _ = PSDGenSamp_results[method]
                PSDTrueData, _ = PSDTrueData_results[method]
                
                if PSDGenSamp is None or PSDTrueData is None:
                    print(f"Warning: Method {method} failed for KLD calculation")
                    continue
                
                # Calculate KLD metrics for current method
                KldPSD_GenTrue = MeanKLD(PSDGenSamp, PSDTrueData)
                KldPSD_TrueGen = MeanKLD(PSDTrueData, PSDGenSamp)
                MeanKld_GTTG = (KldPSD_GenTrue + KldPSD_TrueGen) / 2
                
                # Store method-specific results
                self.method_results[method] = {
                    'KldPSD_GenTrue': KldPSD_GenTrue,
                    'KldPSD_TrueGen': KldPSD_TrueGen, 
                    'MeanKld_GTTG': MeanKld_GTTG,
                    'PSDGenSamp': PSDGenSamp,
                    'PSDTrueData': PSDTrueData
                }
                
                print(f'Method {method} - KldPSD_GenTrue: {KldPSD_GenTrue}')
                print(f'Method {method} - KldPSD_TrueGen: {KldPSD_TrueGen}')
                print(f'Method {method} - MeanKld_GTTG: {MeanKld_GTTG}')
                
            except Exception as e:
                print(f"Error calculating KLD for method {method}: {e}")
                continue
        
        # Set primary results to first successful method for backward compatibility
        if self.method_results:
            primary_method = list(self.method_results.keys())[0]
            primary_results = self.method_results[primary_method]
            self.KldPSD_GenTrue = primary_results['KldPSD_GenTrue']
            self.KldPSD_TrueGen = primary_results['KldPSD_TrueGen']
            self.MeanKld_GTTG = primary_results['MeanKld_GTTG']
        
        # Enhanced plotting for multi-method comparison
        if PlotDist and self.method_results:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, (method, results) in enumerate(self.method_results.items()):
                if idx >= 4:  # Limit to 4 plots
                    break
                    
                ax = axes[idx]
                PSDGenSamp = results['PSDGenSamp']
                PSDTrueData = results['PSDTrueData']
                
                ax.plot(PSDGenSamp, c='green', label='Generated', alpha=0.8)
                ax.plot(PSDTrueData, c='orange', label='True', alpha=0.8)
                ax.fill_between(np.arange(len(PSDTrueData)), PSDTrueData, color='orange', alpha=0.3)
                ax.fill_between(np.arange(len(PSDGenSamp)), PSDGenSamp, color='green', alpha=0.3)
                ax.set_title(f'Method: {method}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(self.method_results), 4):
                axes[idx].set_visible(False)
                
            plt.tight_layout()
            plt.show()
    
    ''' --------------------------------------------------------- Main Evaluation Functions ---------------------------------------------------------'''
    
    def Eval_ZFC (self, AnalSig, SampZModel, SampFCModel, GenModel,  FcLimit= [0, 1.],  WindowSize=3,  SecDataType='FCIN',  Continue=True ):
        
        ## Required parameters
        self.AnalSig = AnalSig              # The data to be used for analysis.
        self.SampZModel = SampZModel        # The model that samples Zs.
        self.SampFCModel = SampFCModel      # The model that samples FCs.
        self.GenModel = GenModel            # The model that generates signals based on given Zs and FCs.
        
        assert SecDataType in ['FCIN','CONDIN', False], "Please verify the value of 'SecDataType'. Only 'FCIN', 'CONDIN'  or False are valid."
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating the permutation sets (default: 3)
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True)
        self.FcLimit = FcLimit           # The threshold value of the max of the FC value input into the generation model (default: 0.05, i.e., frequency 5 Hertz)      
        self.SecDataType = SecDataType   # The ancillary data-type: Use 'FCIN' for FC values or 'CONDIN' for conditional inputs such as power spectral density.
        
        
        
        ## Intermediate variables
        self.Ndata = len(AnalSig) # The dimension size of the data.
        self.NFCs = GenModel.get_layer('Inp_FCEach').output.shape[-1] + GenModel.get_layer('Inp_FCCommon').output.shape[-1] # The dimension size of FCs.
        self.NCommonFC = self.GenModel.input[0].shape[1]
        self.LatDim = SampZModel.output.shape[-1] # The dimension size of Z.
        self.SigDim = AnalSig.shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            # Method-specific candidate tracking
            self.BestZsMetrics = {}
            self.TrackerCand_Temp = {}
            
            for method in self.fft_methods:
                self.BestZsMetrics[method] = {i: [np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
                self.TrackerCand_Temp[method] = {i: {'TrackSecData': [], 'TrackZX': [], 'TrackMetrics': []} 
                                                 for i in range(1, self.MaxFreq - self.MinFreq + 2)}
                setattr(self, f'I_V_ZjZ_{method}', 0)
                setattr(self, f'I_V_FCsZj_{method}', 0)
                setattr(self, f'I_S_FCsZj_{method}', 0)
        
         

        
        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v)
        # Calculate population PSD for all methods
        QV_Pop_results = FFT_PSD_MultiMethod(self.AnalSig, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=self.fft_methods)
        
        for method, (qv_pop, _) in QV_Pop_results.items():
            if qv_pop is not None:
                setattr(self, f'QV_Pop_{method}', qv_pop)
        
        
        def TaskLogic(SubData):

            print('-------------  ',self.Name,'  -------------')

            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData) 

            
            # Sampling Samp_Z and Samp_Zj
            # Please note that the tensor is maintained in a reduced number of dimensions for computational efficiency in practice.
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat, r: NParts, m: NSubGen, j: LatDim; 
            # The values of z are randomly sampled at dimensions b, d, r, and j, while remaining constant across dimension m.
            self.Zbdr = SamplingZ(SubData, self.SampZModel, self.NMiniBat, self.NParts, self.NSubGen, 
                                BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='Modelbdr', ReparaStdZj=self.ReparaStdZj)
            self.Zbdr_Ext = self.Zbdr.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            
            # The values of z are randomly sampled at dimensions b, d, and j, while remaining constant across dimensions r and m.
            self.Zbd = np.broadcast_to(self.Zbdr_Ext[:,0,0][:,None,None], (self.NMiniBat, self.NParts, self.NSubGen, self.LatDim)).reshape(-1, self.LatDim)
            
            # Selecting Samp_Zjs from Zbd 
            self.Zjbd = SamplingZj (self.Zbd, self.NMiniBat, self.NParts, self.NSubGen, self.LatDim, self.NSelZ, ZjType='bd' ).copy()
            
            # Selecting sub-Zjbd from Zjbd for I_V_FCsZj
            self.Zjbd_Ext = self.Zjbd.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of Zjbd_Red1 : (NMiniBat*NSubGen, LatDim)
            ## The values of z are randomly sampled at dimensions b, d, and j, while remaining constant across dimension m.
            self.Zjbd_Red1 = self.Zjbd_Ext[:, 0].reshape(self.NMiniBat*self.NSubGen, -1).copy()
            # Return shape of Zjbd_Red2 : (NMiniBat, LatDim)
            ## The values of z are randomly sampled at dimensions b, d, and j.
            self.Zjbd_Red2 = self.Zjbd_Ext[:, 0, 0].copy()
            
            
            # Sampling Samp_FC
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat, r: NParts, m: NSubGen, k: LatDim; 
            # The values of FC are randomly sampled across all dimensions b, d, r, m, and k.
            self.FCbdrm = SamplingFCs (SubData, self.SampFCModel, self.NMiniBat, self.NParts, self.NSubGen, 
                                      BatchSize = self.SampBatchSize, GPU=self.GPU, SampFCType='Modelbdrm', FcLimit= self.FcLimit)
            self.FCbdrm_Ext = self.FCbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            
            # The values of FC are randomly sampled at the dimensions b, d, m, and k, and constant across dimension r.
            self.FCbdm = np.broadcast_to(self.FCbdrm_Ext[:, 0][:,None], (self.NMiniBat, self.NParts, self.NSubGen, self.NFCs)).reshape(-1, self.NFCs)
            
            # Sorting the arranged FC values in ascending order at the generation index.
            self.FCbdm_Ext = self.FCbdm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of FCbdm_Sort : (NMiniBat*NSubGen, LatDim)
            ## The values of FC are sorted at the generation index after being randomly sampled across the dimensions b, d, m, and k.
            self.FCbdm_Sort = np.sort(self.FCbdm_Ext , axis=2)[:,0].reshape(self.NMiniBat*self.NSubGen, self.NFCs)
            # Return shape of FCbd_Sort : (NMiniBat, LatDim)
            ## The values of FC are sorted at the dimension d after being randomly sampled across the dimensions b, d and k.
            self.FCbd_Sort = np.sort(self.FCbdm_Ext[:, 0, 0], axis=0).copy() 


            
            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation transforming tensors to (NMiniBat * NParts * NSubGen, LatDim) for Zs or (NMiniBat * NParts * NSubGen, NFCs) for FCs. 
              After the computation, we then reverted them back to their original dimensions.
                       
                                        ## Variable cases for the signal generation ##
                    
              # Cases                             # Super Signal                    # Sub-Signal                # Target metric
              1) Zbdr + FCbdrm         ->         Sig_Zbdr_FCbdrm        ->         Sig_Zbd_FCbdm               I() // H() or KLD ()
              2) Zjbd + FCbdrm         ->         Sig_Zjbd_FCbdrm        ->         Sig_Zjbd_FCbdm              I() 
              3) Zjbd + FCbdm_Sort     ->         Sig_Zjbd_FCbdmSt       ->                                     I() 
              4) Zjbd + FCbd_Sort      ->         Sig_Zjbd_FCbdSt        ->                                     I()  
                                                  * St=Sort 
             '''

            # Binding the samples together, generate signals through the model 
            ListZs = [ self.Zbdr,   self.Zjbd,    self.Zjbd_Red1,       self.Zjbd_Red2]
            Set_Zs = np.concatenate(ListZs)            
            Set_FCs = np.concatenate([self.FCbdrm, self.FCbdrm,  self.FCbdm_Sort,  self.FCbd_Sort]) 
            
            Set_Data = [Set_FCs[:, :self.NCommonFC], Set_FCs[:, self.NCommonFC:], Set_Zs]
            
            # Gneraing indices for Re-splitting predictions for each case
            CaseLens = np.array([item.shape[0] for item in ListZs])
            DataCaseIDX = [0] + list(np.cumsum(CaseLens))
            
            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Set_Data, BatchSize=self.GenBatchSize, GPU=self.GPU)
            
            # Re-splitting predictions for each case
            self.Sig_Zbdr_FCbdrm, self.Sig_Zjbd_FCbdrm, self.Sig_Zjbd_FCbdmSt, self.Sig_Zjbd_FCbdSt  = [Set_Pred[DataCaseIDX[i]:DataCaseIDX[i+1]] for i in range(len(DataCaseIDX)-1)] 
            
            self.Sig_Zbdr_FCbdrm = self.Sig_Zbdr_FCbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Zjbd_FCbdrm = self.Sig_Zjbd_FCbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Zjbd_FCbdmSt = self.Sig_Zjbd_FCbdmSt.reshape(self.NMiniBat, self.NSubGen, -1)
            self.Sig_Zjbd_FCbdSt = self.Sig_Zjbd_FCbdSt.reshape(self.NMiniBat, -1)
            
            self.Sig_Zbd_FCbdm = self.Sig_Zbdr_FCbdrm[:, 0]
            self.Sig_Zjbd_FCbdm = self.Sig_Zjbd_FCbdrm[:, 0]



            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics   # Function             # Code                           # Function            # Code 
                1) I_V_ZjZ      q(v|Sig_Zjbd_FCbdm)    <QV_Zjbd_FCbdm>          vs      q(v|Sig_Zbd_FCbdm)    <QV_Zbd_FCbdm>
                2) I_V_FCsZj    q(v|Sig_Zjbd_FCbdSt)   <QV_Zjbd_FCbdSt>         vs      q(v|Sig_Zjbd_FCbdm)   <QV_Zjbd_FCbdm>
                3) I_S_FCsZj    q(s|Sig_Zjbd_FCbdmSt)  <QV//QS_Zjbd_FCbdmSt>    vs      q(s|Sig_Zjbd_FCbdrm)  <QV//QS_Zjbd_FCbdrm>
                4) H()//KLD()   q(v|Sig_Zbdr_FCbdrm)   <QV_Zbdr_FCbdrm>                 q(v)                  <QV_Batch>       
                
                                                       
                ## Metric list : I_V_ZjZ, I_V_FCsZj, I_S_FCsZj, H() or KLD()
                
            '''

            # Multi-method PSD calculations and metric computation
            for method in self.fft_methods:
                try:
                    # Calculate PSDs for all signal types using current method
                    MQV_Zbd_FCbdm_results = FFT_PSD_MultiMethod(self.Sig_Zbd_FCbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    MQV_Zjbd_FCbdm_results = FFT_PSD_MultiMethod(self.Sig_Zjbd_FCbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zjbd_FCbdSt_results = FFT_PSD_MultiMethod(self.Sig_Zjbd_FCbdSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    
                    QV_Zbdr_FCbdrm_results = FFT_PSD_MultiMethod(self.Sig_Zbdr_FCbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zjbd_FCbdrm_results = FFT_PSD_MultiMethod(self.Sig_Zjbd_FCbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zjbd_FCbdmSt_results = FFT_PSD_MultiMethod(self.Sig_Zjbd_FCbdmSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    
                    # Extract results for current method
                    MQV_Zbd_FCbdm, _ = MQV_Zbd_FCbdm_results[method]
                    MQV_Zjbd_FCbdm, _ = MQV_Zjbd_FCbdm_results[method]
                    QV_Zjbd_FCbdSt, _ = QV_Zjbd_FCbdSt_results[method]
                    QV_Zbdr_FCbdrm, _ = QV_Zbdr_FCbdrm_results[method]
                    QV_Zjbd_FCbdrm, _ = QV_Zjbd_FCbdrm_results[method]
                    QV_Zjbd_FCbdmSt, _ = QV_Zjbd_FCbdmSt_results[method]
                    
                    if any(x is None for x in [MQV_Zbd_FCbdm, MQV_Zjbd_FCbdm, QV_Zjbd_FCbdSt]):
                        print(f"Warning: Method {method} returned None results, skipping...")
                        continue
                    
                    # Apply mean operation where needed
                    MQV_Zbd_FCbdm = MQV_Zbd_FCbdm.mean(1)
                    MQV_Zjbd_FCbdm = MQV_Zjbd_FCbdm.mean(1)
                    QV_Zjbd_FCbdSt = QV_Zjbd_FCbdSt[:,0]
                    
                    # Permutation calculations
                    QSV_Zbdr_FCbdrm = np.concatenate([ProbPermutation(QV_Zbdr_FCbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
                    QSV_Zjbd_FCbdrm = np.concatenate([ProbPermutation(QV_Zjbd_FCbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
                    QSV_Zjbd_FCbdmSt = ProbPermutation(QV_Zjbd_FCbdmSt, WindowSize=WindowSize)
                    
                    QS_Zbdr_FCbdrm = np.sum(QSV_Zbdr_FCbdrm, axis=2)
                    QS_Zjbd_FCbdrm = np.sum(QSV_Zjbd_FCbdrm, axis=2)
                    QS_Zjbd_FCbdmSt = np.sum(QSV_Zjbd_FCbdmSt, axis=1)
                    
                    MQS_Zjbd_FCbdrm = np.mean(QS_Zjbd_FCbdrm, axis=1)
                    MQS_Zbdr_FCbdrm = np.mean(QS_Zbdr_FCbdrm, axis=1)
                    
                    # Calculate mutual information for current method
                    I_V_ZjZ_ = MeanKLD(MQV_Zjbd_FCbdm, MQV_Zbd_FCbdm)
                    I_V_FCsZj_ = MeanKLD(QV_Zjbd_FCbdSt, MQV_Zjbd_FCbdm)
                    I_S_FCsZj_ = MeanKLD(QS_Zjbd_FCbdmSt, MQS_Zjbd_FCbdrm)
                    
                    print(f"Method {method} - I(V;z'|z): {I_V_ZjZ_}")
                    print(f"Method {method} - I(V;fc'|z'): {I_V_FCsZj_}")
                    print(f"Method {method} - I(S;fc'|z'): {I_S_FCsZj_}")
                    
                    # Store results for current method
                    self.SubResDic[f'I_V_ZjZ_{method}'].append(I_V_ZjZ_)
                    self.SubResDic[f'I_V_FCsZj_{method}'].append(I_V_FCsZj_)
                    self.SubResDic[f'I_S_FCsZj_{method}'].append(I_S_FCsZj_)
                    
                    # Accumulate for aggregated results
                    current_val = getattr(self, f'I_V_ZjZ_{method}')
                    setattr(self, f'I_V_ZjZ_{method}', current_val + I_V_ZjZ_)
                    
                    current_val = getattr(self, f'I_V_FCsZj_{method}')
                    setattr(self, f'I_V_FCsZj_{method}', current_val + I_V_FCsZj_)
                    
                    current_val = getattr(self, f'I_S_FCsZj_{method}')
                    setattr(self, f'I_S_FCsZj_{method}', current_val + I_S_FCsZj_)
                    
                    # Calculate batch QV for candidate Z location
                    QV_Batch_results = FFT_PSD_MultiMethod(SubData[:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Batch, _ = QV_Batch_results[method]
                    if QV_Batch is not None:
                        setattr(self, f'QV_Batch_{method}', QV_Batch.transpose((1,2,0)))
                    
                except Exception as e:
                    print(f"Error processing method {method}: {e}")
                    continue
            
            # Multi-method candidate Z location
            QV_Zbdr_FCbdrm_T_results = {}
            for method in self.fft_methods:
                try:
                    QV_results = FFT_PSD_MultiMethod(self.Sig_Zbdr_FCbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zbdr_FCbdrm, _ = QV_results[method]
                    if QV_Zbdr_FCbdrm is not None:
                        QV_Zbdr_FCbdrm_T_results[method] = (QV_Zbdr_FCbdrm.reshape(self.NMiniBat, self.NGen, -1).transpose(0,2,1), None)
                except Exception as e:
                    print(f"Error in candidate Z location for method {method}: {e}")
                    continue
            
            self.LocCandZsMaxFreq(QV_Zbdr_FCbdrm_T_results, self.Zbdr, self.FCbdrm)
            
            # Restructure TrackerCand for all methods
            self.TrackerCand = {}
            for method in self.fft_methods:
                self.TrackerCand[method] = {
                    item[0]: {
                        'TrackZX': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackZX']), 
                        'TrackSecData': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackSecData']), 
                        'TrackMetrics': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackMetrics'])
                    } 
                    for item in self.TrackerCand_Temp[method].items() 
                    if len(item[1]['TrackSecData']) > 0
                }
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic, Continue=Continue)

        # Calculate final aggregated results for all methods
        for method in self.fft_methods:
            # Normalize by total iteration size
            current_val = getattr(self, f'I_V_ZjZ_{method}')
            setattr(self, f'I_V_ZjZ_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_V_ZjZ_{method}'].append(getattr(self, f'I_V_ZjZ_{method}'))
            
            current_val = getattr(self, f'I_V_FCsZj_{method}')
            setattr(self, f'I_V_FCsZj_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_V_FCsZj_{method}'].append(getattr(self, f'I_V_FCsZj_{method}'))
            
            current_val = getattr(self, f'I_S_FCsZj_{method}')
            setattr(self, f'I_S_FCsZj_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_S_FCsZj_{method}'].append(getattr(self, f'I_S_FCsZj_{method}'))





    ### -------------------------- Evaluating the performance of the model using both Z and Conditions -------------------------- ###
    def Eval_ZCON (self, AnalData, SampZModel, GenModel, FcLimit= [0, 1.],  WindowSize=3,  SecDataType=None,  Continue=True ):
        
        ## Required parameters
        self.SampZModel = SampZModel         # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and Cons.
        
        assert SecDataType in ['FCIN','CONDIN', False], "Please verify the value of 'SecDataType'. Only 'FCIN', 'CONDIN'  or False are valid."
        
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating the permutation sets (default: 3).
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True).
        self.SecDataType = SecDataType   # The ancillary data-type: Use 'FCIN' for FC values or 'CONDIN' for conditional inputs such as power spectral density.
        
        
    
        ## Intermediate variables
        self.AnalSig = AnalData[0]  # The raw true signals to be used for analysis.
        self.TrueCond = AnalData[1] # The raw true PSD to be used for analysis.
        self.Ndata = len(self.AnalSig) # The dimension size of the data.
        self.LatDim = SampZModel.output.shape[-1] # The dimension size of Z.
        self.SigDim =  self.AnalSig.shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.CondDim = self.TrueCond.shape[-1] # The dimension size of the conditional inputs.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        assert self.NGen >= self.CondDim, "NGen must be greater than or equal to CondDim for the evaluation."
    
        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            # Method-specific candidate tracking
            self.BestZsMetrics = {}
            self.TrackerCand_Temp = {}
            
            for method in self.fft_methods:
                self.BestZsMetrics[method] = {i: [np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
                self.TrackerCand_Temp[method] = {i: {'TrackSecData': [], 'TrackZX': [], 'TrackMetrics': []} 
                                                 for i in range(1, self.MaxFreq - self.MinFreq + 2)}
                
                # Initialize method-specific result trackers
                setattr(self, f'I_V_ZjZ_{method}', 0)
                setattr(self, f'I_V_CONsZj_{method}', 0)
                setattr(self, f'I_S_CONsZj_{method}', 0)
            
            # Initialize method-specific result dictionaries
            self.SubResDic = {}
            self.AggResDic = {}
            
            for method in self.fft_methods:
                self.SubResDic[f'I_V_ZjZ_{method}'] = []
                self.SubResDic[f'I_V_CONsZj_{method}'] = []
                self.SubResDic[f'I_S_CONsZj_{method}'] = []
                
                self.AggResDic[f'I_V_ZjZ_{method}'] = []
                self.AggResDic[f'I_V_CONsZj_{method}'] = []
                self.AggResDic[f'I_S_CONsZj_{method}'] = []

        
         # Always ensure method-specific attributes exist (regardless of Continue flag)
        for method in self.fft_methods:
            # Initialize method-specific result trackers if they don't exist
            if not hasattr(self, f'I_V_ZjZ_{method}'):
                setattr(self, f'I_V_ZjZ_{method}', 0)
            if not hasattr(self, f'I_V_CONsZj_{method}'):
                setattr(self, f'I_V_CONsZj_{method}', 0)
            if not hasattr(self, f'I_S_CONsZj_{method}'):
                setattr(self, f'I_S_CONsZj_{method}', 0)
            
            # Initialize result dictionaries if they don't exist
            if f'I_V_ZjZ_{method}' not in getattr(self, 'SubResDic', {}):
                if not hasattr(self, 'SubResDic'):
                    self.SubResDic = {}
                self.SubResDic[f'I_V_ZjZ_{method}'] = []
            if f'I_V_CONsZj_{method}' not in getattr(self, 'SubResDic', {}):
                if not hasattr(self, 'SubResDic'):
                    self.SubResDic = {}
                self.SubResDic[f'I_V_CONsZj_{method}'] = []
            if f'I_S_CONsZj_{method}' not in getattr(self, 'SubResDic', {}):
                if not hasattr(self, 'SubResDic'):
                    self.SubResDic = {}
                self.SubResDic[f'I_S_CONsZj_{method}'] = []
            
            if f'I_V_ZjZ_{method}' not in getattr(self, 'AggResDic', {}):
                if not hasattr(self, 'AggResDic'):
                    self.AggResDic = {}
                self.AggResDic[f'I_V_ZjZ_{method}'] = []
            if f'I_V_CONsZj_{method}' not in getattr(self, 'AggResDic', {}):
                if not hasattr(self, 'AggResDic'):
                    self.AggResDic = {}
                self.AggResDic[f'I_V_CONsZj_{method}'] = []
            if f'I_S_CONsZj_{method}' not in getattr(self, 'AggResDic', {}):
                if not hasattr(self, 'AggResDic'):
                    self.AggResDic = {}
                self.AggResDic[f'I_S_CONsZj_{method}'] = []
            
        
        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v) - Calculate population PSD for all methods
        QV_Pop_results = FFT_PSD_MultiMethod(self.AnalSig, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=self.fft_methods)
        
        for method, (qv_pop, _) in QV_Pop_results.items():
            if qv_pop is not None:
                setattr(self, f'QV_Pop_{method}', qv_pop)
        
        
        def TaskLogic(SubData):
    
            print('-------------  ',self.Name,'  -------------')
    
            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData[0]) 
            self.SubCond = SubData[1]
    
            
            # Sampling Samp_Z and Samp_Zj
            # Please note that the tensor is maintained in a reduced number of dimensions for computational efficiency in practice.
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat, r: NParts, m: NSubGen, j: LatDim; 
            # The values of z are randomly sampled at dimensions b, d, r, and j, while remaining constant across dimension m.
            self.Zbdr = SamplingZ(SubData, self.SampZModel, self.NMiniBat, self.NParts, self.NSubGen, SecDataType='CONDIN',
                                BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='Modelbdr', ReparaStdZj=self.ReparaStdZj)
            self.Zbdr_Ext = self.Zbdr.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            
            # The values of z are randomly sampled at dimensions b, d, and j, while remaining constant across dimensions r and m.
            self.Zbd = np.broadcast_to(self.Zbdr_Ext[:,0,0][:,None,None], (self.NMiniBat, self.NParts, self.NSubGen, self.LatDim)).reshape(-1, self.LatDim)
            
            # Selecting Samp_Zjs from Zbd 
            self.Zjbd = SamplingZj (self.Zbd, self.NMiniBat, self.NParts, self.NSubGen, self.LatDim, self.NSelZ, ZjType='bd' ).copy()
            
            # Selecting sub-Zjbd from Zjbd for I_V_CONsZj
            self.Zjbd_Ext = self.Zjbd.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of Zjbd_Red1 : (NMiniBat*NSubGen, LatDim)
            ## The values of z are randomly sampled at dimensions b, d, and j, while remaining constant across dimension m.
            self.Zjbd_Red1 = self.Zjbd_Ext[:, 0].reshape(self.NMiniBat*self.NSubGen, -1).copy()
            # Return shape of Zjbd_Red2 : (NMiniBat, LatDim)
            ## The values of z are randomly sampled at dimensions b, d, and j.
            self.Zjbd_Red2 = self.Zjbd_Ext[:, 0, 0].copy()
            
            
            # Processing Conditional information 
            ### Generating random indices for selecting true conditions
            RandSelIDXbdm = np.random.randint(0, self.TrueCond.shape[0], self.NMiniBat * self.NSubGen)
            RandSelIDXbdrm = np.random.randint(0, self.TrueCond.shape[0], self.NMiniBat * self.NParts* self.NSubGen)
            
            ### Selecting the true conditions using the generated indices
            # True conditions are randomly sampled at the dimensions b, d, m, and k, and constant across dimension r.
            self.CONbdm = self.TrueCond[RandSelIDXbdm]
            self.CONbdm = np.broadcast_to(self.CONbdm.reshape(self.NMiniBat, self.NSubGen, -1)[:,None], (self.NMiniBat, self.NParts, self.NSubGen, self.CondDim))
            
            # True conditions are randomly sampled across all dimensions b, d, r, m, and k.
            self.CONbdrm = self.TrueCond[RandSelIDXbdrm]
            
            
            # Sorting the arranged condition values in ascending order at the generation index.
            self.CONbdm_Ext = self.CONbdm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of CONbdm_Sort : (NMiniBat*NSubGen, LatDim)
            ## The conditions are sorted at the generation index after being randomly sampled across the dimensions b, d, m, and k.
            self.CONbdm_Sort = np.sort(self.CONbdm_Ext , axis=2)[:,0].reshape(self.NMiniBat*self.NSubGen, self.CondDim)
            # Return shape of CONbd_Sort : (NMiniBat, LatDim)
            ## The conditions are sorted at the dimension d after being randomly sampled across the dimensions b, d and k.
            self.CONbd_Sort = np.sort(self.CONbdm_Ext[:, 0, 0], axis=0).copy() 
    
            
    
            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation transforming tensors to (NMiniBat * NParts * NSubGen, LatDim) for Zs or (NMiniBat * NParts * NSubGen, CondDim) for CON. 
              After the computation, we then reverted them back to their original dimensions.
                       
                                        ## Variable cases for the signal generation ##
                    
              # Cases                             # Super Signal                    # Sub-Signal                # Target metric
              1) Zbdr + CONbdrm         ->         Sig_Zbdr_CONbdrm        ->        Sig_Zbd_CONbdm              I() // H() or KLD ()
              2) Zjbd + CONbdrm         ->         Sig_Zjbd_CONbdrm        ->        Sig_Zjbd_CONbdm             I() 
              3) Zjbd + CONbdm_Sort     ->         Sig_Zjbd_CONbdmSt       ->                                    I() 
              4) Zjbd + CONbd_Sort      ->         Sig_Zjbd_CONbdSt        ->                                    I()  
                                                  * St=Sort 
             '''
    
                        
            # Binding the samples together, generate signals through the model 
            ListZs = [ self.Zbdr,   self.Zjbd,    self.Zjbd_Red1,       self.Zjbd_Red2]
            Set_Zs = np.concatenate(ListZs)            
            Set_CONs = np.concatenate([self.CONbdrm, self.CONbdrm,  self.CONbdm_Sort,  self.CONbd_Sort]) 
            Set_Data = [Set_Zs, Set_CONs]
            
            # Gneraing indices for Re-splitting predictions for each case
            CaseLens = np.array([item.shape[0] for item in ListZs])
            DataCaseIDX = [0] + list(np.cumsum(CaseLens))
            
            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Set_Data, BatchSize=self.GenBatchSize, GPU=self.GPU)
            
            # Re-splitting predictions for each case
            self.Sig_Zbdr_CONbdrm, self.Sig_Zjbd_CONbdrm, self.Sig_Zjbd_CONbdmSt, self.Sig_Zjbd_CONbdSt  = [Set_Pred[DataCaseIDX[i]:DataCaseIDX[i+1]] for i in range(len(DataCaseIDX)-1)] 
            
            self.Sig_Zbdr_CONbdrm = self.Sig_Zbdr_CONbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Zjbd_CONbdrm = self.Sig_Zjbd_CONbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Zjbd_CONbdmSt = self.Sig_Zjbd_CONbdmSt.reshape(self.NMiniBat, self.NSubGen, -1)
            self.Sig_Zjbd_CONbdSt = self.Sig_Zjbd_CONbdSt.reshape(self.NMiniBat, -1)
            
            self.Sig_Zbd_CONbdm = self.Sig_Zbdr_CONbdrm[:, 0]
            self.Sig_Zjbd_CONbdm = self.Sig_Zjbd_CONbdrm[:, 0]
    
     
            
            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics   # Function             # Code                           # Function             # Code 
                1) I_V_ZjZ      q(v|Sig_Zjbd_CONbdm)    <QV_Zjbd_CONbdm>          vs     q(v|Sig_Zbd_CONbdm)    <QV_Zbd_CONbdm>
                2) I_V_CONsZj   q(v|Sig_Zjbd_CONbdSt)   <QV_Zjbd_CONbdSt>         vs     q(v|Sig_Zjbd_CONbdm)   <QV_Zjbd_CONbdm>
                3) I_S_CONsZj   q(s|Sig_Zjbd_CONbdmSt)  <QV//QS_Zjbd_CONbdmSt>    vs     q(s|Sig_Zjbd_CONbdrm)  <QV//QS_Zjbd_CONbdrm>
                4) H()//KLD()   q(v|Sig_Zbdr_CONbdrm)   <QV_Zbdr_CONbdrm>                q(v)                   <QV_Batch>       
                
                ## Metric list : I_V_ZjZ, I_V_CONsZj, I_S_CONsZj, H() or KLD()
                
             '''
            
            # Multi-method PSD calculations and metric computation
            for method in self.fft_methods:
                try:
                    # Calculate PSDs for all signal types using current method
                    MQV_Zbd_CONbdm_results = FFT_PSD_MultiMethod(self.Sig_Zbd_CONbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    MQV_Zjbd_CONbdm_results = FFT_PSD_MultiMethod(self.Sig_Zjbd_CONbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zjbd_CONbdSt_results = FFT_PSD_MultiMethod(self.Sig_Zjbd_CONbdSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    
                    QV_Zbdr_CONbdrm_results = FFT_PSD_MultiMethod(self.Sig_Zbdr_CONbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zjbd_CONbdrm_results = FFT_PSD_MultiMethod(self.Sig_Zjbd_CONbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zjbd_CONbdmSt_results = FFT_PSD_MultiMethod(self.Sig_Zjbd_CONbdmSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    
                    # Extract results for current method
                    MQV_Zbd_CONbdm, _ = MQV_Zbd_CONbdm_results[method]
                    MQV_Zjbd_CONbdm, _ = MQV_Zjbd_CONbdm_results[method]
                    QV_Zjbd_CONbdSt, _ = QV_Zjbd_CONbdSt_results[method]
                    QV_Zbdr_CONbdrm, _ = QV_Zbdr_CONbdrm_results[method]
                    QV_Zjbd_CONbdrm, _ = QV_Zjbd_CONbdrm_results[method]
                    QV_Zjbd_CONbdmSt, _ = QV_Zjbd_CONbdmSt_results[method]
                    
                    if any(x is None for x in [MQV_Zbd_CONbdm, MQV_Zjbd_CONbdm, QV_Zjbd_CONbdSt]):
                        print(f"Warning: Method {method} returned None results, skipping...")
                        continue
                    
                    # Apply mean operation where needed
                    MQV_Zbd_CONbdm = MQV_Zbd_CONbdm.mean(1)
                    MQV_Zjbd_CONbdm = MQV_Zjbd_CONbdm.mean(1)
                    QV_Zjbd_CONbdSt = QV_Zjbd_CONbdSt[:,0]
                    
                    # Permutation calculations
                    QSV_Zbdr_CONbdrm = np.concatenate([ProbPermutation(QV_Zbdr_CONbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
                    QSV_Zjbd_CONbdrm = np.concatenate([ProbPermutation(QV_Zjbd_CONbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
                    QSV_Zjbd_CONbdmSt = ProbPermutation(QV_Zjbd_CONbdmSt, WindowSize=WindowSize)
                    
                    QS_Zbdr_CONbdrm = np.sum(QSV_Zbdr_CONbdrm, axis=2)
                    QS_Zjbd_CONbdrm = np.sum(QSV_Zjbd_CONbdrm, axis=2)
                    QS_Zjbd_CONbdmSt = np.sum(QSV_Zjbd_CONbdmSt, axis=1)
                    
                    MQS_Zjbd_CONbdrm = np.mean(QS_Zjbd_CONbdrm, axis=1)
                    MQS_Zbdr_CONbdrm = np.mean(QS_Zbdr_CONbdrm, axis=1)
                    
                    # Calculate mutual information for current method
                    I_V_ZjZ_ = MeanKLD(MQV_Zjbd_CONbdm, MQV_Zbd_CONbdm)
                    I_V_CONsZj_ = MeanKLD(QV_Zjbd_CONbdSt, MQV_Zjbd_CONbdm)
                    I_S_CONsZj_ = MeanKLD(QS_Zjbd_CONbdmSt, MQS_Zjbd_CONbdrm)
                    
                    print(f"Method {method} - I(V;z'|z): {I_V_ZjZ_}")
                    print(f"Method {method} - I(V;Con'|z'): {I_V_CONsZj_}")
                    print(f"Method {method} - I(S;Con'|z'): {I_S_CONsZj_}")
                    
                    # Store results for current method
                    self.SubResDic[f'I_V_ZjZ_{method}'].append(I_V_ZjZ_)
                    self.SubResDic[f'I_V_CONsZj_{method}'].append(I_V_CONsZj_)
                    self.SubResDic[f'I_S_CONsZj_{method}'].append(I_S_CONsZj_)
                    
                    # Accumulate for aggregated results
                    current_val = getattr(self, f'I_V_ZjZ_{method}')
                    setattr(self, f'I_V_ZjZ_{method}', current_val + I_V_ZjZ_)
                    
                    current_val = getattr(self, f'I_V_CONsZj_{method}')
                    setattr(self, f'I_V_CONsZj_{method}', current_val + I_V_CONsZj_)
                    
                    current_val = getattr(self, f'I_S_CONsZj_{method}')
                    setattr(self, f'I_S_CONsZj_{method}', current_val + I_S_CONsZj_)
                    
                    # Calculate batch QV for candidate Z location
                    QV_Batch_results = FFT_PSD_MultiMethod(SubData[0][:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Batch, _ = QV_Batch_results[method]
                    if QV_Batch is not None:
                        setattr(self, f'QV_Batch_{method}', QV_Batch.transpose((1,2,0)))
                    
                except Exception as e:
                    print(f"Error processing method {method}: {e}")
                    continue
            
            # Multi-method candidate Z location
            QV_Zbdr_CONbdrm_T_results = {}
            for method in self.fft_methods:
                try:
                    QV_results = FFT_PSD_MultiMethod(self.Sig_Zbdr_CONbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zbdr_CONbdrm, _ = QV_results[method]
                    if QV_Zbdr_CONbdrm is not None:
                        QV_Zbdr_CONbdrm_T_results[method] = (QV_Zbdr_CONbdrm.reshape(self.NMiniBat, self.NGen, -1).transpose(0,2,1), None)
                except Exception as e:
                    print(f"Error in candidate Z location for method {method}: {e}")
                    continue
            
            self.LocCandZsMaxFreq(QV_Zbdr_CONbdrm_T_results, self.Zbdr, self.CONbdrm)
            
            # Restructure TrackerCand for all methods
            self.TrackerCand = {}
            for method in self.fft_methods:
                self.TrackerCand[method] = {
                    item[0]: {
                        'TrackZX': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackZX']), 
                        'TrackSecData': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackSecData']), 
                        'TrackMetrics': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackMetrics'])
                    } 
                    for item in self.TrackerCand_Temp[method].items() 
                    if len(item[1]['TrackSecData']) > 0
                }
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic, Continue=Continue)
    
        # Calculate final aggregated results for all methods
        for method in self.fft_methods:
            # Normalize by total iteration size
            current_val = getattr(self, f'I_V_ZjZ_{method}')
            setattr(self, f'I_V_ZjZ_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_V_ZjZ_{method}'].append(getattr(self, f'I_V_ZjZ_{method}'))
            
            current_val = getattr(self, f'I_V_CONsZj_{method}')
            setattr(self, f'I_V_CONsZj_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_V_CONsZj_{method}'].append(getattr(self, f'I_V_CONsZj_{method}'))
            
            current_val = getattr(self, f'I_S_CONsZj_{method}')
            setattr(self, f'I_S_CONsZj_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_S_CONsZj_{method}'].append(getattr(self, f'I_S_CONsZj_{method}'))



    ### -------------------------- Evaluating the performance of the model using both X and Conditions -------------------------- ###
    def Eval_XCON (self, AnalData, GenModel, FcLimit=0.05,  WindowSize=3, SecDataType=None,  Continue=True, **kwargs ):
        
        ## Required parameters
        self.GenModel = GenModel             # The model that generates signals based on given Xs and Cons.
        
        if 'Wavenet' in self.Name:
            self.NoiseStd = kwargs.get('NoiseStd', 2)
            self.NSplitBatch = kwargs.get('NSplitBatch', 1) 
        elif 'DiffWave' in self.Name or 'VDWave' in self.Name:
            self.GenSteps = kwargs.get('GenSteps', 10) 
            self.StepInterval = kwargs.get('StepInterval', 1) 
        
        assert SecDataType in ['FCIN','CONDIN', False], "Please verify the value of 'SecDataType'. Only 'FCIN', 'CONDIN'  or False are valid."
        
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating the permutation sets (default: 3).
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True).
        self.SecDataType = SecDataType   # The ancillary data-type: Use 'FCIN' for FC values or 'CONDIN' for conditional inputs such as power spectral density.
        
        
    
        ## Intermediate variables
        self.AnalSig = AnalData[0]  # The raw true signals to be used for analysis.
        self.TrueCond = AnalData[1] # The raw true PSD to be used for analysis.
        self.Ndata = len(self.AnalSig) # The dimension size of the data.
        self.SigDim =  np.squeeze(self.AnalSig).shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.CondDim = self.TrueCond.shape[-1] # The dimension size of the conditional inputs.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        assert self.NGen >= self.CondDim, "NGen must be greater than or equal to CondDim for the evaluation."
    
        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            # Method-specific candidate tracking
            self.BestZsMetrics = {}
            self.TrackerCand_Temp = {}
            
            for method in self.fft_methods:
                self.BestZsMetrics[method] = {i: [np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
                self.TrackerCand_Temp[method] = {i: {'TrackSecData': [], 'TrackZX': [], 'TrackMetrics': []} 
                                                 for i in range(1, self.MaxFreq - self.MinFreq + 2)}
                
                # Initialize method-specific result trackers
                setattr(self, f'I_V_CONsX_{method}', 0)
                setattr(self, f'I_S_CONsX_{method}', 0)
            
            # Initialize method-specific result dictionaries
            self.SubResDic = {}
            self.AggResDic = {}
            
            for method in self.fft_methods:
                self.SubResDic[f'I_V_CONsX_{method}'] = []
                self.SubResDic[f'I_S_CONsX_{method}'] = []
                
                self.AggResDic[f'I_V_CONsX_{method}'] = []
                self.AggResDic[f'I_S_CONsX_{method}'] = []
        

        # Always ensure method-specific attributes exist (regardless of Continue flag)
        for method in self.fft_methods:
            # Initialize method-specific result trackers if they don't exist
            if not hasattr(self, f'I_V_CONsX_{method}'):
                setattr(self, f'I_V_CONsX_{method}', 0)
            if not hasattr(self, f'I_S_CONsX_{method}'):
                setattr(self, f'I_S_CONsX_{method}', 0)
            
            # Initialize result dictionaries if they don't exist
            if f'I_V_CONsX_{method}' not in getattr(self, 'SubResDic', {}):
                if not hasattr(self, 'SubResDic'):
                    self.SubResDic = {}
                self.SubResDic[f'I_V_CONsX_{method}'] = []
            if f'I_S_CONsX_{method}' not in getattr(self, 'SubResDic', {}):
                if not hasattr(self, 'SubResDic'):
                    self.SubResDic = {}
                self.SubResDic[f'I_S_CONsX_{method}'] = []
            
            if f'I_V_CONsX_{method}' not in getattr(self, 'AggResDic', {}):
                if not hasattr(self, 'AggResDic'):
                    self.AggResDic = {}
                self.AggResDic[f'I_V_CONsX_{method}'] = []
            if f'I_S_CONsX_{method}' not in getattr(self, 'AggResDic', {}):
                if not hasattr(self, 'AggResDic'):
                    self.AggResDic = {}
                self.AggResDic[f'I_S_CONsX_{method}'] = []
    
        
        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v) - Calculate population PSD for all methods
        QV_Pop_results = FFT_PSD_MultiMethod(self.AnalSig, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=self.fft_methods)
        
        for method, (qv_pop, _) in QV_Pop_results.items():
            if qv_pop is not None:
                setattr(self, f'QV_Pop_{method}', qv_pop)
        
        
        def TaskLogic(SubData):
    
            print('-------------  ',self.Name,'  -------------')
    
            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData[0]) 
            self.SubCond = SubData[1]
    
    
            # Sampling Samp_X
            # Please note that the tensor is maintained in a reduced number of dimensions for computational efficiency in practice.
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat, r: NParts, m: NSubGen, t: SigDim; 
            self.Xbdr_tmp = np.broadcast_to(np.squeeze(SubData[0])[:, None], (self.NMiniBat, self.NParts, self.SigDim))
            # The values of X are perturbed by randomly sampled errors along dimensions b, d, r, and t, while remaining constant along dimension m.
            if 'Wavenet' in self.Name:
                self.Xbdr_tmp = np.round(np.clip(self.Xbdr_tmp + np.random.normal(0, self.NoiseStd, self.Xbdr_tmp.shape), 0, 256))
            elif 'DiffWave' in self.Name:
                t_val, self.GenSteps = find_t(self, self.Xbdr_tmp.copy(), self.GenModel.config['Iter'], self.GenModel.config['GenSteps'], SNR_cutoff=self.GenModel.config['SNR_cutoff'])
                Noise = tf.random.normal(tf.shape(self.Xbdr_tmp), 0, self.GenModel.config['GaussSigma'])
                self.Xbdr_tmp, _ = self.GenModel.diffusion(self.Xbdr_tmp, self.GenModel.alpha_bar[t_val].item(), Noise) 
            elif 'VDWave' in self.Name:
                t_float, self.GenSteps = find_t(self, self.Xbdr_tmp.copy(), self.GenModel.cfg['Iter'], self.GenModel.cfg['GenSteps'], SNR_cutoff=self.GenModel.cfg['SNR_cutoff'])  
                self.Xbdr_tmp, _, noise = self.GenModel.sample_q_t_0(self.Xbdr_tmp, t_float, None, gamma_t=None)
                #t_float = (tf.cast(self.GenModel.cfg['GenSteps'], tf.float32) - 1) / (tf.cast(self.GenModel.cfg['Iter'] - 1, tf.float32))
                #self.Xbdr_tmp, _, _ = self.GenModel.sample_q_t_0(self.Xbdr_tmp, t_float, None, gamma_t=None)
                           
            self.Xbdr_Exp = np.broadcast_to(self.Xbdr_tmp[:,:,None], (self.NMiniBat, self.NParts, self.NSubGen, self.SigDim))
            self.Xbdr = np.reshape(self.Xbdr_Exp, (-1, self.SigDim))
    
            # The values of X are perturbed by randomly sampled errors along dimensions b, d, and j, while remaining constant along dimensions r and m.
            self.Xbd = np.broadcast_to(self.Xbdr_Exp[:,0,0][:,None,None], (self.NMiniBat, self.NParts, self.NSubGen, self.SigDim)).reshape(-1, self.SigDim)
            
            # Selecting sub-Xbd from Xbd for I_V_ConsX
            self.Xbd_Ext = self.Xbd.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of Xbd_Red1 : (NMiniBat*NSubGen, SigDim)
            ## The values of X are perturbed by randomly sampled errors along dimensions b, d, and t, while remaining constant along dimension m.
            self.Xbd_Red1 = self.Xbd_Ext[:, 0].reshape(self.NMiniBat*self.NSubGen, -1).copy()
            # Return shape of Xbd_Red2 : (NMiniBat, SigDim)
            ## The values of X are perturbed by randomly sampled errors along dimensions b, d, and j.
            self.Xbd_Red2 = self.Xbd_Ext[:, 0, 0].copy()
    
            # Processing Conditional information 
            ### Generating random indices for selecting true conditions
            RandSelIDXbdm = np.random.randint(0, self.TrueCond.shape[0], self.NMiniBat * self.NSubGen)
            RandSelIDXbdrm = np.random.randint(0, self.TrueCond.shape[0], self.NMiniBat * self.NParts* self.NSubGen)
            
            
            ### Selecting the true conditions using the generated indices
            # True conditions are randomly sampled at the dimensions b, d, m, and k, and constant across dimension r.
            self.CONbdm = self.TrueCond[RandSelIDXbdm]
            self.CONbdm = np.broadcast_to(self.CONbdm.reshape(self.NMiniBat, self.NSubGen, -1)[:,None], (self.NMiniBat, self.NParts, self.NSubGen, self.CondDim))
            
            # True conditions are randomly sampled across all dimensions b, d, r, m, and k.
            self.CONbdrm = self.TrueCond[RandSelIDXbdrm]
            
            # Sorting the arranged condition values in ascending order at the generation index.
            self.CONbdm_Ext = self.CONbdm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of CONbdm_Sort : (NMiniBat*NSubGen, SigDim)
            ## The conditions are sorted at the generation index after being randomly sampled across the dimensions b, d, m, and k.
            self.CONbdm_Sort = np.sort(self.CONbdm_Ext , axis=2)[:,0].reshape(self.NMiniBat*self.NSubGen, self.CondDim)
            # Return shape of CONbd_Sort : (NMiniBat, SigDim)
            ## The conditions are sorted at the dimension d after being randomly sampled across the dimensions b, d and k.
            self.CONbd_Sort = np.sort(self.CONbdm_Ext[:, 0, 0], axis=0).copy() 
            
    
            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation transforming tensors to (NMiniBat * NParts * NSubGen, SigDim) for Zs or (NMiniBat * NParts * NSubGen, CondDim) for CON. 
              After the computation, we then reverted them back to their original dimensions.
                       
                                        ## Variable cases for the signal generation ##
                    
              # Cases                             # Super Signal                    # Sub-Signal                # Target metric
              1) Xbdr + CONbdrm        ->         Sig_Xbdr_CONbdrm       ->                                    I() 
              2) Xbd + CONbdrm         ->         Sig_Xbd_CONbdrm        ->         Sig_Xbd_CONbdm             I() 
              3) Xbd + CONbdm_Sort     ->         Sig_Xbd_CONbdmSt       ->                                    I() 
              4) Xbd + CONbd_Sort      ->         Sig_Xbd_CONbdSt        ->                                    I()  
                                                  * St=Sort 
             '''
       
            # Binding the samples together, generate signals through the model 
            ListXs = [self.Xbdr, self.Xbd, self.Xbd_Red1, self.Xbd_Red2]
            Set_Xs = np.concatenate(ListXs)   
            Set_CONs = np.concatenate([self.CONbdrm, self.CONbdrm, self.CONbdm_Sort, self.CONbd_Sort]) 
            Set_Data = [Set_Xs[:,:,None], Set_CONs]
    
            # Gneraing indices for Re-splitting predictions for each case
            CaseLens = np.array([item.shape[0] for item in ListXs])
            DataCaseIDX = [0] + list(np.cumsum(CaseLens))
            
            # Choosing GPU or CPU and generating signals
            if 'Wavenet' in self.Name:
                Set_Pred = CompResource(self.GenModel, Set_Data, BatchSize=self.GenBatchSize, NSplitBatch=self.NSplitBatch, GPU=self.GPU)
                
            elif 'DiffWave' in self.Name:
                Set_Pred = DiffWAVE_Restoration(self.GenModel, np.squeeze(Set_Data[0]), Set_Data[1], GenBatchSize=self.GenBatchSize,
                                                StepInterval=self.StepInterval, GenSteps=self.GenSteps, GPU=self.GPU)
            elif 'VDWave' in self.Name:
                Set_Pred = VDiffWAVE_Restoration(self.GenModel, Set_Data[0], Set_Data[1], GenBatchSize=self.GenBatchSize, GenSteps=self.GenSteps, 
                                                 StepInterval=self.StepInterval, Noise=None, GPU=self.GPU)
              
            if self.Name == 'Wavenet_ART_Mimic':
                Set_Pred = mu_law_decode(Set_Pred)
                
            # Re-splitting predictions for each case
            self.Sig_Xbdr_CONbdrm, self.Sig_Xbd_CONbdrm, self.Sig_Xbd_CONbdmSt, self.Sig_Xbd_CONbdSt  = [Set_Pred[DataCaseIDX[i]:DataCaseIDX[i+1]] for i in range(len(DataCaseIDX)-1)] 
            
            self.Sig_Xbdr_CONbdrm = self.Sig_Xbdr_CONbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Xbd_CONbdrm = self.Sig_Xbd_CONbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Xbd_CONbdmSt = self.Sig_Xbd_CONbdmSt.reshape(self.NMiniBat, self.NSubGen, -1)
            self.Sig_Xbd_CONbdSt = self.Sig_Xbd_CONbdSt.reshape(self.NMiniBat, -1)
            
            self.Sig_Xbd_CONbdm = self.Sig_Xbdr_CONbdrm[:, 0]
            self.Sig_Xbd_CONbdm = self.Sig_Xbd_CONbdrm[:, 0]
    
    
            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics   # Function             # Code                           # Function             # Code 
                1) I_V_CONsX   q(v|Sig_Xbd_CONbdSt)   <QV_Xbd_CONbdSt>         vs     q(v|Sig_Xbd_CONbdm)   <QV_Xbd_CONbdm>
                2) I_S_CONsX   q(s|Sig_Xbd_CONbdmSt)  <QV//QS_Xbd_CONbdmSt>    vs     q(s|Sig_Xbd_CONbdrm)  <QV//QS_Xbd_CONbdrm>
                3) H()//KLD()  q(v|Sig_Xbdr_CONbdrm)  <QV_Xbdr_CONbdrm>        vs     q(v)                  <QV_Batch>       
                
                ## Metric list : I_V_CONsX, I_S_CONsX, H() or KLD()
                
             '''
    
            # Multi-method PSD calculations and metric computation
            for method in self.fft_methods:
                try:
                    # Calculate PSDs for all signal types using current method
                    MQV_Xbd_CONbdm_results = FFT_PSD_MultiMethod(self.Sig_Xbd_CONbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Xbd_CONbdSt_results = FFT_PSD_MultiMethod(self.Sig_Xbd_CONbdSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    
                    QV_Xbdr_CONbdrm_results = FFT_PSD_MultiMethod(self.Sig_Xbdr_CONbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Xbd_CONbdrm_results = FFT_PSD_MultiMethod(self.Sig_Xbd_CONbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Xbd_CONbdmSt_results = FFT_PSD_MultiMethod(self.Sig_Xbd_CONbdmSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    
                    # Extract results for current method
                    MQV_Xbd_CONbdm, _ = MQV_Xbd_CONbdm_results[method]
                    QV_Xbd_CONbdSt, _ = QV_Xbd_CONbdSt_results[method]
                    QV_Xbdr_CONbdrm, _ = QV_Xbdr_CONbdrm_results[method]
                    QV_Xbd_CONbdrm, _ = QV_Xbd_CONbdrm_results[method]
                    QV_Xbd_CONbdmSt, _ = QV_Xbd_CONbdmSt_results[method]
                    
                    if any(x is None for x in [MQV_Xbd_CONbdm, QV_Xbd_CONbdSt]):
                        print(f"Warning: Method {method} returned None results, skipping...")
                        continue
                    
                    # Apply mean operation where needed
                    MQV_Xbd_CONbdm = MQV_Xbd_CONbdm.mean(1)
                    QV_Xbd_CONbdSt = QV_Xbd_CONbdSt[:,0]
                    
                    # Permutation calculations
                    QSV_Xbdr_CONbdrm = np.concatenate([ProbPermutation(QV_Xbdr_CONbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
                    QSV_Xbd_CONbdrm = np.concatenate([ProbPermutation(QV_Xbd_CONbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
                    QSV_Xbd_CONbdmSt = ProbPermutation(QV_Xbd_CONbdmSt, WindowSize=WindowSize)
                    
                    QS_Xbdr_CONbdrm = np.sum(QSV_Xbdr_CONbdrm, axis=2)
                    QS_Xbd_CONbdrm = np.sum(QSV_Xbd_CONbdrm, axis=2)
                    QS_Xbd_CONbdmSt = np.sum(QSV_Xbd_CONbdmSt, axis=1)
                    
                    MQS_Xbd_CONbdrm = np.mean(QS_Xbd_CONbdrm, axis=1)
                    MQS_Xbdr_CONbdrm = np.mean(QS_Xbdr_CONbdrm, axis=1)
                    
                    # Calculate mutual information for current method
                    I_V_CONsX_ = MeanKLD(QV_Xbd_CONbdSt, MQV_Xbd_CONbdm)
                    I_S_CONsX_ = MeanKLD(QS_Xbd_CONbdmSt, MQS_Xbd_CONbdrm)
                    
                    print(f"Method {method} - I(V;Con'|x): {I_V_CONsX_}")
                    print(f"Method {method} - I(S;Con'|x): {I_S_CONsX_}")
                    
                    # Store results for current method
                    self.SubResDic[f'I_V_CONsX_{method}'].append(I_V_CONsX_)
                    self.SubResDic[f'I_S_CONsX_{method}'].append(I_S_CONsX_)
                    
                    # Accumulate for aggregated results
                    current_val = getattr(self, f'I_V_CONsX_{method}')
                    setattr(self, f'I_V_CONsX_{method}', current_val + I_V_CONsX_)
                    
                    current_val = getattr(self, f'I_S_CONsX_{method}')
                    setattr(self, f'I_S_CONsX_{method}', current_val + I_S_CONsX_)
                    
                    # Calculate batch QV for candidate Z location
                    QV_Batch_results = FFT_PSD_MultiMethod(np.squeeze(SubData[0])[:, None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Batch, _ = QV_Batch_results[method]
                    if QV_Batch is not None:
                        setattr(self, f'QV_Batch_{method}', QV_Batch.transpose((1,2,0)))
                    
                except Exception as e:
                    print(f"Error processing method {method}: {e}")
                    continue
            
            # Multi-method candidate Z location
            QV_Xbdr_CONbdrm_T_results = {}
            for method in self.fft_methods:
                try:
                    QV_results = FFT_PSD_MultiMethod(self.Sig_Xbdr_CONbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Xbdr_CONbdrm, _ = QV_results[method]
                    if QV_Xbdr_CONbdrm is not None:
                        QV_Xbdr_CONbdrm_T_results[method] = (QV_Xbdr_CONbdrm.reshape(self.NMiniBat, self.NGen, -1).transpose(0,2,1), None)
                except Exception as e:
                    print(f"Error in candidate Z location for method {method}: {e}")
                    continue
            
            self.LocCandZsMaxFreq(QV_Xbdr_CONbdrm_T_results, self.Xbdr, self.CONbdrm)
            
            # Restructure TrackerCand for all methods
            self.TrackerCand = {}
            for method in self.fft_methods:
                self.TrackerCand[method] = {
                    item[0]: {
                        'TrackZX': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackZX']), 
                        'TrackSecData': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackSecData']), 
                        'TrackMetrics': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackMetrics'])
                    } 
                    for item in self.TrackerCand_Temp[method].items() 
                    if len(item[1]['TrackSecData']) > 0
                }
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic, Continue=Continue)
    
        # Calculate final aggregated results for all methods
        for method in self.fft_methods:
            # Normalize by total iteration size
            current_val = getattr(self, f'I_V_CONsX_{method}')
            setattr(self, f'I_V_CONsX_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_V_CONsX_{method}'].append(getattr(self, f'I_V_CONsX_{method}'))
            
            current_val = getattr(self, f'I_S_CONsX_{method}')
            setattr(self, f'I_S_CONsX_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_S_CONsX_{method}'].append(getattr(self, f'I_S_CONsX_{method}'))



    ### -------------------------- Evaluating the performance of the model using only Z inputs  -------------------------- ###
    def Eval_Z (self, AnalSig, SampZModel, GenModel, FcLimit=0.05, WindowSize=3, Continue=True ):
        
        ## Required parameters
        self.AnalSig = AnalSig             # The data to be used for analysis.
        self.SampZModel = SampZModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        self.SecDataType = False             # The ancillary data-type: False means there is no ancillary dataset. 
        
        ## Intermediate variables
        self.Ndata = len(AnalSig) # The dimension size of the data.
        self.LatDim = SampZModel.output.shape[-1] # The dimension size of Z.
        self.SigDim = AnalSig.shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            # Method-specific candidate tracking
            self.BestZsMetrics = {}
            self.TrackerCand_Temp = {}
            
            for method in self.fft_methods:
                self.BestZsMetrics[method] = {i: [np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
                self.TrackerCand_Temp[method] = {i: {'TrackSecData': [], 'TrackZX': [], 'TrackMetrics': []} 
                                                 for i in range(1, self.MaxFreq - self.MinFreq + 2)}
                
                # Initialize method-specific result trackers
                setattr(self, f'I_V_ZjZ_{method}', 0)
            
            # Initialize method-specific result dictionaries
            self.SubResDic = {}
            self.AggResDic = {}
            
            for method in self.fft_methods:
                self.SubResDic[f'I_V_ZjZ_{method}'] = []
                self.AggResDic[f'I_V_ZjZ_{method}'] = []
        
                 
        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v) - Calculate population PSD for all methods
        QV_Pop_results = FFT_PSD_MultiMethod(self.AnalSig, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=self.fft_methods)
        
        for method, (qv_pop, _) in QV_Pop_results.items():
            if qv_pop is not None:
                setattr(self, f'QV_Pop_{method}', qv_pop)
    
        def TaskLogic(SubData):
    
            print('-------------  ',self.Name,'  -------------')
    
            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData) 
    
            # Sampling Samp_Z and Samp_Zj
            # Please note that the tensor is maintained in a reduced number of dimensions for computational efficiency in practice.
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat; 
            # The values of z are randomly sampled at dimensions b, and d.
            self.Zbd = SamplingZ(SubData, self.SampZModel, self.NMiniBat, 1, 1, 
                                BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='Modelbd', ReparaStdZj=self.ReparaStdZj)
           
            # Selecting Samp_Zjs from Zbd 
            self.Zjbd = SamplingZj (self.Zbd, self.NMiniBat, 1, 1, self.LatDim, self.NSelZ, ZjType='bd' )
    
            
            
    
            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
                                        ## Variable cases for the signal generation ##
              # Cases                     # Super Signal                   # Target metric
              1) Zbd           ->         Sig_Zbd              ->          I() // H() or KLD ()
              2) Zjbd          ->         Sig_Zjbd             ->          I() 
                                          
            ''' 
            
            # Binding the samples together, generate signals through the model 
            ListZs = [self.Zbd,    self.Zjbd]
            Set_Data = np.concatenate(ListZs)  
    
            # Gneraing indices for Re-splitting predictions for each case
            CaseLens = np.array([item.shape[0] for item in ListZs])
            DataCaseIDX = [0] + list(np.cumsum(CaseLens))
            
            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Set_Data, BatchSize=self.GenBatchSize, GPU=self.GPU)
    
            # Re-splitting predictions for each case
            self.Sig_Zbd, self.Sig_Zjbd = [Set_Pred[DataCaseIDX[i]:DataCaseIDX[i+1]] for i in range(len(DataCaseIDX)-1)] 
            
            self.Sig_Zbd = self.Sig_Zbd.reshape(self.NMiniBat, -1)
            self.Sig_Zjbd = self.Sig_Zjbd.reshape(self.NMiniBat, -1)
    
            
            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics     # Function       # Code                 # Function            # Code 
                1) I_V_ZjZ        q(v|Sig_Zjbd)    <QV_Zjbd>      vs      q(v|Sig_Zbd)          <QV_Zbd>
                2) H()//KLD()     q(v|Sig_Zbd)     <QV_Zbd>               q(v)                  <QV_Batch>       
                
                ## Metric list : I_V_ZjZ, H() or KLD()
                
            '''
            
            # Multi-method PSD calculations and metric computation
            for method in self.fft_methods:
                try:
                    # Calculate PSDs for all signal types using current method
                    QV_Zbd_results = FFT_PSD_MultiMethod(self.Sig_Zbd, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zjbd_results = FFT_PSD_MultiMethod(self.Sig_Zjbd, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    
                    # Extract results for current method
                    QV_Zbd, _ = QV_Zbd_results[method]
                    QV_Zjbd, _ = QV_Zjbd_results[method]
                    
                    if any(x is None for x in [QV_Zbd, QV_Zjbd]):
                        print(f"Warning: Method {method} returned None results, skipping...")
                        continue
                    
                    # Apply mean operation where needed
                    QV_Zbd = QV_Zbd.mean(1)
                    QV_Zjbd = QV_Zjbd.mean(1)
                    
                    # Calculate mutual information for current method
                    I_V_ZjZ_ = MeanKLD(QV_Zjbd, QV_Zbd)
                    
                    print(f"Method {method} - I(V;z'|z): {I_V_ZjZ_}")
                    
                    # Store results for current method
                    self.SubResDic[f'I_V_ZjZ_{method}'].append(I_V_ZjZ_)
                    
                    # Accumulate for aggregated results
                    current_val = getattr(self, f'I_V_ZjZ_{method}')
                    setattr(self, f'I_V_ZjZ_{method}', current_val + I_V_ZjZ_)
                    
                    # Calculate batch QV for candidate Z location
                    QV_Batch_results = FFT_PSD_MultiMethod(SubData[:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Batch, _ = QV_Batch_results[method]
                    if QV_Batch is not None:
                        setattr(self, f'QV_Batch_{method}', QV_Batch.transpose((1,2,0)))
                    
                except Exception as e:
                    print(f"Error processing method {method}: {e}")
                    continue
            
            # Multi-method candidate Z location
            QV_Zbd_T_results = {}
            for method in self.fft_methods:
                try:
                    QV_results = FFT_PSD_MultiMethod(self.Sig_Zbd, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq, methods=[method])
                    QV_Zbd, _ = QV_results[method]
                    if QV_Zbd is not None:
                        QV_Zbd_T_results[method] = (QV_Zbd.reshape(self.NMiniBat, self.NGen, -1).transpose(0,2,1), None)
                except Exception as e:
                    print(f"Error in candidate Z location for method {method}: {e}")
                    continue
            
            self.LocCandZsMaxFreq(QV_Zbd_T_results, self.Zbd)
            
            # Restructure TrackerCand for all methods
            self.TrackerCand = {}
            for method in self.fft_methods:
                self.TrackerCand[method] = {
                    item[0]: {
                        'TrackZX': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackZX']), 
                        'TrackMetrics': np.concatenate(self.TrackerCand_Temp[method][item[0]]['TrackMetrics'])
                    } 
                    for item in self.TrackerCand_Temp[method].items() 
                    if len(item[1]['TrackZX']) > 0
                }
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic, Continue=Continue)
    
        # Calculate final aggregated results for all methods
        for method in self.fft_methods:
            # Normalize by total iteration size
            current_val = getattr(self, f'I_V_ZjZ_{method}')
            setattr(self, f'I_V_ZjZ_{method}', current_val / self.TotalIterSize)
            self.AggResDic[f'I_V_ZjZ_{method}'].append(getattr(self, f'I_V_ZjZ_{method}'))