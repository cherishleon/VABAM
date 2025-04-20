import sys
# setting path
sys.path.append('../')

import pickle
from argparse import ArgumentParser
import os
import re
import numpy as np
import pandas as pd
from Utilities.Utilities import ReadYaml, SerializeObjects, DeserializeObjects, CompResource
from BatchBMMIEvaluation import LoadModelConfigs, LoadParams, SetVAEs, SetModels
from Models.BenchmarkCaller64 import *
from Models.VDiffWave64 import VDiffWAVE_Restoration
from Models.DiffWave64 import DiffWAVE_Restoration
from Utilities.AncillaryFunctions64 import Denorm, MAPECal, MSECal, mu_law_decode, compute_snr, scale_and_normalize
from Utilities.EvaluationMain import *


# Refer to the execution code
# python .\TabulatingBMResults.py -CP ./Config/ --GPUID 0


# Function to extract Nj value from filename
def ExtractNj(Filename):
    Match = re.search(r'Nj(\d+)\_', Filename)
    return int(Match.group(1)) if Match else 'All'
    
def Aggregation (ConfigName, ConfigPath, NJ=1,  MetricCut = 1., BatSize=3000):

    print()
    print(ConfigName)
    
    # Configuration and Object part
    print('-----------------------------------------------------' )
    print('Loading configurations and objects' )
    ## Loading the model configurations
    EvalConfigs = ReadYaml(ConfigPath)
    ModelConfigSet, ModelLoadPath = LoadModelConfigs(ConfigName, Comp=False, TypeDesig=True)
    CommonParams = EvalConfigs['Common_Param']
    ModelParams = EvalConfigs["Models"][ConfigName]

    ## Loading parameters for the evaluation
    Params = LoadParams(ModelConfigSet, {**CommonParams, **ModelParams})
    Params['Common_Info'] = EvalConfigs['Common_Info']
    Params['Spec_Info'] = EvalConfigs['Models'][ConfigName]['Spec_Info']
    ModelParams['DataSize'] = Params['EvalDataSize']
    NZs = 'All' if Params['NSelZ'] is None else Params['NSelZ']
    SNR_cutoff = Params['SNR_cutoff']
    
    ## Object Load path
    ObjLoadPath = './EvalResults/Instances/Obj_'+ConfigName+'_Nj'+str(NZs)+'.pkl'
    
    # Data part
    print('-----------------------------------------------------' )
    print('Loading data')
    
    
    #### -----------------------------------------------------   Loading data -------------------------------------------------------------------------   
    # Loading data
    SigMax = np.load('../Data/ProcessedData/'+str(Params['DataSource'])+'SigMax.pkl', allow_pickle=True)
    SigMin = np.load('../Data/ProcessedData/'+str(Params['DataSource'])+'SigMin.pkl', allow_pickle=True)
    
    if 'Wavenet' in ConfigName:
        SlidingSize = Params['SlidingSize']
    
        TrRaw = np.load('../Data/ProcessedData/'+str(Params['DataSource'])+'Tr'+Params['SigType']+'.npy')
        ValRaw = np.load('../Data/ProcessedData/'+str(Params['TestDataSource'])+'Val'+Params['SigType']+'.npy')[:Params['EvalDataSize']]
    
        TrSampled = np.load('../Data/ProcessedData/Sampled'+str(Params['DataSource'])+'Tr'+Params['SigType']+'.npy').astype('float64') # Sampled_TrData
        ValSampled = np.load('../Data/ProcessedData/Sampled'+str(Params['TestDataSource'])+'Val'+Params['SigType']+'.npy').astype('float64')[:Params['EvalDataSize']] # Sampled_ValData
        TrOut = np.load('../Data/ProcessedData/MuLaw'+str(Params['DataSource'])+'Tr'+Params['SigType']+'.npy').astype('int64') # MuLaw_TrData
        ValOut = np.load('../Data/ProcessedData/MuLaw'+str(Params['TestDataSource'])+'Val'+Params['SigType']+'.npy').astype('int64')[:Params['EvalDataSize']] # MuLaw_ValData
    
        TrInp = [TrSampled, TrRaw]
        ValInp = [ValSampled, ValRaw]
    
        GroundTruth = np.load('../Data/ProcessedData/'+str(Params['TestDataSource'])+'Val'+Params['SigType']+'.npy')[:Params['EvalDataSize']]
            
    else:
        TrInp = np.load('../Data/ProcessedData/'+str(Params['DataSource'])+'Tr'+Params['SigType']+'.npy')
        ValInp = np.load('../Data/ProcessedData/'+str(Params['TestDataSource'])+'Val'+Params['SigType']+'.npy')[:Params['EvalDataSize']]
    
    # Standardization for certain models.
    if 'DiffWave' in ConfigName or 'VDWave' in ConfigName:
        TrDeNorm = (TrInp * (SigMax[Params['SigType']] - SigMin[Params['SigType']]) + SigMin[Params['SigType']]).copy()
        ValDeNorm = (ValInp * (SigMax[Params['SigType']] - SigMin[Params['SigType']]) + SigMin[Params['SigType']]).copy()
        
        MeanSig, SigmaSig = np.mean(TrDeNorm), np.std(TrDeNorm) 
        TrInp = (TrDeNorm-MeanSig)/SigmaSig
        ValInp = (ValDeNorm-MeanSig)/SigmaSig
    
    
    if 'ART' in ConfigName:
        MaxX, MinX = SigMax['ART'], SigMin['ART']
    elif 'PLETH' in ConfigName:
        MaxX, MinX = SigMax['PLETH'], SigMin['PLETH']
    elif 'II' in ConfigName:
        MaxX, MinX = SigMax['II'], SigMin['II']
    
    Params['DataSize'] = ModelParams['DataSize']
    Params['DataSize'] = CommonParams['SigDim']
    
    
    # Model part
    print('-----------------------------------------------------' )
    print('Loading model structures')
    ## Calling Modesl
    BenchModel, _, AnalData = ModelCall (Params, ConfigName, TrInp, ValInp,  Reparam=False, LoadWeight=True, ModelSaveName=ModelLoadPath) 
    
    if 'Wavenet' not in ConfigName:
        if isinstance(AnalData, list):
            GroundTruth = AnalData[0]
        else:
            GroundTruth = AnalData
        
    # Evaluating MAPEs
    ## Prediction
    print('-----------------------------------------------------' )
    print('MAPE calculation')
    
    if 'VAE' in ConfigName:
        PredSigRec = BenchModel.predict(AnalData, batch_size=BatSize, verbose=1)
        if 'FAC' in ConfigName:
            PredSigRec = PredSigRec[1]
    
    elif 'Wavenet' in ConfigName:
        PredSigRec = CompResource(BenchModel, AnalData, BatchSize=Params['GenBatchSize'], NSplitBatch=5)
        PredSigRec = mu_law_decode(PredSigRec)
    
    elif 'VDWave' in ConfigName:
        for i in range(BenchModel.cfg['Iter']):
            t_tmp = i / float(BenchModel.cfg['Iter'] - 1)
            DiffusedSignal, _, noise = BenchModel.sample_q_t_0(AnalData[0], t_tmp, None, gamma_t=None)
            snr_val = np.mean(compute_snr(AnalData[0], DiffusedSignal[0]))
            if snr_val < SNR_cutoff:
                GenSteps = i - 1
                t_float = GenSteps / float(BenchModel.cfg['Iter'] - 1)
                break;
        DiffusedSignal, _, noise = BenchModel.sample_q_t_0(AnalData[0], t_float, None, gamma_t=None)
        PredSigRec = VDiffWAVE_Restoration(BenchModel,DiffusedSignal[0][..., None], AnalData[1], GenSteps, BenchModel.cfg['StepInterval'], GenBatchSize = Params['GenBatchSize'] )
    
        PredSigRec = scale_and_normalize(PredSigRec, SigmaSig, MeanSig, MinX, MaxX)
        GroundTruth = scale_and_normalize(GroundTruth, SigmaSig, MeanSig, MinX, MaxX)
    
    elif 'DiffWave' in ConfigName:
        for t_tmp in range(BenchModel.config['Iter']):
            Noise = tf.random.normal(tf.shape(AnalData[0]), 0, BenchModel.config['GaussSigma'])
            DiffusedSignal, _ = BenchModel.diffusion(AnalData[0], BenchModel.alpha_bar[t_tmp].item(), Noise)
            snr_val = np.mean(compute_snr(AnalData[0], DiffusedSignal))
            if snr_val < SNR_cutoff:
                GenSteps = t_tmp - 1
                break;
        DiffusedSignal, _ = BenchModel.diffusion(AnalData[0], BenchModel.alpha_bar[GenSteps].item(), Noise)
        PredSigRec = DiffWAVE_Restoration(BenchModel, DiffusedSignal, AnalData[1], GenBatchSize= Params['GenBatchSize'], StepInterval=BenchModel.config['StepInterval'], GenSteps=GenSteps )
        
        PredSigRec = scale_and_normalize(PredSigRec, SigmaSig, MeanSig, MinX, MaxX)
        GroundTruth = scale_and_normalize(GroundTruth, SigmaSig, MeanSig, MinX, MaxX)
        
    
    if Params['SecDataType'] == 'CONDIN':
        ## MAPE    
        MAPEnorm, MAPEdenorm = MAPECal(GroundTruth, np.squeeze(PredSigRec), MaxX, MinX)
        ## MSE    
        MSEnorm, MSEdenorm = MSECal(GroundTruth, np.squeeze(PredSigRec), MaxX, MinX)
    else:
        ## MAPE    
        MAPEnorm, MAPEdenorm = MAPECal(GroundTruth, np.squeeze(PredSigRec), MaxX, MinX)
        ## MSE    
        MSEnorm, MSEdenorm = MSECal(GroundTruth, np.squeeze(PredSigRec), MaxX, MinX)
    
    print('MAPEnorm : ', MAPEnorm,', MAPEdenorm : ', MAPEdenorm, ', MSEnorm : ', MSEnorm, ', MSEdenorm : ', MSEdenorm)
    
    # Evaluating Mutual information
    ## Creating new instances
    NewEval = Evaluator(Name=ConfigName)
    
    # Populating it with the saved data
    DeserializeObjects(NewEval, ObjLoadPath)
    
    # Post evaluation of KLD
    ## MetricCut: The threshold value for selecting Zs whose Entropy of PSD (i.e., SumH) is less than the MetricCut
    NewEval.SecDataType = Params['SecDataType'] if Params['SecDataType'] is not None else False
    PostSamp = NewEval.SelPostSamp(MetricCut)
    
    if 'VAE' in ConfigName:
        ## Calculation of KLD
        NewEval.GenModel = BenchModel.get_layer('ReconModel')
        NewEval.KLD_TrueGen(AnalSig=GroundTruth, PlotDist=False) 
        MeanKld_GTTG = (NewEval.KldPSD_GenTrue + NewEval.KldPSD_TrueGen) / 2
        
    else:
        # Converting the dictionary to the list type.
        PostZsList = []
        PostSecDataList = []
        
        for Freq, Subkeys in PostSamp.items():
            for Subkeys, Values in Subkeys.items():
                PostZsList.append(np.array(Values['TrackZX']))
                if 'TrackSecData' in Values.keys(): 
                    PostSecDataList.append(np.array(Values['TrackSecData']))
        
                
        # Converting the list type to the np-data type.
        PostZsList = np.concatenate(PostZsList)
        PostSecDataList = np.concatenate(PostSecDataList)
        
        if 'VDWave' in ConfigName:
            PredSigRec = VDiffWAVE_Restoration(BenchModel,PostZsList[:,:,None], PostSecDataList, GenSteps, 
                                               BenchModel.cfg['StepInterval'], GenBatchSize = Params['GenBatchSize'] )
            PredSigRec = np.squeeze(PredSigRec)
            
        elif 'DiffWave' in ConfigName:
            PredSigRec = DiffWAVE_Restoration(BenchModel,PostZsList, PostSecDataList, GenBatchSize = Params['GenBatchSize'], 
                                              GenSteps = GenSteps, StepInterval = BenchModel.config['StepInterval'])
    
        elif 'Wavenet' in ConfigName:
            PredSigRec = CompResource(BenchModel, [PostZsList[...,None] , PostSecDataList], BatchSize=Params['GenBatchSize'], NSplitBatch=5)
            PredSigRec = mu_law_decode(PredSigRec)
    
        # Calculating the KLD between the PSD of the true signals and the generated signals    
        PSDGenSamp =  FFT_PSD(PredSigRec, 'All', MinFreq = NewEval.MinFreq, MaxFreq = NewEval.MaxFreq)
        PSDTrueData =  FFT_PSD(GroundTruth, 'All', MinFreq = NewEval.MinFreq, MaxFreq = NewEval.MaxFreq)
        
        KldPSD_GenTrue = MeanKLD(PSDGenSamp, PSDTrueData)
        KldPSD_TrueGen  = MeanKLD(PSDTrueData, PSDGenSamp)
        MeanKld_GTTG = (KldPSD_GenTrue + KldPSD_TrueGen) / 2
        
        print('KldPSD_GenTrue : ', KldPSD_GenTrue,', KldPSD_TrueGen : ', KldPSD_TrueGen, ', MeanKld_GTTG : ', MeanKld_GTTG)
    
    ''' Renaming columns '''
    # r'I(V; \acute{Z} \mid Z)'
    # r'I(V;\acute{\Theta} \mid \acute{Z})'
    # r'I(S;\acute{\Theta} \mid \acute{Z})'
    
    
    MIVals = pd.DataFrame(NewEval.SubResDic)
    if 'VAE' in ConfigName:
        MIVals.columns = [r'(i) $I(V; \acute{Z} \mid Z)$', r'(ii) $I(V;\acute{\Theta} \mid \acute{Z})$', r'(iii) $I(S;\acute{\Theta} \mid \acute{Z})$']
    else:
        MIVals.columns =[r'(i) $I(V;\acute{\Theta} \mid X)$', r'(ii) $I(S;\acute{\Theta} \mid X)$']
        
    MIVals['Model'] = ConfigName
    longMI = MIVals.melt(id_vars='Model', var_name='Metrics', value_name='Values')


    return MSEnorm, MSEdenorm, MAPEnorm, MAPEdenorm, longMI, MeanKld_GTTG
    

if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--ConfigPath', '-CP', type=str, required=True, help='Set the path of the configuration to load (the name of the YAML file).')
    parser.add_argument('--MetricCut', '-MC',type=int, required=False, default=1, help='The threshold for Zs and ancillary data where the metric value is below SelMetricCut (default: 1)')
    parser.add_argument('--BatSize', '-BS',type=int, required=False, default=5000, help='The batch size during prediction.')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    
    args = parser.parse_args() # Parse the arguments
    YamlPath = args.ConfigPath
    MetricCut = args.MetricCut
    BatSize = args.BatSize
    GPU_ID = args.GPUID


    ## GPU selection
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID)
    
    # TensorFlow memory configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            gpu = gpus[0]  # Fix the index as zero since GPU_ID has already been given. 
            tf.config.experimental.set_memory_growth(gpu, False)
            tf.config.experimental.set_virtual_device_configuration
            (
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*23.5))]  
            )
        except RuntimeError as e:
            print(e)            
    
                 
                 
                 
    #### -----------------------------------------------------  Conducting tabulation --------------------------------------------------------------
             
    # Object part
    print('-----------------------------------------------------' )
    print('Scanning objects' )
    print('-----------------------------------------------------' )
    ObjLoadPath = './EvalResults/Instances/'
    FileList = os.listdir(ObjLoadPath)
    FileList = [file for file in FileList if file.endswith('.pkl')]
    print('FileList')
    print(FileList)
    
    ## Loading the model configuration lists
    EvalConfigList = os.listdir(YamlPath) # Retrieve a list of all files in the YamlPath directory.
    EvalConfigList = [i for i in EvalConfigList if 'Eval' in i] # Filter the list to include only files that contain 'Eval' in their names.
    
    # loop
    for Filename in FileList:
        # Extracts the string between 'Obj_' and '_Nj'
        ConfigName =re.search(r'Obj_(.*?)_Nj', Filename).group(1) 
        Match = re.search(r'(VAE)', ConfigName).group(1) if re.search(r'(VAE)', ConfigName) else None
        
        if 'VAE' in ConfigName:
            if 'II' in ConfigName:
               ConfigPath = 'EvalConfigII_VAE.yml'
            elif 'ART' in ConfigName:
                ConfigPath = 'EvalConfigART_VAE.yml'
        else:
            if 'II' in ConfigName:
               ConfigPath = 'EvalConfigII_Other.yml'
            elif 'ART' in ConfigName:
                ConfigPath = 'EvalConfigART_Other.yml'
                
        ConfigPath = YamlPath + ConfigPath
        NJ = ExtractNj(Filename)
    
        
        # Perform aggregation (custom function) and retrieve results.
        MSEnorm, MSEdenorm, MAPEnorm, MAPEdenorm, longMI, MeanKld_GTTG = Aggregation(ConfigName, ConfigPath, NJ=NJ, MetricCut=MetricCut, BatSize=BatSize)
    
        
        # Save the MItables to a CSV file.
        longMI.to_csv('./EvalResults/Tables/MI_' + str(ConfigName) +'_Nj'+str(NJ) + '.csv', index=False)
    
        # Save the AccKLDtables to a CSV file.
        DicRes = {'Model': [ConfigName] , 'MeanKldRes': [MeanKld_GTTG], 'MSEnorm':[MSEnorm] , 'MSEdenorm': [MSEdenorm], 'MAPEnorm': [MAPEnorm], 'MAPEdenorm': [MAPEdenorm] }
        AccKLDtables = pd.DataFrame(DicRes)
        AccKLDtables.to_csv('./EvalResults/Tables/AccKLD_' + str(ConfigName) + '_Nj'+str(NJ) +'.csv', index=False)
            
