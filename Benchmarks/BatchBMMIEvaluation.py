import sys
# setting path
sys.path.append('../')

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import copy
from argparse import ArgumentParser

from Benchmarks.Models.BenchmarkCaller64 import *
from Utilities.EvaluationMain import *
from Utilities.Utilities import ReadYaml, SerializeObjects, DeserializeObjects, LoadModelConfigs, LoadParams


# Refer to the execution code
# python .\BatchBMMIEvaluation.py --Config EvalConfigART_VAE --GPUID 0 
# python .\BatchBMMIEvaluation.py --Config EvalConfigART_VAE --ConfigSpec ConVAE_ART_30_Mimic --GPUID 0    

 
#### -----------------------------------------------------   Defining model structure -----------------------------------------------------------------    
def SetVAEs(Params, ConfigName, TrInp, ValInp, ModelLoadPath):
    # Calling Modesl
    BenchModel, _, AnalData = ModelCall (Params, ConfigName, TrInp, ValInp, LoadWeight=True,  
                                         Reparam=True, ReparaStd=Params['ReparaStd'], ModelSaveName=ModelLoadPath) 
        
    ## The generation model for evaluation
    GenModel = BenchModel.get_layer('ReconModel')
    
    ## The sampling model for evaluation
    Inp_Enc = BenchModel.get_layer('Inp_Enc')

    if 'VDV' in ConfigName:
        Zs = tf.concat([BenchModel.get_layer('Zs'+str(i)).output for i in range(len(Params['LatDim']))], axis=-1)
    else:
        Zs = BenchModel.get_layer('Zs').output
    
    if Params['SecDataType'] == 'CONDIN':
        Inp_Cond = BenchModel.get_layer('Inp_Cond')
        SampZModel = Model([Inp_Enc.input, Inp_Cond.input], Zs)
    else:
        SampZModel = Model(Inp_Enc.input, Zs)
    return SampZModel, GenModel, AnalData
    
def SetModels(Params, ConfigName, TrInp, ValInp, ModelLoadPath):
    GenModel, _, AnalData = ModelCall (Params, ConfigName, TrInp, ValInp, LoadWeight = True, ModelSaveName=ModelLoadPath) 
    return GenModel, AnalData


if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the YAML file).')
    parser.add_argument('--ConfigSpec', nargs='+', type=str, required=False, 
                        default=None, help='Set the name of the specific configuration to load (the name of the model config in the YAML file).')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    parser.add_argument('--Continue', type=bool, required=False, default=False, help='Continue from previous checkpoint')

    
    args = parser.parse_args() # Parse the arguments
    ConfigName = args.Config
    ConfigSpecName = args.ConfigSpec
    GPU_ID = args.GPUID
    Continue = args.Continue
    
    YamlPath = './Config/'+ConfigName+'.yml'
    EvalConfigs = ReadYaml(YamlPath)

    ## GPU selection
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID)

    # TensorFlow memory configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            gpu = gpus[0]  # Fix the index as zero since GPU_ID has already been given. 
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration
            (
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*23.5))]  
            )
        except RuntimeError as e:
            print(e)         

    
    # Checking whether the path to save the object exists or not.
    if not os.path.exists('./EvalResults/Instances/') :
        os.makedirs('./EvalResults/Instances/')

    # Checking whether the path to save the SampZj exists or not.
    if not os.path.exists('./Data/IntermediateData/') :
        os.makedirs('./Data/IntermediateData/')
    
                 
                 
                 
    #### -----------------------------------------------------  Conducting batch evaluation --------------------------------------------------------------
                 
    for ConfigName in EvalConfigs['Models'].keys():
        
        if ConfigSpecName is not None: 
            if ConfigName not in ConfigSpecName:
                continue
                
        print()
        print(ConfigName)
        print()

        #### -----------------------------------------------------  Setting evaluation environment ----------------------------------------------------------
        # Loading the model configurations
        ModelConfigSet, ModelLoadPath = LoadModelConfigs(ConfigName, Comp=False, TypeDesig=True)
        CommonParams = EvalConfigs['Common_Param']
        ModelParams = EvalConfigs["Models"][ConfigName]

        # Loading parameters for the evaluation
        Params = LoadParams(ModelConfigSet, {**CommonParams, **ModelParams})
        Params['Common_Info'] = EvalConfigs['Common_Info']
        Params['Spec_Info'] = EvalConfigs['Models'][ConfigName]['Spec_Info']
        Params['DataSize'] = Params['EvalDataSize']
        
        DataSource = Params['DataSource']
        TestDataSource = Params['TestDataSource']
        SigType = Params['SigType']

        
        #### -----------------------------------------------------   Loading data -------------------------------------------------------------------------   
        # Loading data
        if 'Wavenet' in ConfigName:
            SlidingSize = Params['SlidingSize']
        
            TrRaw = np.load('../Data/ProcessedData/'+str(DataSource)+'Tr'+str(SigType)+'.npy')
            ValRaw = np.load('../Data/ProcessedData/'+str(TestDataSource)+'Test'+str(SigType)+'.npy')[:Params['EvalDataSize']]
        
            TrSampled = np.load('../Data/ProcessedData/Sampled'+str(DataSource)+'Tr'+str(SigType)+'.npy').astype('float64') # Sampled_TrData
            ValSampled = np.load('../Data/ProcessedData/Sampled'+str(TestDataSource)+'Test'+str(SigType)+'.npy').astype('float64')[:Params['EvalDataSize']] # Sampled_ValData
            TrOut = np.load('../Data/ProcessedData/MuLaw'+str(DataSource)+'Tr'+str(SigType)+'.npy').astype('int64') # MuLaw_TrData
            ValOut = np.load('../Data/ProcessedData/MuLaw'+str(TestDataSource)+'Test'+str(SigType)+'.npy').astype('int64')[:Params['EvalDataSize']] # MuLaw_ValData
    
            TrInp = [TrSampled, TrRaw]
            ValInp = [ValSampled, ValRaw]
            
            
        else:
            TrInp = np.load('../Data/ProcessedData/'+str(DataSource)+'Tr'+str(SigType)+'.npy')
            ValInp = np.load('../Data/ProcessedData/'+str(TestDataSource)+'Test'+str(SigType)+'.npy')[:Params['EvalDataSize']]

    
        # Standardization for certain models.
        if 'DiffWave' in ConfigName or 'VDWave' in ConfigName:
            SigMax = np.load('../Data/ProcessedData/'+str(DataSource)+'SigMax.pkl', allow_pickle=True)
            SigMin = np.load('../Data/ProcessedData/'+str(DataSource)+'SigMin.pkl', allow_pickle=True)
            TrDeNorm = (TrInp * (SigMax[str(SigType)] - SigMin[str(SigType)]) + SigMin[str(SigType)]).copy()
            ValDeNorm = (ValInp * (SigMax[str(SigType)] - SigMin[str(SigType)]) + SigMin[str(SigType)]).copy()
            
            MeanSig, SigmaSig = np.mean(TrDeNorm), np.std(TrDeNorm) 
            TrInp = (TrDeNorm-MeanSig)/SigmaSig
            ValInp = (ValDeNorm-MeanSig)/SigmaSig


        #### -----------------------------------------------------  Conducting Evalution -----------------------------------------------------------------    
        # Is the value assigned by ArgumentParser or assigned by YML?
        NZs = 'All' if Params['NSelZ'] is None else Params['NSelZ']
        print('NZs : ', NZs)


        # Setting the model
        if 'VAE' in ConfigName:
            SampModel, GenModel, AnalData = SetVAEs(Params, ConfigName, TrInp, ValInp, ModelLoadPath)
        else:
            GenModel, AnalData = SetModels(Params, ConfigName, TrInp, ValInp, ModelLoadPath)
                    
       
        # Clearing the session before building the model
        tf.keras.backend.clear_session()

        # Object save path
        ObjSavePath = './EvalResults/Instances/Obj_'+ConfigName+'_Nj'+str(NZs)+'.pkl'
        SampZjSavePath = './Data/IntermediateData/'+ConfigName+'_Nj'+str(NZs)+'.npy'
            
        # Instantiation 
        Eval = Evaluator(MinFreq = Params['MinFreq'], MaxFreq = Params['MaxFreq'], SimSize = Params['SimSize'], NMiniBat = Params['NMiniBat'], NParts = Params['NParts'], 
               NSubGen = Params['NSubGen'], ReparaStdZj = Params['ReparaStdZj'], NSelZ = NZs, SampBatchSize = Params['SampBatchSize'],  SelMetricType = Params['SelMetricType'],
               SelMetricCut = Params['SelMetricCut'], GenBatchSize = Params['GenBatchSize'], GPU = Params['GPU'], Name=ConfigName+'_Nj'+str(NZs), fft_methods=['fft', 'welch_evo', 'matching_pursuit'])
        
        if Params['SecDataType'] is None:
            Eval.Eval_Z(AnalData, SampModel, GenModel, Continue=Continue)
        else:
            if 'VAE' in ConfigName:
                Eval.Eval_ZCON(AnalData,  SampModel, GenModel, Continue=Continue, SecDataType=Params['SecDataType'])
            
            elif 'Wavenet' in ConfigName:
                Eval.Eval_XCON([AnalData[0], AnalData[1]], GenModel, Continue=Continue, NSplitBatch=Params['NSplitBatch'], SecDataType='CONDIN')
            
            elif 'DiffWave' or 'VDWave' in ConfigName:
                Eval.Eval_XCON([AnalData[0], AnalData[1]], GenModel, GenSteps=Params['GenSteps'], StepInterval=Params['StepInterval'], Continue=Continue, SecDataType='CONDIN')
            else:
                assert False, "Please verify if ConfigName is properly provided."

        
        # Selecting post Samp_Zj for generating plausible signals
        SelPostSamp = Eval.SelPostSamp( Params['SelMetricCut'], SavePath=SampZjSavePath)


        # Evaluating KLD (P || K)
        #Eval.KLD_TrueGen(SecDataType = Params['SecDataType'], RepeatSize = 1, PlotDist=False) 

        # Saving the instance's objects to a file
        SerializeObjects(Eval, Params['Common_Info']+Params['Spec_Info'], ObjSavePath)
            


