## VABAM: Variational Autoencoder for Amplitude-based Biosignal Augmentation within Morphological Identities.

This repository contains the Python implementation of VABAM along with the Conditional Mutual Information (CMI)-based evaluation metrics introduced in our ongoing research. VABAM enables the generative synthesis of pulsatile physiological signals by decoupling morphological structure from amplitude dynamics. The CMI-based metrics offer a principled, information-theoretic assessment of the model’s ability to achieve this decoupling, supporting evaluation of both structural preservation and amplitude controllability.



### Research Highlights

- **Development of the VABAM Model:** A model capable of synthesizing pulsatile physiological signals through cascaded filtering effects, namely *amplitude-based* modulation, ensuring the preservation of the signals' morphological identity.
<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Anim.%201%20VABAM%20(Our%20Model)%20Synthesis%20Results.gif" width="49%" alt="Pass-filter mechanism">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Anim.%202%20C-VAE%20Synthesis%20Results.gif" width="49%" alt="Pass-filter mechanism">
  <br>
  <em>Figure 1: Amplitude-Based Modulation of ABP via VABAM (left) vs CVAE (right) </em>  
</p>
Figure 1 shows the results of synthesizing 100 signals from a single original Arterial Blood Pressure (ABP). VABAM excels in maintaining the original morphology of signals during synthesis by avoiding phase alterations and horizontal shifts in the time axes. Conversely, conditional VAEs struggle to maintain morphological identities when PSD values are incorporated as conditional input.
<br><br>


- **Introduction of Novel Metrics:** We propose three novel metrics to provide a comprehensive evaluation of the model's synthesis and representation capabilities:
  1. **Morphological Specialization in Latent Z:** Assessing whether latent space disentangles morphological features along specific dimensions.
  2. **Morphological Preservation under Conditional Inputs :** Assessing the model’s ability to maintain waveform structure while adjusting amplitude.
  3. **Amplitude Modulation Controllability within Fixed Morphologies:** Measuring the model's capability to modulate signal amplitude accurately in accordance with the intended input.

 
