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


 ## A Brief Introduction to VABAM
-VABAM is structured around five key components: Feature Extractor, Encoder, Sampler, Feature Generator, and Signal Reconstructor (Figure 4). For detailed information, please refer to our paper.

- **Feature Extractor** $\boldsymbol{g_{x}(\cdot)}$ applies cascading filters to the raw signal $y$, producing four amplitude-modulated subsets $x \in \\{x_{2^{\lambda}-1}, x_{2^{\lambda}}, \dots, x_{2^{\lambda+1}-3}, x_{2^{\lambda+1}-2}\\}$ that guide the Feature Generator.

- **Encoder** $\boldsymbol{g_{e}(\cdot)}$ learns parameters for the latent variable $Z$ and cutoff frequency $\Theta$, under two assumptions:
  - $\theta_k \sim \mathcal{U}(0, 1)$ for $k = 1, \ldots, K$, where $K = \sum_{i=1}^{\lambda} 2^i$ denotes the total number of cascading filters, increasing with depth $\zeta$, approximated by Bernoulli distributions.
  - $z_{j} \sim \mathcal{N}(\mu_{z_j}, \sigma_{z_j}^2)$ for each dimension $j$, with $j \in \\{1, 2, \ldots, J\\}$, where $J$ is a hyperparameter defining dimension count.

- **Sampler** $\boldsymbol{g_{z}(\cdot)}$ and $\boldsymbol{g_{\theta}(\cdot)}$ utilizes the reparameterization trick for backpropagation, allowing sampling of $z_{j}$ and $\theta_{k}$ for gradient flow.

- **Feature Generator** $\boldsymbol{g_{x'}(\cdot)}$ generates four principal feature signals for the Signal Reconstructor, aligning with the amplitude-modulated subsets from the Feature Extractor.

- **Signal Reconstructor** $\boldsymbol{g_{y}(\cdot)}$ reconstructs coherent signals from the feature subsets, keeping the original signal's main aspects and adding latent elements influenced by $z_{j}$ and $\theta_{k}$.

<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Training framework and Generative process.png" width="90%" alt="Intuitive Illustration of VABAM">
  
  <br>
  <em> Figure 4: Intuitive Illustration of VABAM </em>  
</p><br><br>

## Library Dependencies and Test Environment Information
VABAM's training and its post-evaluation were conducted and tested with the following libraries and their respective versions:
- Python == 3.8.16 , 3.9.18
- numpy == 1.19.5 , 1.26.0
- pandas == 1.1.4 , 2.1.1
- tensorflow == 2.4.0 , 2.10.0
- gpu == rtx3090TI , rtx4080 , rtx4090
<br><br>

