import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import math
import numpy as np
from tqdm.auto import tqdm

# =============================================================================
# Custom 1D dilated convolution layer.
# =============================================================================
class DilatedConv1d(tf.keras.layers.Layer):
    """Custom implementation of 1D dilated convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, **kwargs):
        """
        Args:
            in_channels: int, number of input channels.
            out_channels: int, number of output channels.
            kernel_size: int, size of the convolution kernel.
            dilation_rate: int, dilation rate.
        """
        super(DilatedConv1d, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        # Use add_weight to register the kernel and bias.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, self.in_channels, self.out_channels),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True, dtype=tf.float64)
        
        self.bias = self.add_weight(
            name='bias',
            shape=(1, 1, self.out_channels),
            initializer='zeros',
            trainable=True, dtype=tf.float64)
        
        super(DilatedConv1d, self).build(input_shape)

    def call(self, inputs):
        # Use tf.nn.conv1d with the given dilation rate.
        conv = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='SAME', dilations=self.dilation_rate)
        return conv + self.bias

# =============================================================================
# Feature-wise Linear Modulation
# =============================================================================
class FiLM(tf.keras.layers.Layer):
    """Feature-wise Linear Modulation (FiLM) layer for conditional signal modulation."""
    def __init__(self, channels):
        """
        Args:
            channels (int): Number of output channels to scale and shift.
        """
        super(FiLM, self).__init__()
        self.gamma_layer = tf.keras.layers.Dense(channels, activation="linear")  # Scaling factor
        self.beta_layer = tf.keras.layers.Dense(channels, activation="linear")   # Shift factor

    def call(self, inputs, condition):
        """
        Apply FiLM modulation.
        
        Args:
            inputs (Tensor): Input tensor of shape (B, T, C).
            condition (Tensor): Conditioning tensor of shape (B, C).

        Returns:
            Tensor: Modulated tensor of shape (B, T, C).
        """
        gamma = self.gamma_layer(condition)[:, None, :]  # Shape: (B, 1, C)
        beta_w = self.beta_layer(condition)[:, None, :]    # Shape: (B, 1, C)
        return gamma * inputs + beta_w  

# =============================================================================
# WaveNet Block
# =============================================================================
class Block(tf.keras.Model):
    """Modified WaveNet block for 2D condition input with FiLM."""
    def __init__(self, channels, kernel_size, dilation, last=False, **kwargs):
        """
        Args:
            channels: int, number of channels.
            kernel_size: int, size of the convolution kernel.
            dilation: int, dilation rate.
            last: bool, whether this is the last block.
        """
        super(Block, self).__init__(**kwargs)
        self.channels = channels
        self.last = last

        # Projection for diffusion step embedding.
        self.proj_embed = tf.keras.layers.Dense(channels)

        # FiLM Layer 
        self.film = FiLM(channels)

        # Dilated convolution doubling the channels (for gated activation).
        self.conv = DilatedConv1d(channels, channels * 2, kernel_size, dilation)

        if not last:
            self.proj_res = tf.keras.layers.Conv1D(channels, kernel_size=1)
        self.proj_skip = tf.keras.layers.Conv1D(channels, kernel_size=1)

    def call(self, inputs, embedding, condition):
        """
        Process the input through the WaveNet block with FiLM.
        
        Args:
            inputs: Tensor of shape [B, T, C], input signal.
            embedding: Tensor of shape [B, E], diffusion step embedding.
            condition: Tensor of shape [B, C], input conditions.
        
        Returns:
            residual: Tensor of shape [B, T, C] for the residual connection (or None if last block).
            skip: Tensor of shape [B, T, C] for the skip connection.
        """
        # Project and add the embedding (broadcast over time axis).
        emb = self.proj_embed(embedding)  # [B, C]
        x = inputs + emb[:, None]         # [B, T, C]

        # Apply FiLM 
        x = self.film(x, condition)

        # Apply the dilated convolution
        x = self.conv(x)

        # Split into two halves and apply a gated activation.
        context = tf.math.tanh(x[..., :self.channels])
        gate = tf.math.sigmoid(x[..., self.channels:])
        x = context * gate  # [B, T, C]

        # Compute the residual connection (if not the last block).
        if not self.last:
            residual = (self.proj_res(x) + inputs) / tf.cast(tf.math.sqrt(2.0), tf.float64)
        else:
            residual = None
        skip = self.proj_skip(x)
        return residual, skip

# =============================================================================
# WaveNet Model
# =============================================================================
class WaveNet(tf.keras.Model):
    """Modified WaveNet model for 2D condition input with FiLM."""
    def __init__(self, config, **kwargs):
        super(WaveNet, self).__init__(**kwargs)
        self.config = config

        # Projection layer for input signal
        self.proj = tf.keras.layers.Conv1D(config['Channels'], kernel_size=1)

        # FiLM layers
        self.film = FiLM(config['Channels'])

        # Embedding layers
        self.embed = self._embedding(config['Iter'])
        self.proj_embed = [tf.keras.layers.Dense(config['EmbeddingProj']) for _ in range(config['EmbeddingLayers'])]

        # WaveNet blocks
        self.blocks = []
        layers_per_cycle = config['NumLayers'] // config['NumCycles']
        for i in range(config['NumLayers']):
            dilation = config['DilationRate'] ** (i % layers_per_cycle)
            is_last = (i == config['NumLayers'] - 1)
            self.blocks.append(Block(config['Channels'], config['KernelSize'], dilation, last=is_last))

        # Output projection layers
        self.proj_out = [
            tf.keras.layers.Conv1D(config['Channels'], kernel_size=1, activation=tf.nn.relu),
            tf.keras.layers.Conv1D(1, kernel_size=1)
        ]

    def call(self, signal, timestep, condition):
        """
        Forward pass through WaveNet with FiLM applied.
        
        Args:
            signal: Tensor of shape [B, T], input signal.
            timestep: Tensor of shape [B], diffusion timesteps.
            condition: Tensor of shape [B, M], input conditions.

        Returns:
            Tensor of shape [B, T], predicted output signal.
        """
        # Apply FiLM at the input stage
        x = tf.nn.relu(self.proj(tf.expand_dims(signal, axis=-1)))  # [B, T, C]
        x = self.film(x, condition)  # Apply FiLM modulation

        # Gather the embedding for the given timesteps.
        embed = tf.gather(self.embed, timestep - 1)  # [B, E]
        for proj in self.proj_embed:
            embed = tf.nn.swish(proj(embed))

        # Process through each WaveNet block
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x, embed, condition)
            skip_connections.append(skip)

        # Sum the skip connections
        out = tf.add_n(skip_connections) / tf.cast(tf.math.sqrt(float(len(self.blocks))), tf.float64)

        # Apply output projection layers
        for proj in self.proj_out:
            out = proj(out)
        return tf.squeeze(out, axis=-1)

    def _embedding(self, num_steps):
        """
        Generate sinusoidal embeddings for diffusion timesteps.
        
        Args:
            num_steps: int, number of diffusion steps.
        
        Returns:
            Tensor of shape [num_steps, embedding_size] with the embeddings.
        """
        half_dim = self.config['EmbeddingSize'] // 2
        # Create a linear space from 0 to 1.
        logit = tf.linspace(0.0, 1.0, half_dim)
        exp_term = tf.cast(tf.pow(10.0, logit * self.config['EmbeddingFactor']), tf.float64)
        timesteps = tf.cast(tf.range(1, num_steps + 1), tf.float64)  # [num_steps]
        comp = timesteps[:, None] * exp_term[None, :]  # [num_steps, half_dim]
        emb = tf.concat([tf.sin(comp), tf.cos(comp)], axis=-1)  # [num_steps, embedding_size]
        return emb

# =============================================================================
# DiffWave Model with integrated training logic.
# =============================================================================
class ConditionalDiffWave(tf.keras.Model):
    """
    DiffWave: A diffusion model for signal synthesis.
    """
    def __init__(self, config, **kwargs):
        """
        Args:
            config: Configuration object with model parameters.
        """
        super(ConditionalDiffWave, self).__init__(**kwargs)
        self.wavenet = WaveNet(config)
        self.config = config

        # Beta scheduler
        self.beta = np.linspace(self.config['BetaSchedule'][0], self.config['BetaSchedule'][1], self.config['Iter']) 
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        
    def call(self, condition, noise=None, verbose=False):
        """
        Inference call: generate denoised signals from conditions.
        
        Args:
            condition: Tensor of shape [B, M], input conditions.
                 - Note: Internally, this is expanded to [B, 1, M'].
            noise: Optional tensor of shape [B, T], starting noise signal.
        
        Returns:
            Tuple:
              - signal: Tensor of shape [B, T], denoised signal.
        """
        if noise is None:
            b = tf.shape(condition)[0]
            t = self.config['SigDim']
            noise = tf.random.normal([b, t], mean=0.0, stddev=self.config['GaussSigma'], dtype=tf.float64)

        
        signal = noise
        # Iteratively denoise (reverse diffusion process)
        for t_step in range(self.config['Iter'], 0, -1):
            eps = self.pred_noise(signal, tf.fill([tf.shape(signal)[0]], t_step), condition, verbose=verbose)
            mu, sigma = self.pred_signal(signal, eps, self.alpha[t_step - 1], self.alpha_bar[t_step - 1], verbose=verbose)
            signal = mu + tf.random.normal(tf.shape(signal), mean=0.0, stddev=self.config['GaussSigma'], dtype=tf.float64) * sigma
        return signal
       
        
    def diffusion(self, signal, alpha_bar, eps=None):
        """
        Diffuse the signal to a new state.
        
        Args:
            signal: Tensor of shape [B, T], input signal.
            alpha_bar: Float or Tensor of shape [B] (cumulative product of 1-beta).
            eps: Optional noise tensor of shape [B, T].
        
        Returns:
            Tuple:
              - noised: Tensor of shape [B, T], the noised signal.
              - eps: Tensor of shape [B, T], the added noise.
        """
        if eps is None:
            eps = tf.random.normal(tf.shape(signal), mean=0.0, stddev=self.config['GaussSigma'], dtype=tf.float64)
        if isinstance(alpha_bar, tf.Tensor):
            alpha_bar = alpha_bar[:, None]
        return tf.sqrt(alpha_bar) * signal + tf.sqrt(1 - alpha_bar) * eps, eps

    
    def pred_noise(self, signal, timestep, condition, batch_size=None, verbose=False):
        """
        Predict the noise component from the noised signal using smaller batch splits if needed.
    
        Args:
            signal (tf.Tensor): Noised signal of shape [B, T].
            timestep (tf.Tensor): Diffusion timesteps of shape [B].
            condition (tf.Tensor): Conditioning tensor of shape [B, M].
            batch_size (int, optional): If specified, the forward pass is done in 
                batches of size `batch_size` along the batch dimension.
    
        Returns:
            tf.Tensor: Predicted noise of shape [B, T].
        """
        # If batch_size is None, process the entire batch at once
        if batch_size is None:
            return self.wavenet(signal, timestep, condition)
    
        B = tf.shape(signal)[0]  # Total batch size
        outputs = []
        start = 0

        if verbose: # Initialize tqdm progress bar if verbose is enabled
            pbar = tqdm(total=int(B), desc='[pred_noise] Processing mini-batches')
            
        while start < B:
            end = tf.minimum(start + batch_size, B)
            
            # Slice the batch
            signal_batch = signal[start:end]
            timestep_batch = timestep[start:end]
            condition_batch = condition[start:end]

            # Forward pass for this batch
            noise_batch = self.wavenet(signal_batch, timestep_batch, condition_batch)
            outputs.append(noise_batch)
            
            if verbose: # Update progress bar
                pbar.update(int(end - start))
                
            start += batch_size
        
        if verbose:
            pbar.close()
        
        return tf.concat(outputs, axis=0) # Concatenate along the batch dimension


    def pred_signal(self, signal, eps, alpha, alpha_bar, verbose=False):
        """
        Compute the mean and standard deviation of the denoised signal.
        
        Args:
            signal: Tensor of shape [B, T], noised signal.
            eps: Tensor of shape [B, T], estimated noise.
            alpha: float, (1 - beta) for the current timestep.
            alpha_bar: float, cumulative product of (1 - beta) up to current timestep.
        
        Returns:
            Tuple:
              - mean: Tensor of shape [B, T], estimated mean of the denoised signal.
              - stddev: float, estimated standard deviation.
        """
        signal = tf.cast(signal, dtype=tf.float64)
        mean = (signal- (1 - alpha) / np.sqrt(1 - alpha_bar) * eps) / np.sqrt(alpha)
        stddev = np.sqrt((1 - alpha_bar / alpha) / (1 - alpha_bar) * (1 - alpha))
        if verbose:
            print("[pred_signal] mean shape:", mean.shape, "stddev (scalar):", stddev)
        return mean, stddev

    def _compute_loss(self, signal, timesteps, condition):
        """
        Helper function to compute the loss for a given batch.
        
        Args:
            signal: Tensor of shape [B, T], clean signal.
            timesteps: Tensor of shape [B], diffusion timesteps.
            condition: Tensor of shape [B, M], input conditions.
                 - Note: Internally expanded to [B, 1, M'].
        
        Returns:
            loss: Scalar Tensor representing the computed loss.
        """
        
        # For Computing PSD
        def tf_fft_psd(data, min_freq=1, max_freq=51):
            data = tf.cast(data, tf.complex64)
            fft_res = tf.abs(tf.signal.fft(data))
            half_len = tf.shape(fft_res)[-1] // 2
            psd = tf.cast(tf.square(fft_res[..., :half_len]), tf.float64) / tf.cast(tf.shape(data)[-1], tf.float64)
            psd = psd[..., min_freq:max_freq]
            return psd / tf.reduce_sum(psd, axis=(-1),keepdims=True) 
            
        noise_level = tf.gather(tf.constant(self.alpha_bar, dtype=tf.float64), timesteps - 1)
        noised, noise = self.diffusion(signal, noise_level)
        eps = self.pred_noise(noised, timesteps, condition)

        denoised_signal = noised - eps
        psd_denoised = tf_fft_psd(denoised_signal) # Compute PSD of denoised_signal
        
        # Compute the total loss 
        loss = tf.reduce_mean((eps - noise)**2) + tf.reduce_mean((psd_denoised - condition) ** 2) 

        return loss

    def train_step(self, data):
        """
        Custom training step.
        
        Args:
            data: A tuple (signal, condition) where:
                  - signal: Tensor of shape [B, T], raw signal.
                  - condition: Tensor of shape [B, M], input conditions.
                        - Note: Internally expanded to [B, 1, M'].
        
        Returns:
            A dictionary mapping metric names to their current values.
        """
        signal, condition = data
        batch_size = tf.shape(signal)[0]
        timesteps = tf.random.uniform(shape=[batch_size], minval=1, maxval=self.config['Iter'] + 1, dtype=tf.int64)
        with tf.GradientTape() as tape:
            loss = self._compute_loss(signal, timesteps, condition)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        """
        Custom test/validation step.
        
        Args:
            data: A tuple (signal, condition) where:
                  - signal: Tensor of shape [B, T], raw signal.
                  - condition: Tensor of shape [B, M], input conditions.
                        - Note: Internally expanded to [B, 1, M'].
        
        Returns:
            A dictionary mapping metric names to their current values.
        """
        signal, condition = data
        batch_size = tf.shape(signal)[0]
        timesteps = tf.random.uniform(shape=[batch_size],
                                      minval=1,
                                      maxval=self.config['Iter'] + 1,
                                      dtype=tf.int64)
        loss = self._compute_loss(signal, timesteps, condition)
        return {"loss": loss}
        

# =============================================================================
# Restoration function for evaluation.
# =============================================================================

def DiffWAVE_Restoration(Model, DiffusedSignals, Condition, GenBatchSize=1, GenSteps=3, StepInterval=1, GPU=True):
    """
    Performs the diffusion-based signal restoration process.
    
    Args:
        Model: The model used for diffusion and prediction.
        Signal: The original input signal.
        Condition: Conditional inputs for the model.
        GenSteps: Number of generation steps (default: 3).
        GPU (bool): If False, forces execution on CPU.
    
    Returns:
        Restored signal after diffusion process.
    """

    device = "/GPU:0" if GPU else "/CPU:0"
    with tf.device(device):        
        Base = tf.ones([tf.shape(DiffusedSignals)[0]], dtype=tf.int64)
        
        # Initialize tqdm progress bar
        pbar = tqdm(range(GenSteps, 0, -StepInterval), desc="[Restoration] Processing Steps")
        
        for Step in pbar:
            # 1) Predict noise from the diffused sample
            PredNoise = Model.pred_noise(DiffusedSignals, Base * Step, Condition, batch_size=GenBatchSize, verbose=True)
            # 2) Restore the original signal based on the predicted noise
            PredMean, PredStd = Model.pred_signal(DiffusedSignals, PredNoise, Model.alpha[Step - 1], Model.alpha_bar[Step - 1], verbose=True)
            Random = tf.random.normal(tf.shape(DiffusedSignals), 0, Model.config['GaussSigma'], dtype=tf.float64) * PredStd
            Sample = PredMean + Random
            # 3) Update the diffused sample with the reconstructed one for the next iteration
            DiffusedSignals = Sample
            
            # Update progress bar with current step info
            pbar.set_postfix({"Current Step": Step})
        pbar.close()
    return Sample.numpy()