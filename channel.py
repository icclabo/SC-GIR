import torch 
from pytorch_lightning import LightningModule
import lightning as L
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



import torch
import math

class Channels:
    def AWGN(self, Tx_sig, SNR):
        sig_pwr = torch.mean(Tx_sig ** 2)
        n_var = sig_pwr / (10 ** (SNR / 10))
        std = torch.sqrt(n_var).item()  # Convert to scalar
        noise = torch.normal(0, std, size=Tx_sig.shape, device=Tx_sig.device)
        return Tx_sig + noise
    def Rayleigh(self, Tx_sig, SNR):
        shape = Tx_sig.shape
        batch_size = shape[0]
        assert shape[-1] % 2 == 0, "Last dimension must be even for complex representation"
        num_pairs = shape[-1] // 2

        # Generate independent Rayleigh fading for each sample
        H_real = torch.normal(0, math.sqrt(1/2), size=[batch_size, 1], device=Tx_sig.device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[batch_size, 1], device=Tx_sig.device)
        H = torch.cat((torch.cat((H_real, -H_imag), dim=1), torch.cat((H_imag, H_real), dim=1)), dim=0)  # [2, batch_size]
        H = H.t().view(batch_size, 2, 2)  # [batch_size, 2, 2]

        Tx_sig_reshaped = Tx_sig.view(batch_size, num_pairs, 2)  # [batch_size, num_pairs, 2]
        Tx_sig_faded = torch.bmm(Tx_sig_reshaped, H)  # [batch_size, num_pairs, 2]
        Tx_sig_faded = Tx_sig_faded.view(shape)  # Restore original shape

        Rx_sig = self.AWGN(Tx_sig_faded, SNR)
        # Optional: Channel estimation (assuming perfect knowledge)
        H_inv = torch.inverse(H)  # [batch_size, 2, 2]
        Rx_sig_reshaped = Rx_sig.view(batch_size, num_pairs, 2)
        Rx_sig = torch.bmm(Rx_sig_reshaped, H_inv).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, SNR, K=1):
        shape = Tx_sig.shape
        batch_size = shape[0]
        assert shape[-1] % 2 == 0, "Last dimension must be even for complex representation"
        num_pairs = shape[-1] // 2

        # Rician fading parameters
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[batch_size, 1], device=Tx_sig.device)
        H_imag = torch.normal(mean, std, size=[batch_size, 1], device=Tx_sig.device)
        H = torch.cat((torch.cat((H_real, -H_imag), dim=1), torch.cat((H_imag, H_real), dim=1)), dim=0)  # [2, batch_size]
        H = H.t().view(batch_size, 2, 2)  # [batch_size, 2, 2]

        Tx_sig_reshaped = Tx_sig.view(batch_size, num_pairs, 2)  # [batch_size, num_pairs, 2]
        Tx_sig_faded = torch.bmm(Tx_sig_reshaped, H)  # [batch_size, num_pairs, 2]
        Tx_sig_faded = Tx_sig_faded.view(shape)  # Restore original shape

        Rx_sig = self.AWGN(Tx_sig_faded, SNR)
        # Optional: Channel estimation (assuming perfect knowledge)
        H_inv = torch.inverse(H)  # [batch_size, 2, 2]
        Rx_sig_reshaped = Rx_sig.view(batch_size, num_pairs, 2)
        Rx_sig = torch.bmm(Rx_sig_reshaped, H_inv).view(shape)

        return Rx_sig