import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

try:
    from modules.models import Encoder, Decoder
except:
    from my_vae.models import Encoder, Decoder

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 embed_dim=4,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False,
                 load_checkpoint=True
                 ):
        super().__init__()
        self.encoder = Encoder(double_z=True, z_channels=4, resolution=256, in_channels=3, out_ch=3, ch=128, ch_mult=[1,2,4,4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
        self.decoder = Decoder(double_z=True, z_channels=4, resolution=256, in_channels=3, out_ch=3, ch=128, ch_mult=[1,2,4,4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)

        self.quant_conv = torch.nn.Conv2d(2*4, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 4, 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        
        if load_checkpoint:
            state_dict = torch.load('/data07/v-wenjwang/ControlNet/CIConv/models/control_sd15_ini.ckpt', map_location=torch.device("cpu"))
            new_state_dict = {}
            for s in state_dict:
                if "first_stage_model" in s:
                    new_state_dict[s.replace("first_stage_model.", "")] = state_dict[s]
            self.load_state_dict(new_state_dict, strict=False)

    def encode(self, x):
        h, hs = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, hs

    def decode(self, z, hs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, hs)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior, hs = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, hs)
        return dec, posterior

if __name__ == "__main__":
    from data.laion_dataset import create_webdataset
    import torchvision

    image_dataset = create_webdataset(
        data_dir="/data06/v-wenjwang/COCO-2017/*/*.*",
    )

    import webdataset as wds
    image_dataloader = wds.WebLoader(
        dataset          =   image_dataset,
        batch_size       =   1,
        num_workers      =   8,
        pin_memory       =   True,
        prefetch_factor  =   2,
    )

    model = AutoencoderKL().cuda()

    for data in image_dataloader:
        img = data["distorted"].cuda()
        img = model(img)[0]

        torchvision.utils.save_image(img*0.5+0.5, "distorted.png")
        torchvision.utils.save_image(data["distorted"]*0.5+0.5, "original.png")

        break