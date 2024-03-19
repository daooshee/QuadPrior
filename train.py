from cldm.hack import disable_verbosity, enable_sliced_attention
disable_verbosity()

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from coco_dataset import create_webdataset
import webdataset as wds
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = 'checkpoints/COCO-final.ckpt'
coco_images = './COCO-2017/*/*.*'

# The actual batch size is batch_size * number_of_gpu
batch_size = 4 
number_of_gpu = 2
learning_rate = 1e-4
logger_freq = 1000
name = f"COCO-QuadPrior"

sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
state_dict = load_state_dict('./models/control_sd15_ini.ckpt', location='cpu')
new_state_dict = {}
for s in state_dict:
    if "cond_stage_model.transformer" not in s:
        new_state_dict[s] = state_dict[s]
model.load_state_dict(new_state_dict)
model.add_new_layers()

if resume_path != "":
    state_dict = load_state_dict(resume_path, location='cpu')
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        if '_forward_module.control_model' in sd_name:
            new_state_dict[sd_name.replace('_forward_module.control_model.', '')] = sd_param
    model.control_model.load_state_dict(new_state_dict)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = create_webdataset(
    data_dir=coco_images,
)

dataloader = wds.WebLoader(
    dataset          =   dataset,
    batch_size       =   batch_size,
    num_workers      =   2,
    pin_memory       =   False,
    prefetch_factor  =   2,
)

logger = ImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(
    dirpath                   =     'checkpoints',
    filename                  =     name + '-{epoch:02d}-{step}',
    monitor                   =     'step',
    save_last                 =     False,
    save_top_k                =     -1,
    verbose                   =     True,
    every_n_train_steps       =     10000,   # How frequent to save checkpoint
    save_on_train_epoch_end   =     True,
)

strategy = DeepSpeedStrategy(
    stage                     =     2, 
    offload_optimizer         =     True, 
    cpu_checkpointing         =     True
)

trainer = pl.Trainer(devices                   =     number_of_gpu,
                     strategy                  =     strategy,
                     precision                 =     16,
                     sync_batchnorm            =     True,
                     accelerator               =     'gpu',
                     callbacks                 =     [logger, checkpoint_callback])

# Train
trainer.fit(model, dataloader)

# If you want to continue training from a pytorch-lightning checkpoint, you can use
# trainer.fit(model, dataloader, ckpt_path="XXXX.ckpt")