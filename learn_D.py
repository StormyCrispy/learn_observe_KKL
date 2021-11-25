import sys ; sys.path.append('../')
import torch.optim as optim
import torch
import seaborn as sb
import pytorch_lightning as pl
import numpy as np

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from learn_KKL.luenberger_observer_jointly import LuenbergerObserverJointly
from learn_KKL.luenberger_observer import LuenbergerObserver
from learn_KKL.system import RevDuffing
from learn_KKL.learner import Learner
from learn_KKL.utils import generate_mesh

sb.set_style('whitegrid')

# Generate the data
system = RevDuffing()
data = generate_mesh(np.array([[-1., 1.], [-1., 1.]]), 72000, method='LHS')
data, val_data = train_test_split(data, test_size=0.3, shuffle=True)


# observer = LuenbergerObserver(dim_x=2, dim_y=1, method="Autoencoder", wc=0.2,
                            #   recon_lambda=0.8)
observer = LuenbergerObserverJointly(dim_x=2, dim_y=1, method="Autoencoder", wc=0.2,
                              recon_lambda=0.8)
observer.set_dynamics(system)

# Train using pytorch-lightning and the learner class
# Options for training
trainer_options={'max_epochs': 15}
optimizer_options = {'weight_decay': 1e-6}
scheduler_options = {'mode': 'min', 'factor': 0.1, 'patience': 3,
                     'threshold': 5e-4, 'verbose': True}
stopper = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss', min_delta=5e-4, patience=3, verbose=False, mode='min')
# Instantiate learner
learner = Learner(observer=observer, system=system, training_data=data,
                  validation_data=val_data, method='Autoencoder',
                  batch_size=10, lr=5e-4, optimizer=optim.Adam,
                  optimizer_options=optimizer_options,
                  scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                  scheduler_options=scheduler_options)
# Define logger and checkpointing
logger = TensorBoardLogger(save_dir=learner.results_folder + '/tb_logs')
checkpoint_callback = ModelCheckpoint(monitor='val_loss')
trainer = pl.Trainer(
    callbacks=[stopper, checkpoint_callback], **trainer_options, logger=logger,
    log_every_n_steps=1, check_val_every_n_epoch=3)

# To see logger in tensorboard, copy the following output name_of_folder
print(f'Logs stored in {learner.results_folder}/tb_logs')
# which should be similar to jupyter_notebooks/runs/method/exp_0/tb_logs/
# Then type this in terminal:
# tensorboard --logdir=name_of_folder --port=8080

# Train and save results
trainer.fit(learner)
learner.save_results(limits=np.array([[-1, 1.], [-1., 1.]]), nb_trajs=10,
                     tsim=(0, 60), dt=1e-2,
                     checkpoint_path=checkpoint_callback.best_model_path)