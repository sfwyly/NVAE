
"""
    @Author: Junjie Jin
    @Code: Junjie Jin
    @Description: train our model (Relying on our loader framework in https://github.com/sfwyly/loader)

    UnSupervise mask image -> image

"""

from tqdm import tqdm
from utils import *
from trainer import *
from loader import *

def train(epochs = 100, val_per_epochs = 10):

    nvae_trainer = Trainer(z_dim=512)
    nvae_trainer.load_weights()
    for i in range(epochs):

        all_gen_loss = trainer(nvae_trainer)

        print(i," / ",epochs," gen_loss ",all_gen_loss)
        if ((i + 1) % val_per_epochs == 0):
            nvae_trainer.save_weights()
            val_loss = validate()
            print(i, " / ", epochs, " val_loss: ", val_loss)
        log_save() # save log

def log_save():

    pass

def trainer(nvae_trainer):
    generated_mask = True
    batch_size = 6
    train_dataloader = DataLoader(Dataset(root_path="/root/sfwy/inpainting/CeleAHQ/"), batch_size=batch_size,
                                  image_size=(64,64), shuffle=True)
    # val_dataloader = DataLoader(Dataset(root_path=configs['val_path']), batch_size=configs['batch_size'],
    #                               image_size=(configs['image_size'], configs['image_size']), shuffle=True)

    if(generated_mask):

        train_mask_dataloader = DataLoader(Dataset(root_path = "/root/sfwy/inpainting/mask/"), batch_size=batch_size,
                                  image_size=(64,64), shuffle=True, is_mask=True)
        # val_mask_dataloader = DataLoader(Dataset(root_path = configs['val_mask_path']), batch_size=configs['batch_size'],
        #                         image_size=(configs['image_size'], configs['image_size']), shuffle=True)
    length = len(train_mask_dataloader)
    all_gen_loss = []
    par = tqdm(train_dataloader)
    for i,(X_trains,_) in enumerate(par):

        if(not generated_mask):
            mask_list = getHoles((256,256),batch_size)
        else:

            mask_list = 1. - train_mask_dataloader[np.random.randint(length)][0][..., np.newaxis]

        loss_gen_total = trainer_step(X_trains * mask_list, X_trains, mask_list, nvae_trainer).numpy()

        par.set_description("train loss : %.2f"%(loss_gen_total))
        if((i+1) % 100 ==0):
            nvae_trainer.save_weights()
            par.set_description("saved train loss : %.2f" % (loss_gen_total))
        all_gen_loss.append(loss_gen_total)
    return np.mean(all_gen_loss)

def validate():

    return 0

if(__name__=="__main__"):

    train(epochs=100, val_per_epochs= 1)