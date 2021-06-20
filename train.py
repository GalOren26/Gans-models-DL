import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import  create_ui,genrate_new_images_on_existing_model, plot_loss, save_checkpoint, load_checkpoint,initialize_weights,create_tensor_board_dirs
from model import Discriminator, Generator
import my_config
from datetime import datetime

## ----------settings for models are in config.py file -----------------------
##-----------change this  parmater from "WGAN"  to "DC_GAN" to alternate between models. --------------------


config=create_ui (my_config)
MODEL=config.MODEL
if MODEL['name']=="Wgan-GP":
    Model=config.WGAN
else:
    Model=config.DC_GAN

device_name = "cuda:0"  if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)  
Model['device']=device
print( "the current device is : " +device_name)




############# define model and do pre procssing  #####################
transforms = transforms.Compose(
    [
        transforms.Resize(Model['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),(0.5,))
    ]
)
dataset = datasets.FashionMNIST(root="dataset/",train=True, transform=transforms, download=True)
loader = DataLoader(
    dataset,
    batch_size=Model['batch_size'],
    shuffle=True,
)

load_flag= Model['General']['load_existing_mode']
if load_flag:
    gen,disc, epoch,trained= load_checkpoint( Model['General']['load_dir'],Model['name'])
    if not trained:
        my_config.NUM_EPOCHS=epoch
    else: 
        real=loader.__iter__().next()[0]
        date=datetime.now().strftime("%m_%d_%H_%M")
        for i in range(Model['General'] ['number_of_real_images']):
            torchvision.utils.save_image(real[i],f"real_image_from_model_{Model['name']}_At_{date}_{i}.png")
        genrate_new_images_on_existing_model(Model,gen) 
        exit(0) 
# initialize gen and disc, note: discriminator should be called critic (since it no longer outputs between [0, 1])
# for connivance of alternate between models is name remain disc
if not load_flag:
    gen = Generator(Model).to(device)
    disc = Discriminator(Model).to(device)
initialize_weights(gen)
initialize_weights(disc)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=Model['lr_gen'], betas=(Model['beta1'], Model['beta2']))
opt_disc = optim.Adam(disc.parameters(), lr=Model['lr_disc'], betas=(Model['beta1'], Model['beta2']))


############# end define and pre procssing ##############################

# for tensorboard plotting- demonstre improvement by loss graphs and image genration process
writer_real,writer_fake,writer_gen_loss,writer_disc_loss=create_tensor_board_dirs(Model)

NUM_EPOCHS= my_config.NUM_EPOCHS
fixed_noise = torch.randn(Model['batch_size'], Model['z_dim'], 1, 1).to(device)
num_of_batches=len(loader)//Model['batch_size']
step = 0
D_loss=[]
G_loss=[]
for epoch in range(NUM_EPOCHS):
    #apply train mode to models 
    gen.train()
    disc.train()
    for batch_idx, (real,_) in enumerate(loader):
        if len(real) < Model['batch_size']:# not take in consider the last partial batch 
            break 
        real = real.to(device)
        torch.autograd.set_detect_anomaly(True)
        for _ in range(Model['disc_iter']):
            noise = torch.randn(Model['batch_size'], Model['z_dim'], 1, 1).to(device) #BCHW 
            fake = gen(noise)
            loss_disc=disc.calculate_disc_loss(disc,real,fake)
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()
        loss_gen = gen.calculate_gen_loss(disc,fake)
        gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                gen.eval()
                disc.eval()
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)# normalize is for return to range of [0,1 ]
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                D_loss.append(loss_disc.item())
                G_loss.append(loss_gen.item())
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_gen_loss.add_scalar('Discriminator Loss', D_loss[-1], global_step=step)
                writer_disc_loss.add_scalar('Generator Loss ',G_loss[-1], global_step=step)

                # writer_disc_loss.add_graph(gen,fixed_noise)
                # writer_disc_loss.add_graph(disc,real)
                writer_real.flush()
                writer_fake.flush()
                writer_gen_loss.flush()
                writer_disc_loss.flush()


            step += 1
writer_real.close()
writer_fake.close()
writer_gen_loss.close()
writer_disc_loss.close()
print("Training is  finish!... save the train results and plot loss :)")
save_checkpoint(gen,disc,None,True)
plot_loss(G_loss,D_loss,gen.model)

