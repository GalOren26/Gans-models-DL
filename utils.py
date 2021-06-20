
import torch,os
import torch.nn as nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            

def save_checkpoint(G,D,epoch,trained):
    print("=> Saving checkpoint")
    save_dir_g = os.path.join(G.model['General']['save_dir'], G.model['name']+ '_G.pt')
    save_dir_d = os.path.join(D.model['General']['save_dir'], D.model['name']+ '_D.pt')
    torch.save({'G_Model': G,'epoch': epoch,'trained':trained}, save_dir_g)
    torch.save({'D_model':D}, save_dir_d)


def load_checkpoint(load_dir,name):
    load_dir_g= os.path.join(load_dir, name+'_G.pt')
    load_dir_d= os.path.join(load_dir, name+'_D.pt')
    print("=> Loading checkpoint")
    checkpoint_G = torch.load(load_dir_g)
    checkpoint_D = torch.load(load_dir_d)
    gen,disc=checkpoint_G['G_Model'],checkpoint_D['D_model']
    epoch,trained = checkpoint_G['epoch'],  checkpoint_G['trained']
    if trained:
        print('load model from file! validate on new genrated images :)\n')
    else:
        print('continue to train from epoch {}, see log dir for history :)\n'.format(epoch))
    return  gen,disc, epoch,trained

def create_tensor_board_dirs(Model):
    date=datetime.now().strftime("%m_%d-%H_%M")
    writer_real = SummaryWriter(f"{Model['General']['path_tensorboard']}/{Model['name']}/images/{date}/real")
    writer_fake = SummaryWriter(f"{Model['General']['path_tensorboard']}/{Model['name']}/images/{date}/fake")   
    writer_gen_loss = SummaryWriter(f"{Model['General']['path_tensorboard']}/{Model['name']}/loss/{date}/gen_loss")
    writer_disc_loss = SummaryWriter(f"{Model['General']['path_tensorboard']}/{Model['name']}/loss/{date}/disc_loss")
    return writer_real,writer_fake,writer_gen_loss,writer_disc_loss

def plot_loss(gen_loss,disc_loss, Model):
    x = range(len(gen_loss))
    plt.plot(x, disc_loss, label='D_loss',color='green')
    plt.plot(x, gen_loss, label='G_loss',color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.title('Generator and discriminator losses')
    path = os.path.join( Model['General']['save_dir'],  Model['name'] + '_current_loss.png')
    plt.savefig(path)
    plt.show()
    plt.close()

def genrate_new_images_on_existing_model(Model,gen):
    gen.eval()
    for i in range(Model['General']['number_of_fake_images']):
        date=datetime.now().strftime("%m_%d_%H_%M")
        noise = torch.randn(1,Model['z_dim'], 1, 1).to(Model['device']) #BCHW

        fake = gen(noise)
        # img_grid_fake = torchvision.utils.make_grid(fake[0], normalize=True)
        torchvision.utils.save_image(fake,f'fake_image_from_model_{Model["name"]}_At_{date}_{i}.png')

    return 



# config
# {
#     model 
#     DC_GAN
#     WGAN
#     EPOCHS
# }

def create_ui(config):
    print("Welcome to ex3 ! \n\nfollwing is menu to chose the desired settings and operation  for the model ,have fun :)  :\n")
    while (True):
        print("press 1 to see dafault settings for all sections\n")
        print("print 2 tor create new model\n")
        print("print 3 to create new images from existing model \n")
        print ("print 4 to start the program with default/chosen paramters")
        m_input=input()
        if m_input=="1":
            print(f'the genreal setting are: \n \n')
            print_settings(config.MODEL['General'])
            print (f'default model is {config.MODEL["name"]} with {config.NUM_EPOCHS} ephocs. \n')
            print(f'the settings for Wgan  is :  \n')
            print_settings(config.WGAN)
            print(f'\nthe settings for DCgan  is : \n')
            print_settings(config.DC_GAN)
            continue
        
        if m_input=="2":
            print("press 1 for DCgan\n")
            print("print 2 to wgan-gp\n")
            input2=input()
            if input2=="1":
                config.MODEL=config.DC_GAN
            else :
                config.MODEL=config.WGAN
            while(True):
                print("press 1 to launch with chosen/default paramters\n")
                print("press 2 to set model settings \n")
                input3=input()
                if input3=="1":
                    return config
                if input3=="2":
                    print(f"model parameters are: \n") 
                    print_settings(config.MODEL )
                    while(True):
                        print(f"type  parameter you want to change  without spaces as its called\n ")
                        input4= input()
                        if  input4 in config.MODEL:
                            print("set the new value:")
                            input5=input()
                            config.MODEL[f"{input4}"]=float(input5)
                            print_settings(config.MODEL )
                            print("change model settings!\n 1 to run \n 2 to change more settings\n 3 return default menu ")   
                            input6=input()
                            if input6=="1":
                                return config 
                            if input6=="2":
                                continue
                            if input6=="3":
                                break 
                        else:
                            print("seting are not exist,try again!")
                            continue
                    break
        if m_input=="3":
            print("press 1 for DCgan\n")
            print("print 2 to wgan-gp\n")
            input2=input()
            if input2=="1":
                config.MODEL=config.DC_GAN
            else :
                config.MODEL=config.WGAN
            print('type number of real images, 0 for none:\n')
            real_num=float(input())
            print('type number of real images, 0 for none:\n')
            fake_num=float(input())
            config.MODEL['number_of_real_images']=int(real_num)
            config.MODEL['General']['number_of_fake_images']=int(fake_num)
            config.MODEL['General']['load_existing_mode']=True
            return config 
        if m_input=='4':
            return config

                    
def print_settings(model):
    for key in model.keys():
        if key!= "General":
             print(f"{key}:{model[key]} ")
    print()
    