General={
        'path_tensorboard':'logs/GAN_fashion_mnist',
        'save_dir':'',#current dir  -overwrite forsave at specific place -defult is current directory 
        'load_dir':'',#current dir -overwrite for the directory where  yours deired model is  ,without the name of model weill be load the file associate with the current module (can be change by change MODEL arg) 
        'load_existing_mode':False, # if this paramter is true and model is end training (save in the state of model) new image will be genrated.
        'number_of_real_images':1,
        'number_of_fake_images':1,
        }

DC_GAN={ 'name': "DCgan",
        'lr_gen' : 2e-4, 
        'lr_disc' : 1e-4, 
        'batch_size': 64, #maybe change to 128
        'image_size': 32,#will resize from 28*28 size later  
        'num_of_input_channels':1, 
        'num_of_output_channels':1, # same as input - 1 dim for gray image 
        'z_dim': 128 ,# according to source paper  
        'disc_iter': 1, # the number of time train discriminator per generator cycle train .  
        'beta1':0.5, #from paper- for adam optimizer 
        'beta2':0.999, #from paper- for adam optimizer 
        'features_multiplyer':64, #for define the models  
        'General':General
        }
WGAN={ 'name': "Wgan-GP",
        'lr_gen' : 1e-4, 
        'lr_disc':1e-4,
        'batch_size': 64, 
        'image_size': 32, #will resize from 28*28 size later  
        'num_of_input_channels':1, 
        'num_of_output_channels':1, # same as input - 1 dim for gray image 
        'gen_iter':1,
        'z_dim': 100 ,# according to source code implemantion  
        'disc_iter': 5, # the number of time train discriminator per generator cycle train .  
        'beta1':0,#from paper -for adam optimizer 
        'beta2':0.9, #from paper- for adam optimizer 
        'lambada_gp':10,
        'features_multiplyer':64, #for define the models  
        'General':General
        }       
MODEL=DC_GAN
NUM_EPOCHS= 50
