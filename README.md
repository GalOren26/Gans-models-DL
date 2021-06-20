# ex3-gan
#intro : In this project we implemented ex3 as part of deep learning course in tau. our goal is to implement the DCgan Model as well as Wgan-GP and on fashion-mnist dataset  and compare their results. 
 the architecture of the model taken from  the DCgan paper with adaption to 28*28 pictures of fashion-mnist.
for the wgan implementation we add gradient penalty to the loss (to make the critic stand the Lipschitz 1 condition) 
we trained the models for 50 ephocs with 1*10-4 learning rate (2 *10-4 for the generator in the dcgan)
with batch size of 64 and latent dimension of 100 ,and 5 cycles of crtic training per 1 generator training 
,according to wgan paper. 

## How to run :

there is sevreal way to run this project :

1.the easist one is simply enter this link https://colab.research.google.com/drive/12z7tB1rEDhahHAvYGc0oAtPjYJBmYPZH?usp=sharing
 open a colab project and type ctrl+F9 to run all cell toghter then it automaticaly clone files from this project and start the exrcice .

2.import the appended ex2.iptnb at the repo to colab \ and then do what described at 1 .

3.clone the repo and run loccaly by run the train.py file in editor ,make sure that you have all the dependencies istalled (e.g pytorch ,matplotlib ext) if not just install it from pip ;)

## chose configuration and change paramaters
after launch the project you will get a simple ui but detailed ask you what you want to do , follow it. if you want to launch the project with castum settings change the first cell which contain all the paramters to the model or if you run loccaly in the config.py file and change it as you wish.

# additinal features-save,load, log files, visulazation of data(loss and images) in tensorbord and create new genrated images 
## load and save 
The project give thhe ability to perform serialization to existing model . after the end of the each training of the model  both the nets will be saved automaticly and can be loaded  at any time .
In order to load a desired model, the path to its dir must be provided in the "General" dict that in my_config.py-default is current dir and also 'load_existing_mode' flag in the condiguration need to be change to True , this process can be done by the ui and also manually. 
 
## visualization by tensorboard 
the tenosrboard utitly is hranset in this project in order to follow the process of training and  visualize results by using it one can :
- show the loss function of both discrminator and genrator during proegression .
-  track after genrated images made by fixed gussian noise in order to see the improvemnt of the nets on those images and compare it to real images.
-  lock at the training process at any time by storing event files. 
  #### how to use
    when run on colab is already embbeded in the code and nothing you need to do beside use it . 
    when run loccaly :
    1. from pip do - pip install tensorboard
    2.open cmd on the same folder the files is runing-> and type tensorboard --logdir=logs/GAN_fashion_mnist this is the place where ecents will be save by default , you can 
     change it in the config file. 
     
 ## create new images from existing model 
 As stated after each succeeded run of the model we save files for serialization for later usage- there are two files with the extantion .pt for genrator and critic/discrminator   acoordianly. 
 to make new image you  can do it either through the ui or manualy by change 'load_existing_mode' parm  to true in the config which will made automaticly  new image in the directorty from the selected model in the pattern "{real/fake_image} from_model{model}_AT_{data}"
 model can be changed by change MODEL param in config file or through the ui . 
 
 
 have fun to test our work and create new marvelous images from fashion-mnist  :) 
 
 
  
