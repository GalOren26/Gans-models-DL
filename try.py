import torch, time, os, pickle, gzip
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import imageio
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorflow import summary
from datetime import datetime



MODE = 'WGAN_GP' # can be WGAN_GP or DCGAN 
train_ind = 1 # 1 for training, 0 for testing

if MODE == 'WGAN_GP' :
  CRITIC_ITER = 5
else: #DCGAN
  CRITIC_ITER = 1
BATCH_SIZE = 64 # Batch size
INPUT_SIZE = 28 # fashion mnsit input size
EPOCHS     = 50 # number of epochs
lrG = 0.0002 # learning rate generator
lrD = 0.0002 # learning rate discriminator
beta1 = 0.5
beta2 = 0.999

logs_base_dir = 'log'
os.makedirs(logs_base_dir, exist_ok=True)

log_dir_wgan_generator_train = "%s/WGAN_GP/Gen_Train_lossy/%s" % (logs_base_dir,  datetime.now().strftime("%m%d-%H%M"))
log_dir_wgan_discriminator_train = "%s/WGAN_GP/Disc_Train_loss/%s" % (logs_base_dir,  datetime.now().strftime("%m%d-%H%M"))
log_dir_dcgan_generator_train = "%s/DCGAN/Gen_Train_lossy/%s" % (logs_base_dir,  datetime.now().strftime("%m%d-%H%M"))
log_dir_dcgan_discriminator_train = "%s/DCGAN/Disc_Train_loss/%s" % (logs_base_dir,  datetime.now().strftime("%m%d-%H%M"))

if MODE == 'WGAN_GP' : 
  writer_gen_train_loss = SummaryWriter(log_dir_wgan_generator_train);
  writer_disc_train_loss = SummaryWriter(log_dir_wgan_discriminator_train);
elif MODE == 'DCGAN' : 
  writer_gen_train_loss = SummaryWriter(log_dir_dcgan_generator_train);
  writer_disc_train_loss = SummaryWriter(log_dir_dcgan_discriminator_train);

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def save_images(images, size, image_path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(image_path, image)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        if MODE == 'DCGAN':
          self.fc = nn.Sequential(nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Linear(1024, self.output_dim),  nn.Sigmoid() )
        else: #MODE == 'WGAN-GP':
            self.fc = nn.Sequential(nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Linear(1024, self.output_dim) )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class MODEL(object):
    def __init__(self):
        # parameters
        self.sample_num = 100
        self.save_dir = "/content/drive/MyDrive/ex3_305101305_312545965/"
        self.result_dir = "/content/drive/MyDrive/ex3_305101305_312545965/results"
        self.log_dir = "/content/drive/MyDrive/ex3_305101305_312545965/log"
        self.gpu_mode = True
        self.model_name = MODE
        self.z_dim = 62
        self.lambda_ = 10
        self.n_critic = CRITIC_ITER               # the number of iterations of the critic per generator iteration

        # load dataset
        transform = transforms.Compose([transforms.Resize((INPUT_SIZE,INPUT_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        self.data_loader = DataLoader(datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=INPUT_SIZE)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=INPUT_SIZE)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        self.criterion = nn.BCELoss()

        self.real_label = 1
        self.fake_label = 0

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.criterion.cuda()

        # fixed noise
        self.sample_z_ = torch.rand((BATCH_SIZE, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(BATCH_SIZE, 1), torch.zeros(BATCH_SIZE, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(EPOCHS):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // BATCH_SIZE:
                    break

                z_ = torch.rand((BATCH_SIZE, self.z_dim))
                real_labels = torch.ones(BATCH_SIZE)
                fake_labels = torch.zeros(BATCH_SIZE)

                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()
                    real_labels = real_labels.cuda()
                    fake_labels = fake_labels.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                
                if MODE == 'WGAN_GP':
                  D_real_loss = -torch.mean(D_real)
                else : #DCGAN
                  real_labels = real_labels.unsqueeze(1)
                  real_labels = real_labels.float()
                  D_real_loss = self.criterion(D_real, real_labels)

                G_ = self.G(z_)
                D_fake = self.D(G_)

                if MODE == 'WGAN_GP':
                  D_fake_loss = torch.mean(D_fake)
                else : #DCGAN
                  fake_labels = fake_labels.unsqueeze(1)
                  fake_labels = fake_labels.float()
                  D_fake_loss = self.criterion(D_fake, fake_labels)

                # gradient penalty
                alpha = torch.rand((BATCH_SIZE, 1, 1, 1))
                if self.gpu_mode:
                    alpha = alpha.cuda()

                x_hat = alpha * x_.data + (1 - alpha) * G_.data
                x_hat.requires_grad = True

                pred_hat = self.D(x_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                if MODE == 'WGAN_GP':
                  D_loss = D_real_loss + D_fake_loss + gradient_penalty
                else : #DCGAN
                  D_loss = D_real_loss + D_fake_loss

                D_loss.backward()
                self.D_optimizer.step()


                if ((iter+1) % self.n_critic) == 0:
                   # update G network
                   self.G_optimizer.zero_grad()

                   G_ = self.G(z_)
                   D_fake = self.D(G_)
                   if MODE == 'WGAN_GP':
                    G_loss = -torch.mean(D_fake)
                   else : #DCGAN
                    G_loss = self.criterion(D_fake, real_labels)

                   #self.train_hist['G_loss'].append(G_loss.item())

                   G_loss.backward()
                   self.G_optimizer.step()

                   #self.train_hist['D_loss'].append(D_loss.item())

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // BATCH_SIZE, D_loss.item(), G_loss.item()))

            writer_gen_train_loss.add_scalar('Generator Loss', G_loss.item(), epoch+1)
            writer_disc_train_loss.add_scalar('Discriminator Loss', D_loss.item(), epoch+1)
            self.train_hist['D_loss'].append(D_loss.item())
            self.train_hist['G_loss'].append(G_loss.item())
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        writer_gen_train_loss.close()
        writer_disc_train_loss.close()
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              EPOCHS, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, BATCH_SIZE)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((BATCH_SIZE, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))


gan = MODEL()