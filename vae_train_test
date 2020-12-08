import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
import cv2                          ##to eval on external data
import os

l=6    ##compressed layer (l*l)*2c
c=10
p=4
def plot_accuracy_loss(training_results):
    plt.subplot(2, 1, 1)
    plt.plot(training_results['training_loss'], 'r')
    plt.ylabel('loss')
    plt.title('training loss iterations')
    plt.show()


def show_data(data_sample):
    data_sample=data_sample.cpu().data
    plt.imshow(data_sample.numpy().reshape(28, 28), cmap='gray')
    plt.show()

def save_model_all(model, save_dir, model_name, epoch):
    """
    :param model:  nn model
    :param save_dir: save model direction
    :param model_name:  model name
    :param epoch:  epoch
    :return:  None
    """
    save_prefix = os.path.join(save_dir, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close()


class Net(nn.Module):

    # Constructor
    def __init__(self):
        super(Net, self).__init__()
        self.conv=nn.Conv2d(1,1,3)
        self.conv2=nn.Conv2d(1,1,3)
        self.conv3=nn.Conv2d(1,1,3,1,1)# 42 42 14 14 28 28
        self.lin0=nn.Linear(12*12,12*12)
        self.lin2=nn.Linear(12*12,p*p)
        self.lin22=nn.Linear(12*12,p*p)
        # self.max00 = nn.MaxPool2d(2, 1)
        self.max1 = nn.MaxPool2d(2)

        self.lin1 = nn.Linear(p*p,28*28)
        self.lin11=nn.Linear(p*p,28*28)
        self.lin33 = nn.Linear(28 * 28, 28 * 28)
        self.lin3 = nn.Linear(28 * 28, 28 * 28)
        self.bn1=nn.BatchNorm2d(1)
    # Prediction
    def forward(self, x):
        x=self.conv(x)
        x=torch.relu(x)
        x = self.conv2(x)
        x = self.max1(x)
        x = self.bn1(x)
        x = x.view(-1, 12 * 12)
        x=torch.tanh(self.lin0(x))
        es= torch.tanh(self.lin2(x))
        em = self.lin22(x)

        k=torch.empty(p*p).normal_(mean=0.0,std=1.0).cuda()
        x=em+k*es
        ds=torch.tanh(self.lin1(x))
        dm=torch.tanh(self.lin11(x))
        ds=self.lin3(ds)
        ds=torch.relu(ds)+0.05*ds
        dm=self.lin33(dm)
        dm=torch.relu(dm)+0.05*dm

        return ds,dm,es,em

def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    useful_stuff = {'training_loss': [],'validation_accuracy': []}
    for epoch in range(epochs):
        print(epoch)
        los1=0
        los2=0
        j=0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            u =x.view([-1,28*28])
            optimizer.zero_grad()
            s,m,es,em = model(x)
            a = torch.sum((torch.pow(u-m,2))/(torch.pow(s,2)+0.5)+torch.log((torch.pow(s,2)+0.5)*2)/2,1).mean()/784
            b = torch.sum((torch.pow(es,2)+torch.pow(em,2)-1-torch.log(es*es+0.000001))/2,1).mean()/(p*p)
            loss=(a+0.05*b)
            loss.backward()
            optimizer.step()
            # loss for every iteration
            useful_stuff['training_loss'].append(loss.data.item())
            los1+=a
            los2+=b
            j+=1

        print(los1.data/(j+0.1))
        print(los2.data/(j+0.1))

        if(los1.data/(j+0.1)<0.0):
            1
            save_model_all(model, "C:/Users/Rohan/Documents/Misc/", "autoen", 10)
    return useful_stuff



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

criterion = nn.MSELoss()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)



with torch.no_grad():
    torch.cuda.empty_cache()
    model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.99,0.999),eps=0.00001,weight_decay=0.0001)

model.load_state_dict(torch.load("C:/Users/Rohan/Documents/Misc/autoen_epoch_10.pt"))

training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=10)

plot_accuracy_loss(training_results)





if(1):                                                      ##to eval on external data
    while(1):
        a=input("next...")
        if(a=="e"):
            break
        img = cv2.imread('C:/Users/Rohan/Desktop/test/test3.png',0)
        imS = cv2.resize(img, (28, 28))
        imS=np.array(imS)
        imS=imS/255
        imS=(1-imS)
        imS=torch.tensor(imS)
        imS=imS.float()
        show_data(imS)
        imS=imS.view(1,1,28,28)
        imS=imS.to(device)
        z,m,es,em=model(imS)
        z=z.cuda()
        m=m.cuda()
        m=m-m.min()
        m=m/m.max()
        show_data(m)
        z=z-z.mean()
        z=torch.relu(z-0.01*z.std())
        show_data(z)
torch.cuda.empty_cache()

