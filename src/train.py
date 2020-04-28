import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
import cv2
from matplotlib import pyplot as plt

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model, I3D

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')




# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 1 # Run on test set every nTestInterval epochs
snapshot = 1 # Store a model every snapshot epochs
lr = 1e-3 # Learning rate

dataset = 'celeb-df' 

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
elif dataset == 'kaggle':
    num_classes = 2
elif dataset == 'celeb-df':
    num_classes = 2
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'I3D' # Options: C3D or R2Plus1D or R3D or I3D or MC3
saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        # model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        # train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
        #                 {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
        model = models.video.r2plus1d_18(pretrained=True, progress=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        train_params = model.parameters()
        # for name, param in model.named_parameters():
        #     print(name, param.data)
        #     jdks
    elif modelName == 'R3D':
        # model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        # train_params = model.parameters()
        model = models.video.r3d_18(pretrained=True, progress=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        train_params = model.parameters()
    elif modelName == 'MC3':
        # model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        # train_params = model.parameters()
        model = models.video.mc3_18(pretrained=True, progress=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        train_params = model.parameters()
        
        
    elif modelName == 'I3D':
        model = I3D.InceptionI3d(num_classes=157)
        load_file = 'rgb_charades.pt'
        model = model.to(device)
        model.load_state_dict(torch.load(load_file))
        model.replace_logits(num_classes = 2)
        train_params = model.parameters()


    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.0/375, 1.0/4388])) # standard crossentropy loss for classification 
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    #sampler = torch.utils.data.WeightedRandomSampler([1.0/212, 1.0/4388], 8, replacement=True)

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        #checkpoint = torch.load(os.path.join(save_dir, 'checkpoints', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
        #               map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        #print("Initializing weights from: {}...".format(
        #    os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        checkpoint = torch.load('run\\run_10\\models\\I3D-celeb-df_epoch-19.pth.tar', map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        print("Chekpoint loaded")

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)



    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16, preprocess=False),  batch_size= 8, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16),  batch_size=8, num_workers=4, shuffle = True )
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16) , batch_size=8, num_workers=4, shuffle = True)
    
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    training_loss_history = []
    val_loss_history = []
    for epoch in range(resume_epoch, num_epochs):
        #each epoch has a training and validation step
        # for phase in ['train', 'val']:
        #     start_time = timeit.default_timer()

        #     # reset the running loss and corrects
        #     running_loss = 0.0
        #     running_corrects = 0.0

        #     # set model to train() or eval() mode depending on whether it is trained
        #     # or being validated. Primarily affects layers such as BatchNorm or Dropout.
        #     if phase == 'train':
        #         # scheduler.step() is to be called once every epoch during training
        #         scheduler.step()
        #         model.train()
        #     else:
        #         model.eval()

        #     for inputs, labels in tqdm(trainval_loaders[phase]):
        #         # move inputs and labels to the device the training is taking place on
        #         inputs = Variable(inputs, requires_grad=True).to(device)
        #         labels = Variable(labels).to(device)
        #         optimizer.zero_grad()

        #         if phase == 'train':
        #             outputs = model(inputs)

        #         else:
        #             with torch.no_grad():
        #                 outputs = model(inputs)
                
        #         probs = nn.Softmax(dim=1)(outputs)
        #         preds = torch.max(probs, 1)[1]

        #         loss = criterion(outputs, labels.type(torch.long))

        #         if phase == 'train':
        #             loss.backward()
        #             #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #             optimizer.step()
        #             training_loss_history.append(loss.item())
        #         else:
        #             val_loss_history.append(loss.item())
                    
                
        #         running_loss += loss.item() * inputs.size(0)
        #         running_corrects += torch.sum(preds == labels.data)
        #         #print("Running loss: ", running_loss)
        #         #print("Running corrects: ", running_corrects)

        #     epoch_loss = running_loss / trainval_sizes[phase]
        #     epoch_acc = running_corrects.double() / trainval_sizes[phase]

        #     if phase == 'train':
        #         writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
        #         writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
        #     else:
        #         writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
        #         writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                
        #     save_loss(training_loss_history, val_loss_history)

        #     print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
        #     stop_time = timeit.default_timer()
        #     print("Execution time: " + str(stop_time - start_time) + "\n")

        # if epoch % save_epoch == (save_epoch - 1):
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'opt_dict': optimizer.state_dict(),
        #     }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
        #     print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            cat_probs = None
            cat_labels = None

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.type(torch.long))

                if type(cat_probs) != type(None):
                    cat_probs = torch.cat((cat_probs, probs), dim=0)
                    cat_labels = torch.cat((cat_labels, labels), dim=0)
                else:
                    cat_probs = probs
                    cat_labels = labels
                

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            save_roc_curve(cat_labels, cat_probs)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)
            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
        


    writer.close()

def save_roc_curve(labels, probs):
    print("Saving ROC Curve ...")
    fpr, tpr, _ = roc_curve(labels.detach().cpu().numpy(), probs[:,1].squeeze().detach().cpu().numpy())
    np.save('fpr', fpr)
    np.save('tpr', tpr)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")

def save_loss(train_loss_history, val_loss_history):
    # plotting loss
    print("Saving loss history ...")
    plt.figure()
    plt.plot(train_loss_history, label="Training loss")
    plt.plot(val_loss_history, label="Validation loss")
    plt.xlabel('Iteration')
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.legend()
    plt.savefig("loss_history.png")
    np.save("Train_loss_history", train_loss_history)
    np.save("Val_loss_history", val_loss_history)

if __name__ == "__main__":
    train_model()