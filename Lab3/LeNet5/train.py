import numpy as np
import glob
import struct
import matplotlib.pyplot as plt

import LeNet5
import normalize


# 打开minst数据集
def load_mnist(path, type='train'):

    image_path = glob.glob(f'./{path}/{type}*3-ubyte')[0]
    label_path = glob.glob(f'./{path}/{type}*1-ubyte')[0]

    with open(label_path, "rb") as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, "rb") as impath:
        magic, num, rows, cols = struct.unpack('>IIII', impath.read(16))
        images = np.fromfile(impath, dtype=np.uint8).reshape(len(labels), 28*28)

    return images, labels


train_images, train_labels = load_mnist("mnist_dataset", type="train")
test_images, test_labels = load_mnist("mnist_dataset", type="t10k")

train_batch = 64  # 训练时的batch size
test_batch = 50  # 测试时的batch size
epoch = 10
lr = 1e-3

###
# 绘图所需变量
TrainTimes = []
TrainLoss = []
TrainAcc = []
TestTimes = []
TestAcc = []
plt.ion()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
###


IterNum = 0     # 迭代次数
net = LeNet5.LeNet5()

for E in range(epoch):
    batch_loss = 0
    batch_acc = 0

    epoch_loss = 0
    epoch_acc = 0

    ###
    # 训练
    for i in range(train_images.shape[0] // train_batch):
        img = train_images[i * train_batch:(i + 1) * train_batch].reshape(train_batch, 1, 28, 28)
        label = train_labels[i * train_batch:(i + 1) * train_batch]
        img = normalize.normalize(img)
        loss, pred = net.forward(img, label, is_train=True)   # 训练阶段

        epoch_loss += loss
        batch_loss += loss
        for j in range(pred.shape[0]):
            if np.argmax(pred[j]) == label[j]:
                epoch_acc += 1
                batch_acc += 1

        net.backward(lr)

        ###
        # 日志输出、图像绘制
        print_size = 10
        if (i+1) % print_size == 0:
            print(f"Epoch{E+1}:\tbatch:{i+1}\tBatch acc:{batch_acc/(train_batch * print_size):{6}.{4}}\tBatch loss:{batch_loss/(train_batch * print_size):{6}.{4}}")
            IterNum += 1
            TrainTimes.append(IterNum)
            TrainLoss.append(batch_loss / (train_batch * 50))
            TrainAcc.append(batch_acc / (train_batch * 50))

            # 绘图
            # plt.figure(1)
            # plt.clf()
            # plt.subplot(1, 2, 1)
            # plt.title('Training Loss')
            # plt.xlabel('iter', fontsize=10)
            # plt.ylabel('Loss', fontsize=10)
            # plt.plot(TrainTimes, TrainLoss, 'g-')
            #
            # plt.subplot(1, 2, 2)
            # plt.title('Training Accuracy')
            # plt.xlabel('iter', fontsize=10)
            # plt.ylabel('acc', fontsize=10)
            # plt.plot(TrainTimes, TrainAcc, 'g-')
            # plt.pause(0.5)

            batch_loss = 0
            batch_acc = 0
        ###
    ###

    print(f"[Epoch{E+1}]\tEpoch acc:{epoch_acc/train_images.shape[0]:.{4}}\tEpoch loss:{epoch_loss/train_images.shape[0]:.{4}}")
    ###
    # 测试集
    test_acc = 0
    for k in range(test_images.shape[0] // test_batch):
        img = test_images[k*test_batch:(k+1)*test_batch].reshape(test_batch, 1, 28, 28)
        label = test_labels[k*test_batch:(k+1)*test_batch]
        img = normalize.normalize(img)
        _, pred = net.forward(img, label, is_train=False)

        for j in range(pred.shape[0]):
            if np.argmax(pred[j]) == label[j]:
                test_acc += 1

    print(f"[Epoch{E+1}]\tTest Accuracy:{test_acc / test_images.shape[0]:.{4}}")
    print("---------------------------------------------------------------")
    TestTimes.append(E)
    TestAcc.append(test_acc / test_images.shape[0])
    # plt.figure(2)
    # plt.clf()
    # plt.subplot()
    # plt.title('Test Acc')
    # plt.xlabel('iter', fontsize=10)
    # plt.ylabel('acc', fontsize=10)
    # plt.plot(TestTimes, TestAcc, 'g-')
    # plt.pause(0.5)

plt.ioff()
plt.show()
