import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import model
import torch.utils.data

# Adquiriendo el dataset

batch_size = 4

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

valnset = torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valnset, batch_size=batch_size, shuffle=False, num_workers=2)


# Neural Network (Modelo)
my_net = model.network

# Optimizador y función de costo

lr = 0.001

criterio = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_net.parameters(), lr)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        input, labels = data

        # Hacemos zero los gradientes
        optimizer.zero_grad()

        # Le pasamos las entradas a la red neuronal
        outputs = my_net(input)
        loss = criterio(outputs, labels) #forward
        loss.backward() #backward
        optimizer.step() #optimizer

        # Calculamos el error
        running_loss += loss.item()

        # Imprimimos el error de la epoca
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Entrenamiento terminado")


# Salvando el resultado de la red neuronal

PATH = './cifar_net.pt'
torch.save(my_net, PATH)

# Evaluando la red neuronal

dataiter = iter(valloader)
images, labels = dataiter.__next__()


def imshow(img):
    img = img / 2.0 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


imshow(torchvision.utils.make_grid(images))
print("Ground truth: ", " ".join('%s' % classes[labels[j]] for j in range(batch_size)))

my_net = torch.load(PATH)


outputs = my_net(images)
_, predicted = torch.max(outputs, 1)
print("Predicted: ", " ".join('%s' % classes[predicted[j]] for j in range(batch_size)))

# Performance


correct = 0
total = 0
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = torch.eq(predicted, labels).sum()


print('Accuracy of the network on the 10000 test images: %0.2f %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = my_net(images)
        _, predicted = torch.max(outputs, 1)
        c = np.squeeze((predicted == labels))
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %0.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



