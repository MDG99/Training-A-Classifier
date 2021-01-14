import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import torchvision.transforms as transforms
import torch.utils.data
import torchvision
import torch

classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

def evaluate(model, loss_function, val_dataloader):
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []

    for data in val_dataloader:
        images, labels = data
        y_true += labels.tolist()

        with torch.no_grad():
            out = model(images)
        y_pred = torch.argmax(out, 1).tolist()
        loss = loss_function(out, labels)
        val_loss += loss.item()

    # Sklearn metrics
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    conf = confusion_matrix(y_true, y_pred)

    return val_loss, recall, precision, conf



batch_size = 10000

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

valnset = torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valnset, batch_size=batch_size, shuffle=False, num_workers=2)


PATH = './cifar_net.pt'
my_model = torch.load(PATH)


loss_function = torch.nn.CrossEntropyLoss()


val_loss, recall, precision, config = evaluate(my_model, loss_function, valloader)


print(f'Error: {val_loss}, Recall: {recall}, Precision: {precision}')
ConfusionMatrixDisplay(config, display_labels=classes).plot(plt)
plt.show()
