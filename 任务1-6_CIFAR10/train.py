"""CIFAR-10 小型 CNN：训练集/验证集分离，最终只评估一次测试集。"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path(r'D:\PythonWorkSpace\files\Deep Learning\vision_transformer_practice\data')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),
            nn.Linear(64*8*8,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self, x): return self.net(x)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct=total=0; loss_sum=0.; criterion=nn.CrossEntropyLoss()
    class_correct=torch.zeros(10,dtype=torch.long); class_total=torch.zeros(10,dtype=torch.long)
    for x,y in loader:
        x,y=x.to(device),y.to(device); logits=model(x); pred=logits.argmax(1)
        loss_sum += criterion(logits,y).item()*len(y); correct += (pred==y).sum().item(); total += len(y)
        for i in range(10):
            mask=y==i; class_total[i]+=mask.sum().item(); class_correct[i]+=((pred==y)&mask).sum().item()
    return {'loss':loss_sum/total,'accuracy':correct/total,'correct':correct,'total':total,
            'per_class_accuracy':(class_correct / class_total.clamp(min=1)).tolist()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=DEFAULT_DATA); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--seed',type=int,default=42); p.add_argument('--download',action='store_true'); a=p.parse_args()
    if a.epochs<1 or a.batch_size<1: p.error('epochs 和 batch-size 必须为正数')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)])
    trainset=datasets.CIFAR10(a.data,train=True,download=a.download,transform=transform)
    testset=datasets.CIFAR10(a.data,train=False,download=a.download,transform=transform)
    indices=torch.randperm(len(trainset),generator=torch.Generator().manual_seed(a.seed)).tolist()
    train=DataLoader(Subset(trainset,indices[5000:]),batch_size=a.batch_size,shuffle=True,num_workers=0)
    val=DataLoader(Subset(trainset,indices[:5000]),batch_size=a.batch_size,num_workers=0)
    test=DataLoader(testset,batch_size=a.batch_size,num_workers=0)
    model=Net().to(device); optim=torch.optim.Adam(model.parameters(),lr=0.001); criterion=nn.CrossEntropyLoss()
    out=HERE/'output'; out.mkdir(exist_ok=True); history=[]; best=-1; started=time.time()
    print(f'device={device}, train=45000, val=5000, test={len(testset)}',flush=True)
    for epoch in range(1,a.epochs+1):
        model.train(); loss_sum=0
        for x,y in train:
            x,y=x.to(device),y.to(device); optim.zero_grad(); loss=criterion(model(x),y); loss.backward(); optim.step(); loss_sum+=loss.item()*len(y)
        metrics=evaluate(model,val,device); row={'epoch':epoch,'train_loss':loss_sum/len(train.dataset),'val_loss':metrics['loss'],'val_accuracy':metrics['accuracy']}; history.append(row)
        print(json.dumps(row),flush=True)
        if metrics['accuracy']>best:
            best=metrics['accuracy']; torch.save(model.state_dict(),out/'best_model.pt')
    model.load_state_dict(torch.load(out/'best_model.pt',map_location=device,weights_only=True))
    result={'device':str(device),'torch_version':torch.__version__,'seed':a.seed,'epochs':a.epochs,'best_val_accuracy':best,'test':evaluate(model,test,device),'classes':testset.classes,'seconds':round(time.time()-started,1),'history':history}
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
