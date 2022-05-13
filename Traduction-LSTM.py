import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import numpy as np
import time
import re
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm
from torchtext.data.metrics import bleu_score



logging.basicConfig(level=logging.INFO)

FILE = "../data/en-fra.txt"

writer = SummaryWriter("runs/")

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=20
BATCH_SIZE=100
DIM_EMBEDDING=32
DIM_HIDDEN=32

PATH = "/Users/sarahkerriche/Documents/Documents perso/ML:IA/S3MasterDAC-main/TME4_5_6-AMAL/TME6/src/savedModels/"

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage


class Encodeur(nn.Module):
    
    def __init__(self,lenVoc,dim_emb,dim_hidden):
        super().__init__()
        self.lenVoc=lenVoc
        self.dim_emb=dim_emb
        self.dim_hidden=dim_hidden
        self.emb=nn.Embedding(lenVoc,dim_emb,padding_idx=0)
        self.rnn=nn.LSTM(dim_emb,dim_hidden,bidirectional=True,dropout=0.3)
    
    def forward(self,x):
        return(self.rnn(self.emb(x)))
    
class Decodeur(nn.Module):
    def __init__(self,lenVoc,dim_emb,dim_hidden,voc):
        super().__init__()
        self.voc=voc
        self.lenVoc=lenVoc
        self.dim_emb=dim_emb
        self.dim_hidden=dim_hidden
        self.emb=nn.Embedding(lenVoc,dim_emb,padding_idx=0)
        self.rnn=nn.LSTM(dim_emb,dim_hidden,bidirectional=True,dropout=0.3)
        self.linear=nn.Linear(dim_hidden*2,lenVoc)
        
    def generate(self,hidden,cell_state,lenseq=None):
        cpt=0
        condition=True
        xinput=self.emb(torch.tensor(self.voc.get("SOS"))).expand(1,hidden.shape[1],-1)
        hn = hidden
        cn = cell_state
        out = []
        while (condition):
            h,(hn,cn) = self.rnn(xinput,(hn,cn))
            yhat = self.linear(hn.view(-1,2*self.dim_hidden))
            proba = nn.functional.softmax(yhat,dim=1)
            yout = torch.multinomial(proba.squeeze(), 1)
            xinput=self.emb(yout).permute(1,0,2)
            out.append(yhat)
            cpt+=1
            if (cpt==lenseq or yout.squeeze()[-1]==self.voc.get("EOS")):
                condition=False
        if len(torch.stack(out).squeeze().shape) < 3:
            return torch.stack(out).squeeze().unsqueeze(dim=0)
        else :
            return torch.stack(out).squeeze()
    
    def decode(self,x2,hnE,cnE):
        sos=torch.tensor(self.voc.get("SOS")).expand(1,x2.shape[1])
        x2=torch.vstack((sos,x2))
        xinput=self.emb(x2[:-1,:])
        hlist,(hn,cn)=self.rnn(xinput,(hnE,cnE))
        yhat = self.linear(hlist)
        return yhat
    

encoder=Encodeur(vocEng.__len__(), DIM_EMBEDDING, DIM_HIDDEN)
decoder=Decodeur(vocFra.__len__(), DIM_EMBEDDING, DIM_HIDDEN, vocFra)

epochs=500
lr=0.01
optim1=torch.optim.Adam(encoder.parameters(),lr)
optim2=torch.optim.Adam(decoder.parameters(),lr)
optim1.zero_grad()
optim2.zero_grad()
loss=nn.CrossEntropyLoss(ignore_index=0)

liste_lossTest = list()

for epoch in range(epochs):
    proba=torch.tensor([0.5,0.5])
    losstrain=0
    cpt=0
    # ----------------------- Trainning loop ----------------------
    encoder.train()
    decoder.train()
    for x1,_,x2,_ in train_loader:
        cpt=1
        nbx=torch.randint(0,int(x1.shape[1]/20)+1,size=(1,))
        #boucle pour ajout de OOV dans l'entraineement : IMPORTANT 
        for i in torch.randint(0, x1.shape[1], (nbx,)):
            for j in torch.randint(0,x1[:,i].shape[0],(1,)):
                x1[j,i]=vocEng.get("__OOV__")
        
        methode=torch.multinomial(proba,1)
        # inferecne ou techer forcing 

        if (methode.item()==0):
            #***************teacher forcing**********************
            hlistE,(hnE,cnE) = encoder(x1)
            yhat=decoder.decode(x2,hnE,cnE)
            l=loss(yhat.flatten(start_dim=0,end_dim=1),x2.flatten())
            losstrain+=l
            l.backward()
            optim2.step()
            optim1.step()
            optim2.zero_grad()
            optim1.zero_grad()
            print(epoch,':',l.item())
        else:
            #*****************inférence***************************
            hlistE,(hnE,cnE) = encoder(x1)
            yhat=decoder.generate(hnE,cnE,x2.shape[0])
            #complétion de yhat avec des pads dans le cas ou le decoder génere EOS avant la lenSeq soit atteint
            if (x2.shape[0]-yhat.shape[0] > 0):
                pad=torch.zeros(yhat.shape[2])
                pad[vocEng.get("PAD")]=1
                pad=pad.expand(x2.shape[0]-yhat.shape[0],yhat.shape[1],yhat.shape[2])
                yhat=torch.vstack((yhat,pad))
            l=loss(yhat.flatten(start_dim=0,end_dim=1),x2.flatten())
            losstrain+=l
            l.backward()
            optim2.step()
            optim1.step()
            optim2.zero_grad()
            optim1.zero_grad()
            print(epoch,':',l.item())
            
    writer.add_scalar("Loss train",losstrain/cpt,epoch)
    # ----------------------- test loop ----------------------
    encoder.eval()
    decoder.eval()
    for x1,_,x2,_ in test_loader:
        cpt=0
        ltest=0
        with torch.no_grad():
            hlistE,(hnE,cnE)=encoder(x1)
            yhat=decoder.generate(hnE,cnE,x2.shape[0])
            #complétion de yhat avec des pads dans le cas ou le decoder génere EOS avant la lenSeq soit atteint
            if (x2.shape[0]-yhat.shape[0] > 0):
                pad=torch.zeros(yhat.shape[2])
                pad[vocEng.get("PAD")]=1
                pad=pad.expand(x2.shape[0]-yhat.shape[0],yhat.shape[1],yhat.shape[2])
                yhat=torch.vstack((yhat,pad))
            ltest+=loss(yhat.flatten(start_dim=0,end_dim=1),x2.flatten())
            cpt+=1
    liste_lossTest.append(ltest/cpt)
    print("Loss moyenne Test:",epoch,':',liste_lossTest[-1])
    
    writer.add_scalar("Loss test",liste_lossTest[-1],epoch)    
    # Saving current states of the models
    torch.save(encoder.state_dict(),PATH+f"encoder{epoch}")
    torch.save(decoder.state_dict(),PATH+f"decoder{epoch}")     


def generate2(encoder,decoder,sentence="",lenseq=None):
    if len(sentence.split(" "))>1:
        x=torch.tensor([vocEng[s] for s in normalize(sentence).split(" ")]).reshape(-1,1)
    else:
        x=torch.tensor(vocEng[normalize(sentence)]).reshape(-1,1)
    hElist, (hn,cn) = encoder(x)
    xinput = decoder.emb(torch.tensor(decoder.voc.get("SOS")).to(device)).expand(1,hn.shape[1],-1)
    cpt=0
    condition=True
    out=[]
    while (condition):
        hDlist, (hn,cn)=decoder.rnn(xinput,(hn,cn))
        yhat = decoder.linear(hn.view(-1,2*DIM_HIDDEN))
        proba = nn.functional.softmax(yhat,dim=1)
        yout = torch.multinomial(proba.squeeze(), 1)
        xinput = decoder.emb(yout).unsqueeze(dim=0)
        out.append(yout)
        cpt+=1
        if (cpt==lenseq or yout.squeeze().item()==decoder.voc.get("EOS")):
            condition=False
    return torch.stack(out).squeeze()



x,_,y,y_len=next(iter(test_loader))
trad_list = list()
target_list=list()
for i in range(100):
    input = " ".join(vocEng.getwords(x[:,i] ))
    target = vocFra.getwords(y[:,i])
    trad = vocFra.getwords(generate2(encoder,decoder,input,lenseq=y[:,i].size))
    max_length = y_len[i]
    trad_blue_score = trad + [ "PAD" for i in range (max_length-len(trad)+1) ]
    trad_list.append(np.array(trad)[:max_length].tolist())
    target_list.append(np.array(target)[:max_length].tolist())
    print("Input :",input)
    print("True trad: "," ".join(target_list[-1]))
    print("Model Trad: "," ".join(trad),"\n")

print(bleu_score( target_list, trad_list))

vocEng.getwords(x[:,1])
vocFra.getwords(y[:,1])
x[:,1]

" ".join(vocFra.getwords(generate2(encoder,decoder,"i am the legend",lenseq=10)))

#save & load best models 
idx_argmin = np.argmin(liste_lossTest)
encoder.load_state_dict(torch.load(PATH+f"Encoder{idx_argmin}"))
decoder.load_state_dict(torch.load(PATH+f"Decoder{idx_argmin}"))



torch.save(encoder.state_dict(),PATH+f"BestEncoder{idx_argmin}")
torch.save(decoder.state_dict(),PATH+f"BestDecoder{idx_argmin}")


encoder.load_state_dict(torch.load(PATH+f"Encoder171"))
decoder.load_state_dict(torch.load(PATH+f"Decoder171"))

liste_lossTest
#encoder.load_state_dict(torch.load(PATH+"BestEncoder")) rajouter la meilleure epoch
#decoder.load_state_dict(torch.load(PATH+"BestDecoder")) 





