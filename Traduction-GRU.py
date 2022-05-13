import logging
from torch._C import dtype
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
import datetime
import time
import re
from torch.utils.tensorboard import SummaryWriter
import  sentencepiece  as spm
from generate import generate_beam, p_nucleus


torch.manual_seed(2) # do not remove this line if model with first Dataset (Vocabulary is bound to aleatory, you need to save them to change the seed)

logging.basicConfig(level=logging.INFO)

FILE = "data/en-fra.txt"

# writer = SummaryWriter("runs/tag-"+time.asctime())

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
    SOS = 2#introduit que la partie decodeur 
    OOVID = 3 #OUT OF VOCABULARY , WORD never seen 

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID #intrdouit uniquement avec word2piece
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



class TradDataset():#Wordpiece: token par mot 
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]
    def getSizes(self):
    	return len(vocEng), len(vocFra)

class TradDataset_2():#Sentencepiece heureu se: deux toekn crées 
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.s = spm.SentencePieceProcessor(model_file='model.model')
        self.sentences =[]
        Vocabulary.PAD = self.s.encode("PAD",out_type=int)[1]
        Vocabulary.EOS = self.s.encode("EOS",out_type=int)[1]
        # print(Vocabulary.PAD,Vocabulary.EOS)
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            # print(orig)
            self.sentences.append((torch.tensor(self.s.encode(orig + "EOS", out_type=int)), torch.tensor(self.s.encode(dest + "EOS", out_type=int))))
            # self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]
    def getSizes(self):
    	return 10000, 10000

def collate(batch):#prepare batch of dataset
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len#x, longueur origine (car packedqequence sans le pad),


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#CUDA for activate GPU calculations

class Encoder(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size):
        
        super(Encoder, self).__init__()
        print(vocab_size)
        self.emb = torch.nn.Embedding(vocab_size, embedding_size,padding_idx = Vocabulary.PAD)
        self.gru = torch.nn.GRU(input_size = embedding_size,hidden_size = hidden_size)
        
        
    def forward(self, input):
        x, y = input
        batch_sizes = x.batch_sizes
        x = self.emb(x)

        x = torch.nn.utils.rnn.pack_padded_sequence(x.data,batch_sizes,enforce_sorted=False)#packer sequence deja padder
        _, hn = self.gru(x)#etat final du etat final , un seul emb repsresente tt la sequence 
        return (hn, y)

class ModeSelecter(nn.Module):#teacher forcing or inference teaching proba= 0.5
    '''
        Select if learning contraint or not
    '''
    def __init__(self,p = 0.5):
        super(ModeSelecter, self).__init__()
        self.p = 0.5
        self.test = False
    def forward(self, input):
        if self.test: return (input, "non contraint")
        return (input, "contraint" if torch.rand(1)[0] < self.p else "non contraint")

    def isTest(self,bool):
        self.test = bool

class Decoder(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,class_size,random = False):
        
        super(Decoder, self).__init__()
        print(vocab_size)
        self.emb = torch.nn.Embedding(vocab_size, embedding_size,padding_idx = Vocabulary.PAD)#padding is added 
        self.gru = torch.nn.GRU(input_size = embedding_size,hidden_size = hidden_size)#vecteur hn donné par encoder
        self.classifieur = torch.nn.Linear(hidden_size,class_size)#chaque fois un truq decode il faut predir le mot, classifier le mot
        self.random = random
        self.vocab_size = vocab_size
        
    def forward(self, input):
        (h_0, y), mode = input
        
        
        # print(mode)
        if mode == "contraint":#teacher forcing

            batch_sizes = y.batch_sizes

            # add SOS at start
            starts = torch.tensor([[Vocabulary.SOS] * y.shape[1]],dtype=torch.int)
            starts = starts.to(device)
            y = torch.cat((starts,y[:-1])) #prend le y:réalité , ajoute le SOS

            #hiddent se fait automatiquement , ici inoput=output
            y = self.emb(y)#chercher l'em (y*embd_size)
            y = torch.nn.utils.rnn.pack_padded_sequence(y.data,batch_sizes,enforce_sorted=False)
            output, _ = self.gru(y,h_0)#list de tout les etat cache , predire lemot actuele 
            output, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(output)
            output = self.classifieur(output)
            output.batch_sizes = batch_sizes
            return output, mode
        else:#teacher inference
            
            res = [self.generate(h_0[0][i].unsqueeze(0).unsqueeze(0),len_seq = y.shape[0]) for i in range(h_0.shape[1])]
            if y.shape[0] - res[0].shape[0] != 0: # to be sure that same size as before
                to_add = torch.zeros((y.shape[0] - res[0].shape[0],res[0].shape[1]))
                to_add[:,Vocabulary.PAD] = 1
                to_add = to_add.to(device)#GPU activate 
                res[0] = torch.cat([res[0],to_add]) 
            res = pad_sequence(res,batch_first=True,padding_value=Vocabulary.PAD)

            res = res.transpose(0,1)
            return res, mode
            # return self.generate(h_0)

    def generate(self,hidden,len_seq=None,return_hidden = False, start = None):
        '''
            Take a hidden state and return a sentence of size len_seq or less
        '''
        i = 0
        cur_state = torch.tensor(Vocabulary.SOS) if start is None else torch.tensor(start)
        cur_state = cur_state.to(device)
        outputs = []
        
        while (len_seq is None or len_seq != i):
            cur_state = self.emb(cur_state)
            cur_state = cur_state.unsqueeze(0).unsqueeze(0)
            
            _, hn = self.gru(cur_state, hidden)
            hidden = hn
            output = self.classifieur(hn)[0]
            
            if self.random:
                with torch.no_grad():
                  output_normalized = torch.nn.functional.softmax(output[0],dim = 0)
                cur_state = torch.multinomial(output_normalized,1)[0] # to verify   
            else:
                cur_state = torch.argmax(output)
            
            outputs += [output[0]]
            if cur_state == Vocabulary.EOS:
                break

            i += 1
        if return_hidden:
            return torch.stack(outputs), hidden
        return torch.stack(outputs)

class State:
    def __init__(self,model,optim, selecter):
        self.model = model
        self.optim = optim
        self.selecter = selecter
        self.epoch, self.iteration = 0, 0
        
        
#_________________________________________________

#TO EXCECUTE FOR TRAINING


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100
embedding_space = 200
hidden_size = 164
ITERATIONS = 20


datatrain = TradDataset_2("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset_2("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=500, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

savepath = Path("/Users/sarahkerriche/Downloads/ML-IKSEM/model_trad_3_sarah.pch")



if savepath.is_file():
    print("file pris")
    with savepath.open("rb") as fp:
        state = torch.load(fp, map_location=torch.device('cpu') )
        print(datatrain.getSizes())
else:
    selecter = ModeSelecter()    
    entry_size, exit_size = datatrain.getSizes()
    model = torch.nn.Sequential(
        Encoder(entry_size,embedding_space,hidden_size),
        selecter,
        Decoder(exit_size,embedding_space,hidden_size,exit_size,random = False))
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = State(model,optim,selecter,)

loss = torch.nn.CrossEntropyLoss(ignore_index = Vocabulary.PAD)

test = 100
log_dir = "logs/fit3/"
writer = SummaryWriter(log_dir)
for epoch in range(state.epoch,ITERATIONS):
    print("epoch",epoch)
    for x, x_size, y, y_size in train_loader:
        state.optim.zero_grad()
        
    
        x = x.to(device)
        y = y.to(device)
        x.batch_sizes = x_size
        y.batch_sizes = y_size
        
        yhat, mode = state.model((x,y))
        yhat_l = yhat.flatten(0,1)
        y_l = y.flatten()
    
        # yhat = torch.nn.utils.rnn.pack_padded_sequence(yhat.data,yhat.batch_sizes,enforce_sorted=False)
        l = loss(yhat_l,y_l)
        # output, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(output)

        l.backward()
        state.optim.step()
        
        print("here i am")
        
        #geener tensorboard
        writer.add_scalar("Loss/Train",l,state.iteration)
        if mode == "contraint":
            writer.add_scalar("Loss/TrainContraint",l,state.iteration)
        else:
            writer.add_scalar("Loss/TrainNonContraint",l,state.iteration)
        
        if state.iteration % test == 0:
            # test phase
            with torch.no_grad():
                #non con,traint dans le test
                state.selecter.isTest(True)
                for x, x_size, y, y_size in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    x.batch_sizes = x_size
                    y.batch_sizes = y_size

                    yhat, _ = state.model((x,y))
                    
                    yhat_l = yhat.flatten(0,1)
                    y_l = y.flatten()
                    # yhat = torch.nn.utils.rnn.pack_padded_sequence(yhat.data,yhat.batch_sizes,enforce_sorted=False)
                    l = loss(yhat_l,y_l)
                    yhat_l = torch.argmax(yhat_l,dim = 1)
                    # print(yhat_l.shape)
                    writer.add_scalar("Loss/Test",l,state.iteration)
                    pad = (y_l == Vocabulary.PAD)
                    writer.add_scalar("Accuracy/Test",torch.sum((yhat_l == y_l)*torch.logical_not(pad))/(yhat_l.shape[0]-torch.sum(pad)),state.iteration)
                    break
                state.selecter.isTest(False)#retrun mdoe entraineemnt 
        state.iteration+=1
        print("here i am2")

    
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state,fp)
#-----------------------------------------------END OF TRAING



#____________________________________________________________________________
#TO test our prediction on the already saved model



#Load frist load our pre-entrained model 
PATH="/Users/sarahkerriche/Downloads/ML-IKSEM/model_trad_3.pch"



dic = torch.load(PATH,map_location=torch.device('cpu'))

state = dic

#To predict with data test results type1
print("Final Results")
with torch.no_grad():
                state.selecter.isTest(True)
                for x, x_size, y, y_size in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    x.batch_sizes = x_size
                    y.batch_sizes = y_size
                    yhat, _ = state.model((x,torch.zeros(y.shape)))
                    
                    # yhat_l = yhat.flatten(0,1)
                    # y_l = y.flatten()
                    # yhat = torch.nn.utils.rnn.pack_padded_sequence(yhat.data,yhat.batch_sizes,enforce_sorted=False)
                    yhat = torch.nn.functional.softmax(yhat)
                    yhat = torch.argmax(yhat,dim = 2)

                    # print(yhat_l.shape)
                    
                    for i in range(10):
                        print("original:",datatest.s.decode(x[:,i].to(torch.int).tolist()))
                        print("translated:",datatest.s.decode(yhat[:,i].to(torch.int).tolist()))
                        print("wanted:",datatest.s.decode(y[:,i].to(torch.int).tolist())) 
                
                    
                    """ # version  Wordpiece 
                    for i in range(10):
                        print("original:",vocEng.getwords(x[:,i].to(torch.int).tolist()))
                        print("translated:",vocFra.getwords(yhat[:,i].to(torch.int).tolist()))
                        print("wanted:",vocFra.getwords(y[:,i].to(torch.int).tolist()))
                    break
                state.selecter.isTest(False)"""

#To predict with data test results type 2
 

l = [module for module in state.model.modules() if not isinstance(module, nn.Sequential)]
enc, sel ,dec = l[0],l[3],l[4]
#res = generate_beam()
with torch.no_grad():
    state.selecter.isTest(True)
    for x, x_size, y, y_size in test_loader:
        x = x.to(device)
        y = y.to(device)
        x.batch_sizes = x_size
        y.batch_sizes = y_size
        for i in range(10):
            s = x[:,i]
            s.batch_sizes = [x_size[i]]
            
            # to exchange with previous line if first or second model
            res = generate_beam(s,enc,dec,vocFra.get,vocFra.getword,Vocabulary.EOS,5,nucleus = p_nucleus(dec,0.95))
            #res = generate_beam(s,enc,dec,datatest.s.piece_to_id,datatest.s.id_to_piece,Vocabulary.EOS,5,nucleus = p_nucleus(dec,0.95),spaccing="~") 
            
            print("x:"," ".join(vocEng.getwords(s)))
            # to exchange with previous line if first or second model
            #print("x: ",datatest.s.decode(s.tolist())) 

        break
    state.selecter.isTest(False)

