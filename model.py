from wsgiref import headers
import timm
import torch
from torch.nn import functional as F
from torch import nn
import math


class base(nn.Module):
    def __init__(self, model='vgg19_bn',num_class=512):
        super(base,self).__init__()
        self.baseModel  =  timm.create_model(model, pretrained=True,num_classes=num_class)

    def forward(self,x):
        batch_size, C,H, W = x.size()
        self.x=x.contiguous().view(batch_size,C,H,W)
        extracted=self.baseModel(self.x).contiguous().view(batch_size,-1)
        
        return extracted


class Pos_embedding(nn.Module):
    def __init__(self, dev , dim_emb=512,dropout=0.1):
        super(Pos_embedding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        self.dim_emb=dim_emb
        self.dev=dev
    def forward(self, x):
        batch_size=x.shape[0]
        pe=torch.zeros(batch_size,self.dim_emb).to(self.dev)

        for batch in range(batch_size):
            for i in range(0,self.dim_emb,2):
                pe[batch,i] = math.sin(batch/(10000**((2*i)/self.dim_emb)))
                pe[batch,i+1] = math.cos(batch/(10000**((2*(i+1))/self.dim_emb)))

        x= x*math.sqrt(self.dim_emb)

        x+= pe[:,:x.size(1)]
        x=self.dropout(x)
        

        return x

class Transformer(nn.Module):
    def __init__(self,dim_emb=512,head=8, layers=6,actv='gelu'):
        super(Transformer,self).__init__()
        self.encoder=nn.TransformerEncoderLayer(d_model=dim_emb,nhead=head,activation=actv)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder,num_layers=layers)

    def forward(self,x):
        x=self.transformer_encoder(x)

        return x
    
class CNN_Vit(nn.Module):
    def __init__(self,dev,basemodel='vgg19_bn',dim_emb=512,
    mid_layer=1024,dropout=0.1,
    classes=2, encoder_layernum=6, 
    encoder_heads=8,activation='gelu'):
        super(CNN_Vit,self).__init__()
        
        #self.Base=base(model=basemodel, num_class=dim_emb)
        #self.pos=Pos_embedding(dim_emb,dropout)
        #self.trans=Transformer(dim_emb,head=encoder_heads,layers=encoder_layernum,actv=activation)
        #self.flat=nn.Flatten()
        #self.linear1=nn.Linear(dim_emb,mid_layer)
        #self.drop=nn.Dropout(dropout)
        #self.relu=nn.ReLU()
        #self.linear2=nn.Linear(mid_layer,classes)

        self.model=nn.Sequential(base(model=basemodel, num_class=dim_emb),
                            Pos_embedding(dev,dim_emb,dropout),
                            Transformer(dim_emb,head=encoder_heads,layers=encoder_layernum,actv=activation),
                            nn.Flatten(),
                            nn.Linear(dim_emb,mid_layer),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                            nn.Linear(mid_layer,classes)
                            )
                        
    def forward(self,x):
        
        #x=self.Base(x)
        #x=self.pos(x)
        #x=self.trans(x)
        #x=self.flat(x)
        #x=self.linear1(x)
        #x=self.drop(x)
        #=self.relu(x)
        #x=self.linear2(x)
        x=self.model(x)
        return x
