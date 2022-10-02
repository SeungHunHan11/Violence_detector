from wsgiref import headers
import timm
import torch
from torch.nn import functional as F
from torch import nn
import math
import sys

sys.path.append('./Violence_detector')

class base(nn.Module):
    def __init__(self, basemodel,timestep=30):
        super(base,self).__init__()
        self.basemodel=basemodel
        self.timestep=timestep
    def forward(self,x):
        output=[]
        for i in range(self.timestep):
            #input one frame at a time into the basemodel
            x_t = self.basemodel(x[:, i, :, :, :]) # Put each timestep image vector into vgg basemodel
            output.append(x_t)
        
        x = torch.stack(output, dim=0).transpose_(0, 1) #Stack each timestep output

        x_t=None
        output=None
        
        return x

class Pos_embedding(nn.Module):
    def __init__(self, dev , dim_emb=512,dropout=0.4,timestep=30):
        super(Pos_embedding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        self.dim_emb=dim_emb
        self.dev=dev
        self.timestep=timestep
    def forward(self, x):

        pe=torch.zeros(self.timestep,self.dim_emb).to(self.dev)

        for pos in range(self.timestep):
            for i in range(0,self.dim_emb,2):
                pe[pos,i] = math.sin(pos/(10000**((2*i)/self.dim_emb)))
                pe[pos,i+1] = math.cos(pos/(10000**((2*(i+1))/self.dim_emb)))

        x= x*math.sqrt(self.dim_emb)
        pe=pe.unsqueeze(0)
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
    def __init__(self,dev,timestep=40,model_name='vgg19_bn',dim_emb=512,
    mid_layer=1024,dropout=0.4,
    classes=1, encoder_layernum=6, 
    encoder_heads=8,activation='relu'):
        super(CNN_Vit,self).__init__()
        
        #self.Base=base(model=basemodel, num_class=dim_emb)
        #self.pos=Pos_embedding(dim_emb,dropout)
        #self.trans=Transformer(dim_emb,head=encoder_heads,layers=encoder_layernum,actv=activation)
        #self.flat=nn.Flatten()
        #self.linear1=nn.Linear(dim_emb,mid_layer)
        #self.drop=nn.Dropout(dropout)
        #self.relu=nn.ReLU()
        #self.linear2=nn.Linear(mid_layer,classes)
        self.baseModel  =  timm.create_model(model_name, pretrained=True,num_classes=dim_emb)

        i = 0
        for child in self.baseModel.features.children():
            
            if i < 40:
                for param in child.parameters():
                    param.requires_grad = False #Freeze Vgg19_bn layer up until layer 40
            else:
                for param in child.parameters():
                    param.requires_grad = True
            i +=1

        self.model=nn.Sequential(base(self.baseModel,timestep),
                            Pos_embedding(dev,dim_emb,dropout,timestep),
                            Transformer(dim_emb,head=encoder_heads,layers=encoder_layernum,actv=activation),
                            nn.Flatten(),
                            nn.Linear(timestep*dim_emb,mid_layer),
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
