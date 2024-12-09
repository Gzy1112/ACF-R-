import argparse
import pickle
import torchtext
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import math
from transformers import AutoTokenizer,BertForMaskedLM, BertForSequenceClassification,BertModel,BertTokenizer
from transformers import DistilBertTokenizer,DistilBertModel,AutoModel
import random
from random import choice

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImage(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size  # 1024
        self.no_imgnorm = no_imgnorm  # false

        self.cnn = self.get_cnn(cnn_type, True)  # resnet152
        print('finetune:', finetune)  # false

        for param in self.cnn.parameters():
            param.requires_grad = finetune

        if cnn_type.startswith('vgg'):  # false
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            # print(self.cnn.module.fc.in_features)
            self.fc = nn.Linear(
                self.cnn.module.fc.in_features, embed_size)  # 2048,1024
            self.cnn.module.fc = nn.Sequential()
        self.fc_attn_i = nn.Linear(1024, 1024)
        self.fusion = Fusion()
        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
        else:
            model = nn.DataParallel(model)

        if torch.cuda.is_available():
            model.cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImage, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        r = np.sqrt(6.) / np.sqrt(self.fc_attn_i.in_features +
                                  self.fc_attn_i.out_features)
        self.fc_attn_i.weight.data.uniform_(-r, r)
        self.fc_attn_i.bias.data.fill_(0)

    def forward(self, images, local_image):
        """Extract image feature vectors."""
        with torch.no_grad():
            features = self.cnn(images)
            features = l2norm(features)
        # linear projection to the joint embedding space
        features = self.fc(features)
        # normalization in the joint embedding space
        features_1 = self.fc_attn_i(local_image)

        # features = l2norm(local_imgae)
        features = l2norm(self.fusion(features, features_1))
        return features


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.f_size = 1024
        self.gate0 = nn.Linear(self.f_size*2, self.f_size)
#         self.gate1 = nn.Linear(self.f_size, self.f_size)

        self.fusion0 = nn.Linear(self.f_size, self.f_size)
        self.fusion1 = nn.Linear(self.f_size, self.f_size)

    def forward(self, vec1, vec2):
        vec = torch.cat((vec1, vec2), dim=1)
        features_1 = self.gate0(vec)
#         features_2 = self.gate1(vec2)
        t = torch.sigmoid(features_1)
        f = t * vec1 + (1 - t) * vec2
        return f


class EncoderRegion(nn.Module):
    def __init__(self, opt):
        super(EncoderRegion, self).__init__()
        self.fc_region = nn.Linear(2048, opt.embed_size)  # 2048,1024
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_region.in_features +
                                  self.fc_region.out_features)
        self.fc_region.weight.data.uniform_(-r, r)
        self.fc_region.bias.data.fill_(0)

    def forward(self, region_feat):  # [128,36,2048]
        region_feat = self.fc_region(region_feat)  # [128,36,1024]
        region_feat = l2norm(region_feat, dim=-1)  # [128,36,1024]
        return region_feat  # [128,36,1024]

class EncoderWord_bert_tiny(nn.Module):
    def __init__(self,opt):
        super(EncoderWord_bert_tiny,self).__init__()
        self.embed_size = opt.embed_size
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny").cuda().eval()#BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")#AutoTokenizer.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(128,self.embed_size).cuda()

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self,x,length):# (128 sentence string)
        #print(max(length))
        text = self.tokenizer(x, padding='longest', max_length=60, return_tensors="pt")#.cuda()
        length = text["attention_mask"].sum(1).numpy().tolist()
        text = {item:text[item].cuda() for item in text}
        with torch.no_grad():
            bert_output = self.bert(**text,output_hidden_states=True)
        out = self.fc(bert_output.hidden_states[-1]) #.cuda()
        #out = l2norm(out, dim=-1)
        return out,length

def shuffleTextEmb(out):
    #out_neg = out.cpu().detach().numpy()
    out_neg = out.clone()
    out_neg = out_neg.cpu().detach().numpy()
    time_index_list = [n for n in range(out_neg.shape[1])]
    np.random.shuffle(time_index_list)#随机打乱
    save_shuff_x = []
    for i in time_index_list:
        shuff_x = out_neg[:,i,:] #.cpu().detach().numpy()#128,1024按照随机打乱的顺序组合成新的new_x
        save_shuff_x.append(np.expand_dims(shuff_x,axis=1))
    out_neg = np.concatenate(save_shuff_x,axis=1)
    out_neg = torch.from_numpy(out_neg).cuda()
    return out_neg

def resetZeroTextEmb(out):
    random_noise1 = random.randint(0, out.shape[2] - 1)
    out_neg41 = out.clone()
    out_neg41[ : , : , random_noise1:random_noise1+1] = 0
    #random_noise1 = random.randint(0, out.shape[1] - 1)
    random_noise2 = random.randint(0, out.shape[1] - 1)
    #random_noise3 = random.randint(0, out.shape[1] - 1)
    out_neg42 = out.clone()
    #out_neg42[: , random_noise1:random_noise1+1 , :] = 0
    out_neg42[: , random_noise2:random_noise2+1 , :] = 0
    #out_neg42[: , random_noise3:random_noise3+1 , :] = 0
    out_neg = choice([out_neg41,out_neg42])
    #out_neg = out_neg42
    return out_neg

class EncoderWord_bert(nn.Module):
    def __init__(self,opt):
        super(EncoderWord_bert,self).__init__()
        self.embed_size = opt.embed_size
        self.bert = BertModel.from_pretrained("bert-base-uncased").cuda().eval()#BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")#AutoTokenizer.from_pretrained("bert-base-uncased")
        #self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
        #self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768,self.embed_size).cuda()
        self.dropout = nn.Dropout(p=0.5)
        self.shuffle = shuffleTextEmb
        self.resetZero = resetZeroTextEmb
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self,x,length):# (128 sentence string)
        #print(max(length))
        text = self.tokenizer(x, padding='longest', max_length=60, return_tensors="pt")#.cuda()
        text = {item:text[item].cuda() for item in text}
        with torch.no_grad():
            bert_output = self.bert(**text,output_hidden_states=True)
        out = self.fc(bert_output.hidden_states[-1]).cuda() #128,28,1024
        
        # 1. 
        out_noise = torch.randn(out.shape[0], out.shape[1], out.shape[2]).cuda() #128,28,1024
        out_neg1 = out.add(out_noise)

        # 2. out dropout 直接对out进行dropout 
        out_neg2 = self.dropout(out)
        
        # 3. out 在28维度打乱s
        out_neg3 = self.shuffle(out)

        # 4. out 在第一维和第二维随机置0
        out_neg4 = self.resetZero(out)

        out_neg = choice([out_neg1, out_neg2, out_neg3, out_neg4])
        #out_neg = out_neg4
        out = torch.cat((out, out_neg), 0)
        return out,text["attention_mask"].sum(1).cpu().numpy().tolist() + text["attention_mask"].sum(1).cpu().numpy().tolist()

class EncoderWord_bert_distil(nn.Module):
    def __init__(self,opt):
        super(EncoderWord_bert_distil,self).__init__()
        self.embed_size = opt.embed_size
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased").cuda().eval()#BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")#AutoTokenizer.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768,self.embed_size).cuda()

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self,x,length):# (128 sentence string)
        #print(max(length))
        text = self.tokenizer(x, padding='longest', max_length=60, return_tensors="pt")#.cuda()
        text = {item:text[item].cuda() for item in text}
        with torch.no_grad():
            bert_output = self.bert(**text,output_hidden_states=True)
        out = self.fc(bert_output.hidden_states[-1]) #.cuda()
        #out = l2norm(out, dim=-1)
        return out,text["attention_mask"].sum(1).cpu().numpy().tolist()

class EncoderWord(nn.Module):

    def __init__(self, opt):
        super(EncoderWord, self).__init__()
        self.embed_size = opt.embed_size  # 1024
        # word embedding
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)  # 11755,300
        # caption embedding
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size,
                          opt.num_layers, batch_first=True)
        vocab = pickle.load(open('vocab/'+opt.data_name+'_vocab.pkl', 'rb'))
        word2idx = vocab.word2idx
        # self.init_weights()
        self.init_weights('glove', word2idx, opt.word_dim)  # 11685 found
        self.dropout = nn.Dropout(0.1)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()  # download glove
            else:
                raise Exception(
                    'Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace(
                        '-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):  # ([196, 42]),[196]
        # return out
        x = self.embed(x)  # emb:[11755, 300] ([196, 42, 300])
        x = self.dropout(x)  # ([196, 42, 300])

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded  # [196, 42, 1024] [196]

        cap_emb = l2norm(cap_emb, dim=-1)
        cap_emb_mean = torch.mean(cap_emb, 1)  # PADDING
        cap_emb_mean = l2norm(cap_emb_mean)  # 196,1024

        return cap_emb, cap_emb_mean


class EncoderText(nn.Module):
    def __init__(self, opt):
        super(EncoderText, self).__init__()
        # self.sa = TextSA(opt.embed_size, 0.4) fix by wang
        self.fc_text = nn.Linear(1024, 1024)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_text.in_features +
                                  self.fc_text.out_features)
        self.fc_text.weight.data.uniform_(-r, r)
        self.fc_text.bias.data.fill_(0)

    def forward(self, word_emb):
        # word_emb_mean = torch.mean(word_emb, 1)
        # cap_emb = self.sa(word_emb, word_emb_mean)
        word_emb = l2norm(self.fc_text(word_emb))
        return word_emb


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def func_attention(query, context, opt, smooth, eps=1e-8):
    q = query #[128, 36, 1024]
    k = context #[128, 11, 1024]
    v = context #[128, 11, 1024]
    k = k.view(k.shape[0], k.shape[1], 8, k.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(q.shape[0], q.shape[1], 8, q.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(v.shape[0], v.shape[1], 8, v.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    
    att = (q @ k.transpose(-2, -1)) * smooth #(1.0 / math.sqrt(k.size(-1))) 128,8,36,11
    #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)#(B, nh, T, T)
    #att = self.attn_dropout(att)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) 128,8,36,128
     # re-assemble all head outputs side by side

    #y = torch.nn.functional.scaled_dot_product_attention(
    #    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False) #([196, 8, 36, 128])
    y = y.transpose(1, 2).contiguous().view(
        query.shape[0], query.shape[1], query.shape[2]) #128,36,1024
    return y #([128,36, 1024])

# 196,36,1024; [196,14,1024] ; smooth 4
def func_attention_old(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)  # 196,1024,36

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)  # 196,14,36

    attn = nn.LeakyReLU(0.1)(attn)  # 196,14,36
    attn = l2norm(attn, 2)  # 196,14,36

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()  # 196,36,14
    # --> (batch*queryL, sourceL)
    attn = F.softmax(attn * smooth, dim=2)  # 196,36,14

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()  # 196,14,36
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)  # [196, 1024, 14])
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)  # ([196, 1024, 36])
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(
        weightedContext, 1, 2)  # ([196, 36, 1024])

    return weightedContext


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    weiContext_i = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1) 
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext = func_attention(cap_i_expand, images, opt, smooth=9.)
        cap_i_expand = cap_i_expand.contiguous() 
        #weiContext = weiContext.contiguous() fix by wang for faster
        # (n_image, n_word) 
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)

        row_sim = row_sim.mean(dim=1, keepdim=True)

        similarities.append(row_sim)

        weiContext = weiContext.mean(dim=1, keepdim=True)

        weiContext_i.append(weiContext)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    weiContext_i = torch.cat(weiContext_i, 1)
    weiContext_i = [weiContext_i[i, i, :].view(
        1, 1024) for i in range(n_image)]
    weiContext_i = torch.cat(weiContext_i, 0)

    return similarities, weiContext_i

# [128, 36, 1024] ([128, 27, 1024]) [128]
def xattn_score_i2t_val(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    weiContext_t = []
    n_image = images.size(0)  # 128
    n_caption = captions.size(0)  # 128

    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]  # 15
        cap_i = captions[i, :n_word, :].unsqueeze(
            0).contiguous()  # 1, 15, 1024
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)  # 128,15,1024
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext = func_attention(
            images, cap_i_expand, opt, smooth=4.)  # [196, 36, 1024]) text embedding
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)  # ([196, 36])
        row_sim = row_sim.mean(dim=1, keepdim=True)  # 196,1 #
        similarities.append(row_sim)  # one sentence
        weiContext = weiContext.mean(dim=1, keepdim=True)
        weiContext_t.append(weiContext)  # 196,1,1024 full image

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)  # 196,196
    weiContext_t = torch.cat(weiContext_t, 1)  # 196,196,1024 ???
    weiContext_t = [weiContext_t[i, i, :].view(
        1, 1024) for i in range(n_image)]
    weiContext_t = torch.cat(weiContext_t, 0)  # 196,1024
    return similarities, weiContext_t

# [196, 36, 1024] ([256, 42, 1024]) [256]
def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    weiContext_t = []
    n_image = images.size(0)  # 128
    n_caption = captions.size(0)  # 256

    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]  # 11
        cap_i = captions[i, :n_word, :].unsqueeze(
            0).contiguous()  # 1, 11, 1024
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)  # 128,11,1024
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext = func_attention(
            images, cap_i_expand, opt, smooth=4.)  # [1, 36, 1024]) text embedding
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)  # ([128, 36])
        row_sim = row_sim.mean(dim=1, keepdim=True)  # 196,1 #
        similarities.append(row_sim)  # one sentence
        weiContext = weiContext.mean(dim=1, keepdim=True)
        weiContext_t.append(weiContext)  # 196,1,1024 full image

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)  # 128,256
    weiContext_t = torch.cat(weiContext_t, 1)  # 128,256,1024
    weiContext_t1 = [weiContext_t[i, i, :].view(
        1, 1024) for i in range(n_image)] #[128]
    weiContext_t2 = [weiContext_t[i,i+ n_image,:].view(1,1024) for i in range(n_image)]
    weiContext_t1 = torch.cat(weiContext_t1, 0)  # 128,1024
    weiContext_t2 = torch.cat(weiContext_t2, 0)  # 128,1024
    weiContext_t = torch.cat([weiContext_t1, weiContext_t2], 0) #256,1024
    return similarities, weiContext_t # 128,256; 256,1024


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

class Loss_intra(nn.Module):
    def __init__(self, opt):
        super(Loss_intra, self).__init__()
        self.temp = 0.05
        self.opt = opt
        self.sim = cosine_sim
        self.criterion = nn.CrossEntropyLoss().cuda()
    def forward(self, cap_emb):
        print(cap_emb.shape)
        return 0.01

class NTxent_Loss_v2_old(nn.Module):
    def __init__(self,opt):
        super(NTxent_Loss_v2_old,self).__init__()
        #LABELS = torch.cat([torch.arange(opt.batch_size) for i in range(2)], dim=0)
        #self.LABELS = torch.eye(opt.batch_size)#(LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() #one-hot representations
        #self.LABELS = self.LABELS.cuda()
        self.temp = 0.05
        self.opt = opt
        self.sim = cosine_sim
        self.criterion = nn.CrossEntropyLoss().cuda()


    def forward(self,im, s,sims_local):
        """
        NT-Xent Loss.
    
        Args:
            z1: The learned representations from first branch of projection head
            z2: The learned representations from second branch of projection head 
        Returns:
            Loss
        """
        scores_global = self.sim(im, s) #128,128
        scores_local = sims_local #128,128
        similarity_matrix_i2t = self.opt.ratio * scores_local + (1 - self.opt.ratio) * scores_global
        mask = torch.eye(im.shape[0], dtype=torch.bool).cuda()
        positives_i2t = similarity_matrix_i2t[mask.bool()].view(mask.shape[0], -1)
        negatives_i2t = similarity_matrix_i2t[~mask.bool()].view(mask.shape[0], -1)
        logits_i2t = torch.cat([positives_i2t, negatives_i2t], dim=1)
        labels_i2t = torch.zeros(logits_i2t.shape[0], dtype=torch.long).cuda()
        logits_i2t = logits_i2t / self.temp
        loss_i2t = self.criterion(logits_i2t, labels_i2t)

        similarity_matrix_t2i = similarity_matrix_i2t.t()
        positives_t2i = similarity_matrix_t2i[mask.bool()].view(mask.shape[0], -1)
        negatives_t2i = similarity_matrix_t2i[~mask.bool()].view(mask.shape[0], -1)
        logits_t2i = torch.cat([positives_t2i, negatives_t2i], dim=1)
        labels_t2i = torch.zeros(logits_t2i.shape[0], dtype=torch.long).cuda()
        logits_t2i = logits_t2i / self.temp
        loss_t2i = self.criterion(logits_t2i, labels_t2i)

        loss = loss_i2t+loss_t2i
        return loss
    
class NTxent_Loss_v2(nn.Module):
    def __init__(self, opt):
        super(NTxent_Loss_v2, self).__init__()
        #LABELS = torch.cat([torch.arange(opt.batch_size) for i in range(2)], dim=0)
        # self.LABELS = torch.eye(opt.batch_size)#(LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() #one-hot representations
        #self.LABELS = self.LABELS.cuda()
        self.temp = 0.1
        self.opt = opt
        self.sim = cosine_sim
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, im, s, sims_local): #128,1024;256,1024;128,256
        """
        NT-Xent Loss.

        Args:
            z1: The learned representations from first branch of projection head
            z2: The learned representations from second branch of projection head 
        Returns:
            Loss
        """
        scores_global = self.sim(im, s)  # 128,256
        scores_local = sims_local  # 128,256
        similarity_matrix_i2t = self.opt.ratio * \
            scores_local + (1 - self.opt.ratio) * scores_global #128,256

        for i in range(len(similarity_matrix_i2t)):
            for j in range(len(similarity_matrix_i2t[0])//2, len(similarity_matrix_i2t[0])):
                if similarity_matrix_i2t[i][j] > similarity_matrix_i2t[i][i]:
                    similarity_matrix_i2t[i][j] = 0

        ## new add
        mask_pos = torch.eye(im.shape[0], dtype=torch.bool).cuda() #128,128(对角线为true，其他地方为false)
        mask_neg = torch.zeros([im.shape[0],im.shape[0]],dtype=torch.bool).cuda() #128,128（全部为false）
        mask = torch.cat([mask_pos,mask_neg],1) #128,256的矩阵，128*128的对角线为true，其他所有部分都为false
        positives_i2t = similarity_matrix_i2t[mask.bool()].view(
            mask.shape[0], -1) #128,1
        negatives_i2t = similarity_matrix_i2t[~mask.bool()].view(
            mask.shape[0], -1) #128,255
        logits_i2t = torch.cat([positives_i2t, negatives_i2t], dim=1) #128,256
        labels_i2t = torch.zeros(logits_i2t.shape[0], dtype=torch.long).cuda() #128
        logits_i2t = logits_i2t / self.temp #128,256
        loss_i2t = self.criterion(logits_i2t, labels_i2t)

        similarity_matrix_t2i = similarity_matrix_i2t.t() #256,128
        similarity_matrix_t2i = similarity_matrix_t2i[:similarity_matrix_t2i.shape[0] // 2,:] #128,128
        positives_t2i = similarity_matrix_t2i[mask_pos.bool()].view(
            mask.shape[0], -1) #128,1
        negatives_t2i = similarity_matrix_t2i[~mask_pos.bool()].view(
            mask.shape[0], -1) #128,127
        logits_t2i = torch.cat([positives_t2i, negatives_t2i], dim=1) #128,128
        labels_t2i = torch.zeros(logits_t2i.shape[0], dtype=torch.long).cuda() #[128]
        logits_t2i = logits_t2i / self.temp #128,128
        loss_t2i = self.criterion(logits_t2i, labels_t2i)

        loss = loss_i2t+loss_t2i
        return loss


class NTxent_Loss(nn.Module):
    def __init__(self, opt):
        super(NTxent_Loss, self).__init__()
        #LABELS = torch.cat([torch.arange(opt.batch_size) for i in range(2)], dim=0)
        # self.LABELS = torch.eye(opt.batch_size)#(LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() #one-hot representations
        #self.LABELS = self.LABELS.cuda()
        self.temp = 0.05
        self.opt = opt
        self.sim = cosine_sim
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, im, s, sims_local):
        """
        NT-Xent Loss.

        Args:
            z1: The learned representations from first branch of projection head
            z2: The learned representations from second branch of projection head 
        Returns:
            Loss
        """
        scores_global = self.sim(im, s)  # 128,128
        scores_local = sims_local  # 128,128
        similarity_matrix = self.opt.ratio * scores_local + \
            (1 - self.opt.ratio) * scores_global
        #labels = torch.eye(im.shape[0]).cuda()
        mask = torch.eye(im.shape[0], dtype=torch.bool).to(im.device)
        #labels = labels[~mask].view(labels.shape[0], -1)
        #similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[mask.bool()].view(mask.shape[0], -1)

        negatives = similarity_matrix[~mask.bool()].view(mask.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temp

        loss = self.criterion(logits, labels)
        return loss


class ContrastiveLoss(nn.Module):

    def __init__(self, opt, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        #         self.net_type = opt.type
        self.margin = margin
        #         self.margin = 0.2
        self.opt = opt
        self.sim = cosine_sim
        self.max_violation = max_violation

    # [128,1024] [128,1024] 128,36,1024;128,22,1024;128,[128,128]
    def forward(self, im, s, region_feats, word_feats, length, sims_local):

        scores_global = self.sim(im, s)  # [128,128]

        scores_local = sims_local  # 128,128

        scores = self.opt.ratio * scores_local + \
            (1 - self.opt.ratio) * scores_global  # 0.8,0.2
        diagonal = scores.diag().view(im.size(0), 1)  # [128,1]
        d1 = diagonal.expand_as(scores)  # 128,128
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class JZK(object):

    def __init__(self, opt, pre_scan=False):
        #         self.net_type = opt.type
        self.opt = opt
        self.grad_clip = opt.grad_clip  # 2.0
        self.img_enc = EncoderImage(opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    no_imgnorm=opt.no_imgnorm)
        #self.img_enc = torch.compile(self.img_enc)
        self.region_enc = EncoderRegion(opt)
        #self.region_enc = torch.compile(self.region_enc)
        self.cap_enc = EncoderText(opt)
        #self.cap_enc = torch.compile(self.cap_enc)
        self.word_enc = EncoderWord_bert(opt) #EncoderWord(opt)
        #self.word_enc = torch.compile(self.word_enc)
        # self.label_enc = EncoderLabel(opt)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.cap_enc.cuda()
            self.region_enc.cuda()
            self.word_enc.cuda()
            # self.label_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = NTxent_Loss_v2(opt)
        # ContrastiveLoss(opt, margin=opt.margin,
        #                measure=opt.measure,
        #                max_violation=opt.max_violation)

        #new add
        #self.criterion_intra = Loss_intra(opt)
        #new add

        params = list(self.img_enc.fc.parameters())
        
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
            params += list(self.word_enc.bert.parameters())
        params += list(self.word_enc.fc.parameters())
        params += list(self.region_enc.parameters())
        params += list(self.cap_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        # state_dict = [self.img_enc.state_dict(), self.cap_enc.state_dict(), self.label_enc.state_dict(),
        #               self.region_enc.state_dict(), self.word_enc.state_dict()]
        state_dict = [self.img_enc.state_dict(), self.cap_enc.state_dict(),
                      self.region_enc.state_dict(), self.word_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.cap_enc.load_state_dict(state_dict[1])
        # self.label_enc.load_state_dict(state_dict[2])
        self.region_enc.load_state_dict(state_dict[2])
        self.word_enc.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.cap_enc.train()
        # self.label_enc.train()
        self.region_enc.train()
        self.word_enc.fc.train() # fix for freeze bert

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.cap_enc.eval()
        # self.label_enc.eval()
        self.region_enc.eval()
        self.word_enc.eval()

    def forward_emb(self, images, region_feat,  captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images)  # [128, 3, 224, 224])
        #captions = Variable(captions)  # 128,35

        region_feat = Variable(region_feat)  # [128, 36, 2048])
        if torch.cuda.is_available():
            images = images.cuda()
            #captions = captions.cuda()
            region_feat = region_feat.cuda()

        # Forward

        region_emb = self.region_enc(region_feat)  # 128,36,1024
        word_emb, lengths = self.word_enc(captions, lengths)  # 256,35,1024 fix lengths for bert
        sims_local, attn_txt, attn_img = self.local_sim(
            region_emb, word_emb, lengths)  # 128,256 ; 256,1024; 128,1024
        img_emb = self.img_enc(images, attn_img)  # 128,1024
        cap_emb = self.cap_enc(attn_txt)  # 256,1024
        # img_label, cap_label, label = self.label_enc(img_emb, cap_emb, region_emb, region_cls, word_emb, lengths)
        return img_emb, cap_emb, region_emb, word_emb, lengths, sims_local

    def forward_loss(self, img_emb, cap_emb, region_emb, word_emb, lengths, sims_local, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        #if type == "inter":
        loss = self.criterion(
            img_emb, cap_emb,  sims_local)  # region_emb, word_emb, lengths,
        #else:
            # new add
            #loss = self.criterion_intra(cap_emb)
            # new add

        self.logger.update('Loss', loss.item(), img_emb.size(0))
        return loss

    # ([196, 3, 224, 224]),[196, 36, 2048],[196,42],[196]
    def train_emb(self, images, region_feat, captions, lengths, ids=None,captions_aug = None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # captions_all = captions + caption_neg #[256]
        captions_all = captions #[128]

        # compute the embeddings
        img_emb, cap_emb, region_emb, word_emb, lengths, sims_local= self.forward_emb(images, region_feat, captions_all,
                                                                                       lengths)

        #128,1024; 256,1024; 128,36,1024; 256,44,1024; 256; 128,256
        

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        loss = self.forward_loss(
            img_emb, cap_emb, region_emb, word_emb, lengths, sims_local)
    
        # compute gradient and do SGD step
        # loss.backward()

        # captions_all_aug = captions_aug + caption_neg #[256]
        captions_all_aug = captions_aug

        img_emb_aug, cap_emb_aug, region_emb_aug, word_emb_aug, lengths_aug, sims_local_aug = self.forward_emb(images, region_feat, captions_all_aug,
                                                                                       lengths)
        loss_aug = self.forward_loss(
            img_emb_aug, cap_emb_aug, region_emb_aug, word_emb_aug, lengths_aug, sims_local_aug)
        
        # loss_aug.backward()

        loss = 0.4* loss + 0.6* loss_aug  #loss = 0.8 * loss + 0.2 * loss_aug #inter-loss
        loss.backward()
        
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()

    # 128,36,1024; 256,28,1024;[256]
    def local_sim(self, region_emb, word_emb, length):
        attn_i = None
        attn_t = None
        scores = None
        if self.opt.cross_attn == 't2i':
            scores, attn_i = xattn_score_t2i(
                region_emb, word_emb, length, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores, attn_t = xattn_score_i2t(
                region_emb, word_emb, length, self.opt)
        elif self.opt.cross_attn == 'all':
            if region_emb.shape[0] == word_emb.shape[0]:
                score1, attn_t = xattn_score_i2t_val(
                region_emb, word_emb, length, self.opt)
            else:
                score1, attn_t = xattn_score_i2t(
                    region_emb, word_emb, length, self.opt)  # 128,256; 256,1024
            score2, attn_i = xattn_score_t2i(
                region_emb, word_emb, length, self.opt)  # 128,256 ; 128,1024
            scores = 0.5 * (score1 + score2)  # 128,256
        elif self.opt.cross_attn == "flash":
            scores,attn_t,attn_i = xattn_flash(region_emb, word_emb, length)
            
        return scores, attn_t, attn_i


def xattn_flash(image, caption, cap_lens, i2t=True):

    image_feat = image.mean(1).squeeze()
    cap_feat = caption.sum(1).squeeze()
    length = torch.from_numpy(np.array(cap_lens)).cuda()
    cap_feat = cap_feat/length.unsqueeze(1)
    score1 = cosine_similarity(image_feat, cap_feat, dim=1)

    
    q = image
    k = caption
    v = caption
    k = k.view(k.shape[0], k.shape[1], 8, k.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(q.shape[0], q.shape[1], 8, q.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(v.shape[0], v.shape[1], 8, v.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.1, is_causal=False)
    y_t = y.transpose(1, 2).contiguous().view(
        image.shape[0], image.shape[1], image.shape[2])


    q = caption
    k = image
    v = image
    k = k.view(k.shape[0], k.shape[1], 8, k.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(q.shape[0], q.shape[1], 8, q.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(v.shape[0], v.shape[1], 8, v.shape[2] //
                8).transpose(1, 2)  # (B, nh, T, hs)
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.1, is_causal=False)
    y_i = y.transpose(1, 2).contiguous().view(
        caption.shape[0], caption.shape[1], caption.shape[2])
    """
    

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)#(B, nh, T, T)
    #att = self.attn_dropout(att)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
     # re-assemble all head outputs side by side
    """
    return score1, y_t.mean(1).squeeze(),y_i.mean(1).squeeze()


def xattn_score_t2i1(images, captions, cap_lens, opt):

    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext = func_attention(cap_i_expand, images, opt, smooth=9.)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        row_sim = row_sim.mean(dim=1, keepdim=True)
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t1(images, captions, cap_lens, opt):

    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext = func_attention(images, cap_i_expand, opt, smooth=4.)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        row_sim = row_sim.mean(dim=1, keepdim=True)
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities
