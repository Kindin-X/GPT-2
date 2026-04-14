import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


#处理数据################################

# 创建数据集类
class GPTDatasetv1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
# 创建数据加载器
def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    tokenizer=tiktoken.get_encoding("gpt2")
    dataset=GPTDatasetv1(txt,tokenizer,max_length,stride)
    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

#将文本转化为词元id
def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

#层归一化类
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        #可训练参数，用于调整归一化的效果
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        var = x.var(dim = -1,keepdim = True,unbiased = False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x + self.shift

#GELU激活函数类
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5*x*(1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x + 0.044715*torch.pow(x,3))
                                     ))

#前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]),
        )
    def forward(self,x):
        return self.layers(x)


#多头注意力的实现
class MutilHeadAttention(nn.Module):
    def __init__ (self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保输出维度可以被头数整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        # 每个头的维度
        self.head_dim = d_out // num_heads
        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 计算查询、键、值
        #从原来的[b,num_tokens,d_out]变成[b,num_tokens,num_heads,head_dim]，进行多头处理
        q = self.w_q(x).view(b,num_tokens,self.num_heads, self.head_dim)
        #对于一个输出维度，进行分割，让它每个token的输出维度变成头数个部分
        k = self.w_k(x).view(b,num_tokens,self.num_heads, self.head_dim)
        v = self.w_v(x).view(b,num_tokens,self.num_heads, self.head_dim)
        # 转置以便进行矩阵乘法，转置使一个token的输出维度变成self.head_dim，但是有num_heads种
        q = q.transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]
        k = k.transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]
        v = v.transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]

        # 计算分数
        attn_scores = q@k.transpose(2, 3)
        # 应用掩码
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / (k.shape[-1] ** 0.5), dim=-1)
        # 应用dropout
        attn_weights = self.attn_dropout(attn_weights)
        # 计算加权和
        context_vec = attn_weights @ v
        context_vec = context_vec.transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]
        # 转置并合并头
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        # 应用输出投影
        return context_vec
    
#transformer模块
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = MutilHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            dropout = cfg["drop_rate"],
            num_heads = cfg["n_heads"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
    def forward(self,x):
        #注意力残差连接
        shortcut = x
        #层归一化
        x = self.norm1(x)
        #多头注意力
        x = self.attn(x)
        #dropout
        x = self.dropout(x)
        #残差连接
        x = shortcut + x
    

        #前馈神经网络残差连接
        shortcut = x
        #层归一化
        x = self.norm2(x)
        #前馈神经网络
        x = self.ff(x)
        #dropout
        x = self.dropout(x)
        #残差连接
        x = shortcut + x
        return x
    



#GPT-2模型
class GPTModel(nn.Module):
    def __init__ (self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias = False)
    def forward(self,in_idx):
        batch_size,seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len,device = in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits


#用于生成
def generate_text(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        #获取当前上下文
        idx_cond = idx[:, -context_size:]
        #获取模型输出
        with torch.no_grad():
            logits = model(idx_cond)
        #获取所有样本的最后一个token的全部输出
        logits = logits[:, -1, :]
        #获取概率最大的token
        probas = torch.softmax(logits,dim=-1)
        idx_next = torch.argmax(probas,dim=-1,keepdim=True)
        #将新token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

#计算损失################################
#用于计算单个batch的损失
def calc_loss_batch(inputs,targets,model,divice):
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)
    #将logits和targets展平
    logits = logits.flatten(0,1)
    targets = targets.flatten()
    #计算损失
    loss = nn.functional.cross_entropy(logits,targets)#函数的优点，只需要传入logits和targets即可
    return loss

def clac_loss_dataloader(dataloader, model, device,num_batches=None):
    total_loss = 0.0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        else:
            loss = calc_loss_batch(inputs, targets, model, device)
            total_loss += loss.item()
    return total_loss / num_batches

#评估模型################################
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = clac_loss_dataloader(train_loader, model, device,num_batches=eval_iter)
        val_loss = clac_loss_dataloader(val_loader, model, device,num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_text(model,  tokenizer, device,start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    encoded = encoded.to(device)
    with torch.no_grad():
        token_ids = generate_text(model, encoded.unsqueeze(0), max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids[0], tokenizer)
    print(decoded_text.replace("\n"," "))  # 替换换行符以便更好地显示
    model.train()


#训练主函数
def train_model_simple(model, train_loader,val_loader,
                        optimizer, device, num_epochs,
                        eval_freq,eval_iter,start_context,tokenizer):
    train_losses,val_losses,track_tokens_seen = [],[],[]
    tokens_seen,global_step = 0,-1

    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, Step {global_step}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      )
            generate_and_print_text(
                model, tokenizer, device,start_context)
    return train_losses, val_losses, track_tokens_seen


#将id转化为文本
def token_ids_to_text(token_ids, tokenizer):
    # 如果是二维，遍历每一行
    if len(token_ids.shape) == 2:
        return [tokenizer.decode(sample.tolist()) for sample in token_ids]
    else:
        return tokenizer.decode(token_ids.tolist())
    
#可视化
def plot_losses(epochs_seen,tokens_seen,train_losses, val_losses):
    fig,ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs_seen, val_losses, linestyle='-.',label='Validation Loss', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, label='Train Loss vs Tokens Seen', color='blue', alpha=0.5)
    ax2.set_xlabel('Tokens Seen')
    fig.tight_layout()
    plt.show()




#创建字典
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate":0.1,
    "qkv_bias":False
}


    
#初始化token表
tokenizer = tiktoken.get_encoding("gpt2")


'''
#高级索引的规则是输入的三个张量，分别对应索引为一个索引组，提取一个元素,如果形状不匹配，就先进行广播
target_probas_2 = probas[text_idx,[0,2,1],targets[text_idx]]
print(target_probas_2)'''

#数据集准备

file_path="the-verdict.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

#创建训练集
train_ratio = 0.8
train_size = int(train_ratio*len(text))
train_text = text[:train_size]
#创建验证集
val_text = text[train_size:]
#设置随机种子
torch.manual_seed(123)
#创建数据加载器
train_loader = create_dataloader_v1(
    train_text,
    batch_size=4,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,drop_last=False,
    num_workers=0)
val_loader = create_dataloader_v1(
    val_text,
    batch_size=4,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,drop_last=False,
    num_workers=0)


#初始化模型
model = GPTModel(GPT_CONFIG_124M)
#初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
#将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 10
train_losses, val_losses, track_tokens_seen = train_model_simple(
    model, train_loader, val_loader,
    optimizer, device, num_epochs,
    eval_freq=5, eval_iter=5,
    start_context="The quick brown fox jumps over the lazy dog.",
    tokenizer=tokenizer)

#可视化损失
epochs_tensor = torch.linspace(1, num_epochs, len(train_losses))
tokens_tensor = torch.linspace(0, track_tokens_seen[-1], len(train_losses))
plot_losses(epochs_tensor,tokens_tensor,train_losses, val_losses)


