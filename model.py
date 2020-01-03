
from torch import nn
from pytorch_pretrained_bert import BertForMaskedLM

class BertPunc(nn.Module):  
    
    def __init__(self, segment_size, output_size, dropout):
        super(BertPunc, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_vocab_size = 30522
        self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bert(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(self.dropout(self.bn(x)))
        return x
    
class BertQA(nn.Module):
    def __init__(self):
        super(BertQA, self).__init__()
        self.fc1 = nn.Linear(768, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.out = nn.Linear(200, 2)
        
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(1000)
        
        self.dropout1 = nn.Dropout(p = 0.2)
        self.dropout2 = nn.Dropout(p = 0.1)
        self.dropout3 = nn.Dropout(p = 0.05)

    def forward(self, x):
        fc1 = self.fc1(x)
        x1 = F.relu(fc1)
        fc2 = self.fc2(self.dropout1(x1))
        x2 = F.relu(fc2)
        fc3 = self.fc3(self.dropout2(x2))
        x3 = F.relu(fc3)
        out = self.out(self.dropout3(x3))
    
        return F.softmax(out, dim=1)