import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re

data_dir='../data/custom/'

dir_path = data_dir+'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def filter_emoji(desstr, restr=''):
    # 过滤表情
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)

def get_category(data):
    temp = data['tag_sentiment_list'].map(lambda x: [j[1] for j in eval(x)])
    res = []
    for i in temp:
        res.extend(i)
    result = list(set(res))
    return result

data = pd.read_csv(data_dir+'data1119.csv')
data.columns = ['text', 'tag_sentiment_list']

# preprocess for emoji
data['text'] = data['text'].map(lambda x: filter_emoji(x, restr='xx'))
# 只保留review的长度小于600的
data = data[data['text'].str.len() <= 600]
data['idx'] = data.index
# train test split
x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)
cate_ls = get_category(data)

print ("category list: ",cate_ls)

with open(dir_path+"test_NLI_M.csv","w",encoding="utf-8") as g:
   for i in range(len(x_test)):
       category = []
       polarity = []
       id = str(x_test.iloc[i,:]['idx'])
       text = x_test.iloc[i,:]['text']
       label = eval(x_test.iloc[i,:]['tag_sentiment_list'])
       for j in range(len(label)):
            category.append(label[j][1])
            polarity.append(label[j][3])
       for cate in cate_ls:
           if cate in category:
               g.write(id + "\t" + polarity[category.index(cate)] + "\t" + cate + "\t" + text + "\n")
           else:
               g.write(id + "\t" + "none" + "\t" + cate + "\t" + text + "\n")

with open(dir_path+"train_NLI_M.csv","w",encoding="utf-8") as g:
   for i in range(len(x_train)):
       category = []
       polarity = []
       id = str(x_train.iloc[i,:]['idx'])
       text = x_train.iloc[i,:]['text']
       label = eval(x_train.iloc[i,:]['tag_sentiment_list'])
       for j in range(len(label)):
            category.append(label[j][1])
            polarity.append(label[j][3])
       for cate in cate_ls:
           if cate in category:
               g.write(id + "\t" + polarity[category.index(cate)] + "\t" + cate + "\t" + text + "\n")
           else:
               g.write(id + "\t" + "none" + "\t" + cate + "\t" + text + "\n")