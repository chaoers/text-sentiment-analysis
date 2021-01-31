# 项目依赖包见`requirements.txt`
# 根目录下使用`pip install -r requirements.txt`安装

# 下载数据，首次安装时运行
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

import pandas as pd

file_dir = './test_data.txt'

with open(file_dir, encoding='utf-8') as f:
    # 获取原始数据
    data = pd.read_table(f, header=None)

# 查看数据
# print(data ,data.dtypes)

# 提取评论数据
comm_data = data.iloc[:, [0, 6]]

print(comm_data)

# 调用textblob进行情绪分析
# https://textblob.readthedocs.io/

from textblob import TextBlob

# polarity项为文本积极性，是在[-1.0，1.0]范围内的浮点数
# subjectivity项为主观评分，是在[0.0，1.0]范围内的浮点数，其中0.0是非常客观的，而1.0是非常主观的
comm_data.insert(comm_data.shape[1], 'polarity', 0)
comm_data.insert(comm_data.shape[1], 'subjectivity', 0)

for i in range(0, data.shape[0]):
    blob = TextBlob(data[6][i])
    sentiment = blob.sentiment
    comm_data.loc[i, 'polarity'] = sentiment.polarity
    comm_data.loc[i, 'subjectivity'] = sentiment.subjectivity

print(comm_data)