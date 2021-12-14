# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 18:27:17 2021

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


#顯示中文
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False



#讀取資料
df = pd.read_excel("C://Users/user/OneDrive - 國立宜蘭大學/文件/ESSAY/essaydb/M2800_1.xlsx",engine='openpyxl')
print(df)
#print("測試")
#M1100

#df = pd.DataFrame(np.array([[15,160,48],[14,175,66],[15,153,50],[15,162,44]]))
#print(df)

"""
df = pd.read_excel("C://Users/user/Downloads/data.xlsx",'sheet2')
print(df)

frame = pd.DataFrame(np.random.random((4,4)),
                     index=['exp1','exp2','exp3','exp4'],
                     columns=['jan2015','Fab2015','Mar2015','Apr2005'])
print(frame)
frame.to_excel("C://Users/user/Downloads/data2.xlsx")


"""

"""
text = "I am happy today. I feel sad today."
from textblob import TextBlob
blob = TextBlob(text)
#print(blob)

#print(blob.sentences)
#print(blob.sentences[0].sentiment)
#(blob.sentences[1].sentiment)


text = u"我今天很快乐。我今天很愤怒。"

from snownlp import SnowNLP

s = SnowNLP(text)
for sentence in s.sentences:
 #print(sentence)

 #s1 = SnowNLP(s.sentences[0])
 #print(s1.sentiments)
 
 s2 = SnowNLP(s.sentences[1])
#print(s2.sentiments)

"""


text = df.comments.iloc[0]
from snownlp import SnowNLP
s3 = SnowNLP(text)

print(s3.sentiments)

def get_sentiment(text):
    s3 = SnowNLP(text)
    return s3.sentiments

df["sentiment"] = df.comments.apply(get_sentiment)
print(df.head())
#print(df.sentiment.mean())

frame = pd.DataFrame(df.sentiment)
                    
print(frame)
frame.to_excel("C://Users/user/OneDrive - 國立宜蘭大學/文件/ESSAY/essaydb/M2800_result.xlsx", index = False)



x = df.date
y = df.sentiment
plt.plot(x, y, label = 'sentiment')
#x軸旋轉45度
plt.xticks(rotation=45)
#設定網格
plt.grid(True)

plt.title('Sentiment index')
plt.xlabel('date')
plt.ylabel('sentiment')
plt.legend() #圖例
plt.show()

print("實驗結束")
#plt.xticks(rotation=45)

#-----------------------------


plt.savefig('C:/Users/user/OneDrive - 國立宜蘭大學/文件/ESSAY/essaydb/plt_result/M2800_result.png')
