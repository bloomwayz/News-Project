import pandas as pd
from konlpy.tag import Okt
import numpy as np

okt = Okt()

c_df = pd.read_excel('news_right.xlsx')
c_titles = c_df['제목'].tolist()

l_df = pd.read_excel('news_left.xlsx')
l_titles = l_df['제목'].tolist()

stoptags = ['Josa', 'Eomi', 'KoreanParticle', 'Punctuation', 'Number', 'Foreign']
stopwords = ['전', '장연', '장애인', '지하철', '시위', '농성', '이다', '있다', '않다', '하다', '이', '그', '저', '때', '명', '은', '등', '들', '요']
c_res = {}
l_res = {}
all_res = {}

for title in c_titles:
    mt = okt.pos(title, stem=True)
    for morph, tag in mt:
        if morph not in stopwords and tag not in stoptags:
            c_res[morph] = c_res.get(morph, 0) + 1
            all_res[morph] = {1}

for title in l_titles:
    mt = okt.pos(title, stem=True)
    for morph, tag in mt:
        if morph not in stopwords and tag not in stoptags:
            l_res[morph] = l_res.get(morph, 0) + 1
            all_res[morph] = all_res.get(morph, set()) | {2}

c_tfidf = []
l_tfidf = []

for term, tf in c_res.items():
    df = len(all_res[term])
    idf = 2 / (1 + df)
    c_tfidf.append((term, tf*idf))

for term, tf in l_res.items():
    df = len(all_res[term])
    idf = 2 / (1 + df)
    l_tfidf.append((term, tf*idf))

c_tfidf.sort(key=lambda x: -x[1])
l_tfidf.sort(key=lambda x: -x[1])

fout1 = open('con_result.csv', 'w')
fout2 = open('lib_result.csv', 'w')

for term, tfidf in c_tfidf:
    fout1.write(f'{term}\t{tfidf}\n')

for term, tfidf in l_tfidf:
    fout2.write(f'{term}\t{tfidf}\n')

fout1.close()
fout2.close()