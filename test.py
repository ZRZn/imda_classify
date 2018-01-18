# import jieba
#
# a = "this movie is not worth seeing . <sssss> has no merits what so ever . <sssss> does not make sense . <sssss> a total waste of time . <sssss> i don ´ t know why brannagh bothered to try and make this . <sssss> de niro is a joke as the monster . <sssss> read a horror story instead of this exploitation of viewers . <sssss> simply not worth our time ."
# result = jieba.cut(a)
#
# b = "今天天气不错啊，我都不想上班了。你想去吗？"
# result2 = jieba.cut(b)
# i = 0
# for t in result:
#     if i == 14:
#         print("t == ", t)
#         break
#     i += 1
# print(str(result2))

# import nltk
# from nltk.tokenize import WordPunctTokenizer
# word_cut = WordPunctTokenizer()
# tokenizer = nltk.data.load('/Users/zrzn/Downloads/nltk_data/tokenizers/punkt/PY3/english.pickle')
#
#
# a = tokenizer.tokenize("My name is Tom. I am a boy. I like soccer!")
# b = word_cut.tokenize("i excepted a lot from this movie , and it did deliver . <sssss> there is some great buddhist wisdom in this movie . <sssss> the real dalai lama is a very interesting person , and i think there is a lot of wisdom in buddhism . <sssss> the music , of course , sounds like because it is by philip glass . <sssss> this adds to the beauty of the movie . <sssss> whereas other biographies of famous people tend to get very poor this movie always stays focused and gives a good and honest portrayal of the dalai lama . <sssss> all things being equal , it is a great movie , and i really enjoyed it . <sssss> it is not like taxi driver of course but as a biography of a famous person it is really a great film indeed .")
# a.extend(b)
# print(a)

from collections import Counter

a = [1, 4, 7, 8, 0, 3, 5, 2, 5, 2, 7, 3, 6, 2, 3, 5, 6, 7, 1, 4, 5, 6, 2, 17, 23]
b = Counter(a)
b = b.most_common(len(b))
print(b)
d = dict()
for t in b:
    if t[1] > 1:
        d[t[0]] = len(d)
print(d)