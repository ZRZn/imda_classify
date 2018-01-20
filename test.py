import numpy as np

# x = [[[1, 3, 2], [1], [2, 3]], [[1, 3, 2], [1]], [[1, 3, 2], [1], [2, 3], [1], [8, 9]], [[1, 3, 2], [1], [2, 3], [0]]]
# y = [[1, 0], [2, 0], [3, 0], [4, 0]]
#
# x = np.array(x)
# y = np.array(y)
#
# c = np.append(x, y)
# print(c)

# x = [29, [[78, 1], [1], [3], [4]], [[2]], [[5, 6, 7], [1], [2]], 3, 11, 37]
#
# def sortBySen(s):
#     max = 0
#     for sen in s:
#         if len(sen) > max:
#             max = len(sen)
#     return max
#
#
# y = sorted(x[1:4], key=lambda t: sortBySen(t))
# x[1:4] = y
# print(x)


x = [29, [[78, 1], [1], [3], [4]], [[2]], [[5, 6, 7], [1], [2]], 3, 11, [37, 23]]
print(x.pop())
print(x)
