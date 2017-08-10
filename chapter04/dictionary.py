# 字典
mydict = {'a': 6.18, 'b': 'str', 'c': True}
print('A value: %.2f' % mydict['a'])
# 增加字典元素
mydict['a'] = 523
print('A value: %d' % mydict['a'])
print('keys: %s' % mydict.keys())
print('values: %s' % mydict.values())
for key in mydict:
    print(mydict[key])


mydict = {'a': 6.18, 'b': 'str', 'c': True}
# 删除特定元素
mydict.pop('a')
print(mydict)
# 删除字典的全部元素
mydict.clear()
print(mydict)



