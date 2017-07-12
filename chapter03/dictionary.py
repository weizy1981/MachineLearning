# 字典
mydict = {'a': 6.18, 'b': 'str', 'c': True}
print('A value: %.2f' % mydict['a'])
mydict['a'] = 523
print('A value: %d' % mydict['a'])
print('keys: %s' % mydict.keys())
print('values: %s' % mydict.values())
for key in mydict:
    print(mydict[key])