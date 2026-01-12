s={'g','e','k','s'}
s.add('f')
print('set after updating',s)

s.discard('g')
print('\nset after updating',s)

s.remove('e')
print('set after updating',s)

print('\npopped element',s.pop())
print('set after updating',s)

s.clear()
print('set after updating',s)