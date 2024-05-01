i= int(input())
m=0
for a in range(len(str(i))):
    m+=(i%10)*2**a
    i=i//10
print(m)
