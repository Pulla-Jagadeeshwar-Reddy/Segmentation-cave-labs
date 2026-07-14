arr = [40,10,20,30, 40]
l=[]
for i in arr :
    l.append(i)
print(l)
l = (sorted(l))
print(l)
l = set(l)
print(set(l))
l = list(l)
print(l)
for i in range(len(arr)) :
    for j in range(len(l)) :
        if l[j] == arr[i] :
            arr[i] = j+1
            break
print(arr)
