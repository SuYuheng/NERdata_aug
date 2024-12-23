a = [1, 23, 42, 344]

with open("./stop_id.txt", 'r', encoding='utf-8') as f:
    ids = f.readlines()
    for element in ids:
        a.append(int(element))
print(a)
