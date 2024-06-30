for num in range(100, 201):
    if sum(map(int, str(num))) % 2 == 0:
        print(num)