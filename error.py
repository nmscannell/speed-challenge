file1 = 'files/train.txt'
file2 = 'test2.txt'

with open(file1) as f:
    lines1 = [line.rstrip() for line in f]

with open(file2) as f:
    lines2 = [line.rstrip() for line in f]

error = 0
for i in range(len(lines1)):
    error += (float(lines1[i])-float(lines2[i]))**2

error = error/len(lines1)
print('error: ' + str(error))
