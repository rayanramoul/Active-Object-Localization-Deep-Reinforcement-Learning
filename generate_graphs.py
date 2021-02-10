import matplotlib.pyplot as plt
import numpy as np
r = open("logs_over_epochs","r")
d = r.readlines()
d = [i.strip() for i in d]

print(d[0])
print(d[1])
print(d[2])
print(d[3])
print(d[4])
print(d[5])
print(d[6])
print(d[7])
print(list(filter(None, d[1].split(' '))))


aps = {}
aps["0.1"] = []
aps["0.2"] = []
aps["0.3"] = []
aps["0.4"] = []
aps["0.5"] = []

recalls = {}
recalls["0.1"] = []
recalls["0.2"] = []
recalls["0.3"] = []
recalls["0.4"] = []
recalls["0.5"] = []


for i in range(1,130, 3):   #(17):
    a = list(filter(None, d[i+1].split(' ')))
    b = list(filter(None, d[i+2].split(' ')))
    c = list(filter(None, d[i+3].split(' ')))
    print("a : "+str(a))
    print("b : "+str(b))
    print("c : "+str(c))
    aps["0.1"].append(float(a[1]))
    aps["0.2"].append(float(a[2]))
    aps["0.3"].append(float(a[3]))
    aps["0.4"].append(float(a[4]))
    aps["0.5"].append(float(a[5]))

    recalls["0.1"].append(float(b[1]))
    recalls["0.2"].append(float(b[2]))
    recalls["0.3"].append(float(b[3]))
    recalls["0.4"].append(float(b[4]))
    recalls["0.5"].append(float(b[5]))
    
ar = []


ar = np.array(ar)
print(len(plt.plot(ar)))
labels = ["Threshold 0.1","Threshold 0.2","Threshold 0.3","Threshold 0.4", "Threshold 0.5"]
colors=['r','g','b','brown','purple']
c = 0
for key in aps:
    plt.plot(aps[key], label=labels[c])
    c += 1
plt.xlabel('Episode')
plt.ylabel('AP')
plt.legend()
plt.show()