#!/usr/bin/python2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys


if len(sys.argv) != 4:
    print sys.argv
    print "Usage: ./multi_model_pr_curve.py <plot_title> <result1_file> <result2_file>"
    exit(1)

matplotlib.rcParams.update({'font.size': 22})

plt.title(sys.argv[1])
red_patch = mpatches.Patch(color='red', label=sys.argv[2])
blue_patch = mpatches.Patch(color='blue', label=sys.argv[3])
black_patch = mpatches.Patch(color='black', label='Baseline: Energy difference')

result_line = False
m1_p, m1_r, m2_p, m2_r= [], [], [], []

with open(sys.argv[2], 'r') as f:
    for line in f.readlines():
        if len(line.rstrip().split()) == 0:
            continue
        if line.rstrip().split()[0] == 'thresh':
            result_line = True
            continue

        if result_line == True:
            seg = line.rstrip().split()
            m1_p.append(float(seg[1]))
            m1_r.append(float(seg[2]))

result_line = False
with open(sys.argv[3], 'r') as f:
    for line in f.readlines():
        if len(line.rstrip().split()) == 0:
            continue
        if line.rstrip().split()[0] == 'thresh':
            result_line = True
            continue

        if result_line == True:
            seg = line.rstrip().split()
            m2_p.append(float(seg[1]) )
            m2_r.append(float(seg[2]) )

Baseline_P = [64.49, 60.08, 56.22,  55.46, 55.41]
Baseline_R = [32.44, 70.80, 94.8, 97.15, 97.34]

l1 = plt.plot(m1_r, m1_p, color='red', linewidth=4, label='U')
l2 = plt.plot(m2_r, m2_p, color='blue', linewidth=4, label='U + R')
l3 = plt.plot(Baseline_R, Baseline_P, color='black', linewidth=4, label='Baseline')

#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
plt.legend(handles=[red_patch, blue_patch, black_patch])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()
