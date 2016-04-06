# e.g. http://matplotlib.org/examples/api/barchart_demo.html

import matplotlib.pyplot as plt

lab_per = [0.005, 0.01, 0.05, 0.10, 0.2, 0.5]
test_acc = [0.785, 0.823, 0.880503144654, 0.867352773013, 0.883647798742, 0.889651229274]

fig, ax = plt.subplots()

ax.plot(lab_per, test_acc)
ax.set_xscale('log')
ax.set_xlim([.0025, 1])
ax.set_title("TSVM Validation")
ax.set_ylabel("Accuracy on Validation Set")
ax.set_xlabel("Fraction of Labeled Data")
#ax.set_xticklabels(lab_per)

plt.show()

