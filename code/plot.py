# e.g. http://matplotlib.org/examples/api/barchart_demo.html

import matplotlib.pyplot as plt

lab_per = [0.005, 0.01, 0.02, 0.05, 0.10, 0.2, 0.5, 0.8]
test_acc = [0.785, 0.823, None, 0.880503144654, 0.867352773013, 0.883647798742, 0.889651229274, None]
neural_acc = [0.76, 0.82, None, 0.853344768439, 0.866209262436, 0.874213836478, None, None]
tsvm = [0.85277301315, 0.86649514008, 0.889651229274, 0.931675242996, 0.947398513436, 0.974271012007, 0.978845054317, 0.982275586049]

fig, ax = plt.subplots()

x = [1, 2, 3, 4, 5, 6, 7, 8]

ax.plot(x, test_acc, 'k--', label='baseline TSVM')
ax.plot(x, neural_acc, 'k:', label='neural network')
ax.plot(x, tsvm, 'k', label='Gaussian kernel TSVM')

#ax.set_xscale('log')
#ax.set_xlim([.005, .2])
plt.xticks(x, lab_per)
ax.legend(loc='lower right')
ax.set_title("Test Accuracy of Techniques")
ax.set_ylabel("Accuracy on Validation Set")
ax.set_xlabel("Fraction of Labeled Data")
#ax.set_xticklabels(lab_per)

plt.show()

