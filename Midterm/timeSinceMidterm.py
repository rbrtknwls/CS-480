import matplotlib.pyplot as plt

xVals = ["CS343", "CS 486", "CS350", "CS341", "CS480", "Math239", "Math137"]
yVals = [9, 15, 14, 15, 21, 16, 14]

plt.grid(zorder=0)
barlist = plt.bar(xVals, yVals, zorder=3)
barlist[4].set_color('r')
plt.xlabel("Classes")
plt.ylabel("Days since Midterm")
plt.title("Days Vs Midterm Marking Time")
plt.show()