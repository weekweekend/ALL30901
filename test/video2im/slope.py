from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})
 
file_name = '14-0'
# Import
df_orig = pd.read_csv('csv/'+file_name+'_out_slope.csv', parse_dates=['time'], index_col='time')
s = df_orig.s.values
s2 = df_orig.s2.values
s5 = df_orig.s5.values

res = []
res2 = []
res5=[]

for i in range(0,len(s),3000):
  if(i+6000 <= len(s)):
    ans = 0
    ans2 = 0
    ans5 = 0
    for j in range(i,i+6000,1200):
      t = s[j:j+1200]
      ans += sum(val<0 for val in t)/1200/4*0.1*(-np.sum(t[t<0]))
      t2 = s2[j:j+1200]
      ans2 += sum(val<0 for val in t2)/1200/4*0.1*(-np.sum(t2[t2<0]))
      t5 = s5[j:j+1200]
      ans5 += sum(val<0 for val in t5)/1200/4*0.1*(-np.sum(t5[t5<0]))

    for j in range(i,i+6000,2400):
      t = s[j:j+2400]
      ans += sum(val<0 for val in t)/2400/2*0.3*(-np.sum(t[t<0]))
      t2 = s2[j:j+2400]
      ans2 += sum(val<0 for val in t2)/2400/2*0.3*(-np.sum(t2[t2<0]))
      t5 = s5[j:j+2400]
      ans5 += sum(val<0 for val in t5)/2400/2*0.3*(-np.sum(t5[t5<0]))

    t = s[i:i+6000]
    ans += sum(val<0 for val in t)/6000*0.6*(-np.sum(t[t<0]))
    t2 = s2[i:i+6000]
    ans2 += sum(val<0 for val in t2)/6000*0.6*(-np.sum(t2[t2<0]))
    t5 = s5[i:i+6000]
    ans5 += sum(val<0 for val in t5)/6000*0.6*(-np.sum(t5[t5<0]))

    res.append(ans)
    res2.append(ans2)
    res5.append(ans5)

print(res)
print(res2)
print(res5)

# fig, axes = plt.subplots(3,1, figsize=(10, 6), sharex=True, dpi=120)
# df_orig['s'].plot(ax=axes[0],  title='before')
# df_orig['s2'].plot(ax=axes[1], title='affter Smoothed 2%')
# df_orig['s5'].plot(ax=axes[2], title='affter Smoothed 5%')
# fig.suptitle('Slope before and after smoothing', y=0.95, fontsize=14)
# plt.show()
# plt.savefig('res/'+ file_name+"_slope.png")
