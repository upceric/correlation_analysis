import seaborn as sns
import matplotlib.pyplot as plt
x,y=[],[]
with open('result','r') as f:
    for line in f:
        a=line.strip().split()
        print(a[0],a[1][1:-1])
        x.append(float(a[0]))
        y.append(float(a[1][1:-1]))
plt.plot(x,y,'ro-')   
fs=16
ts=12     
#plt.axis('equal')
#plt.gcf().set_size_inches(6,6)
ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fs)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fs)
    tick.label1.set_fontweight('bold')
ax=plt.gca()
#ax.axis('off')
#plt.xlim([0,1])
plt.ylim([-0.1,0.1])
plt.xlabel(r'L[d_s]',fontsize=fs,fontweight='bold')
plt.ylabel(r'corr(L)',fontsize=fs,fontweight='bold')
m_filepath='/home/lww/Documents/BP/2017/BP-2017-12-28'
plt.savefig(m_filepath+'/1000_50p_1_4_bidi.png',dpi=200,bbox_inches='tight')
plt.savefig(m_filepath+'/1000_50p_1_4_bidi.svg',dpi=200,bbox_inches='tight')		
        
