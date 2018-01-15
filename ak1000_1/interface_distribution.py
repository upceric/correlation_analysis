import os
ff1=open('empty_fill_interface','w')
ff2=open('fill_fill_interface','w')
ff3=open('total','w')
ff4=open('sat','w')
for i in range(0,44):
    filename="./ak1000_"+str(i)+"/ak1000_"+str(i)
    filename_s="./ak1000_"+str(i)+"/ak1000_"+str(i)+"_sat"
    j=0
    if os.path.isfile(filename_s):
        with open(filename_s,'r') as f:
            for line in f:
                ff4.write(line+'\n')
    if os.path.isfile(filename):
        with open(filename,'r') as f:
            for line in f:
                if j==0:
                    ff1.write(line)
                    j+=1
                elif j==1:
                    ff2.write(line)
                    j+=1
                elif j==3:
                    ff3.write(line+'\n')
                    j+=1
                else:
                    j+=1

ff1.close()
ff2.close()
ff3.close()
ff4.close()
