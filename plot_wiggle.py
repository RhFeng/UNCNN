def plot_wiggle(axhdl, data, t, excursion):

    import numpy as np
    import matplotlib.pyplot as plt
    
    dt = 0.001
    
    nt = t[-1] / dt

    T = dt * np.linspace(0.0, nt, num = nt + 1)
    

    [ntrc, nsamp] = data.shape
    
    data_temp = np.zeros((ntrc, len(T)))
    
    for i in range(ntrc):
        temp = data[i,:]
        data_temp[i,:] = np.interp(T.ravel(), t.ravel(), temp.ravel())
        
    
    data = data_temp
    [ntrc, nsamp] = data.shape
    t = T
    
    t = np.reshape(t,len(t))
    
    t = np.hstack([0, t, t.max()])
    
    
    
    for i in range(0, ntrc):
        tbuf = excursion * data[i,:] / np.max(np.abs(data)) + i
#        tbuf = excursion * data[i,:] / 0.3079532062387342
        
        tbuf = np.hstack([i, tbuf, i])
            
        axhdl.plot(tbuf, t, color='black', linewidth=0.5)
        plt.fill_betweenx(t, tbuf, i, where=tbuf>i, facecolor=[1.0,0.0,0.0], linewidth=0) #[0.6,0.6,1.0]
        plt.fill_betweenx(t, tbuf, i, where=tbuf<i, facecolor=[0.0,0.0,1.0], linewidth=0) #[1.0,0.7,0.7]
    
    axhdl.set_xlim((-excursion, ntrc+excursion))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()