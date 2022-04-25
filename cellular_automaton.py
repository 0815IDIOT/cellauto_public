import numpy as np
import numpy.random as npr
import time
import copy
import numba
from numba import jit, objmode

@jit(nopython=True)
def getGamma(grid):
    nm = [len(grid), len(grid[0])]
    gammaMoor = [0.0, 0.0]
    cntTypMoor = [0.0, 0.0]
    gammaNeum = [0.0, 0.0]
    cntTypNeum = [0.0, 0.0]
    for i in range(nm[0]):
        for j in range(nm[1]):
            cntNachbarnMoor = 0.0
            cntHeteroMoor = 0.0
            cntNachbarnNeum = 0.0
            cntHeteroNeum = 0.0

            if i < nm[0] - 2:
                cntNachbarnMoor += 1
                cntNachbarnNeum += 1
                if grid[i][j] != grid[i+1][j]:
                    cntHeteroMoor += 1
                    cntHeteroNeum += 1
            if i > 1:
                cntNachbarnMoor += 1
                cntNachbarnNeum += 1
                if grid[i][j] != grid[i-1][j]:
                    cntHeteroMoor += 1
                    cntHeteroNeum += 1
            if j < nm[1] - 2:
                cntNachbarnMoor += 1
                cntNachbarnNeum += 1
                if grid[i][j] != grid[i][j+1]:
                    cntHeteroMoor += 1
                    cntHeteroNeum += 1
            if j > 1:
                cntNachbarnMoor += 1
                cntNachbarnNeum += 1
                if grid[i][j] != grid[i][j-1]:
                    cntHeteroMoor += 1
                    cntHeteroNeum += 1
            if i > 1 and j > 1:
                cntNachbarnMoor += 1
                if grid[i][j] != grid[i-1][j-1]:
                    cntHeteroMoor += 1
            if i < nm[0] - 2 and j > 1:
                cntNachbarnMoor += 1
                if grid[i][j] != grid[i+1][j-1]:
                    cntHeteroMoor += 1
            if i > 1 and j < nm[1] - 2:
                cntNachbarnMoor += 1
                if grid[i][j] != grid[i-1][j+1]:
                    cntHeteroMoor += 1
            if i < nm[0] - 2 and j < nm[1] - 2:
                cntNachbarnMoor += 1
                if grid[i][j] != grid[i+1][j+1]:
                    cntHeteroMoor += 1

            gammaNeum[grid[i][j]] = gammaNeum[grid[i][j]] + cntHeteroNeum / cntNachbarnNeum
            cntTypNeum[grid[i][j]] = cntTypNeum[grid[i][j]] + 1.0
            gammaMoor[grid[i][j]] = gammaMoor[grid[i][j]] + cntHeteroMoor / cntNachbarnMoor
            cntTypMoor[grid[i][j]] = cntTypMoor[grid[i][j]] + 1.0

    for i in range(len(gammaMoor)):
        gammaMoor[i] = gammaMoor[i] / cntTypMoor[i]
        gammaNeum[i] = gammaNeum[i] / cntTypNeum[i]

    return gammaMoor[0], gammaMoor[1], gammaNeum[0], gammaNeum[1]

@jit(nopython=True)
def cell_sorting_numba(grid, b11,b12,b22,tau,MCS,printEveryStep,dataEveryStep,threshold=0.9,ind=None, grid_hist_safe = True):
    nm = np.array(grid.shape)
    b = np.array([[b11,b12],[b12,b22]])
    n = nm[0]*nm[1]
    neighborhood = np.array([[0,1],[1,0],[0,-1],[-1,0]])
    neighborhood__ = np.array([[0,1],[1,0],[0,-1],[-1,0],[0,0]])

    N = len(neighborhood)
    MCS = MCS*nm[0]*nm[1]
    order = []
    ifl = []
    order_rel = []
    ifl_rel = []
    cond_hist = []
    grid_hist = []
    gammasMoor = []
    gammasNeum = []
    lastSteps = n

    gammaMoor0 = 0.0
    gammaMoor1 = 0.0
    gammaNeum0 = 0.0
    gammaNeum1 = 0.0

    for i in range(lastSteps):
        cond_hist.append(1.)

    # minimal binding strength
    dEmin = 8*np.amin(b)
    dEmin = 0
    t_real, tmp_t_real = [], 0
    # estimate maximal order parameter assuming same amount of a and b
    max_order = n*2 - 2*np.amin(nm)
    max_ifl = n / 2.0 * 4.0

    E_ = np.zeros(grid.shape)
    E = np.zeros((nm[0],nm[1],2))
    hetero = np.zeros((nm[0],nm[1],2),dtype=np.bool_)

    def init_bonds(grid, E_, E, hetero, neighborhood):
        # initialize bond energies
        for neigh in neighborhood:  # compute sum of bond energies to neighbors
            for i in range(nm[0]):
                for j in range(nm[1]):
                    E_[i,j] = E_[i,j] + b[grid[i,j],grid[(i+neigh[0])%nm[0],(j+neigh[1])%nm[1]]]

        # sum bond energies of neighboring cells
        for i in range(nm[0]):
            for j in range(nm[1]):
                # vertical
                E[i,j,0] = E_[i,j] + E_[(i+1)%nm[0],j]
                hetero[i,j,0] = np.invert(np.equal(grid[i,j], grid[(i+1)%nm[0],j]))
                # horizontal
                E[i,j,1] = E_[i,j] + E_[i,(j+1)%nm[1]]
                hetero[i,j,1] = np.invert(np.equal(grid[i,j], grid[i,(j+1)%nm[1]]))
        return E_, E, hetero

    E_, E, hetero = init_bonds(grid, E_, E, hetero, neighborhood)

    # FOR LARGE FIELDS
    tmp_hetero = hetero.flatten()
    tmp_hetero = np.arange(len(tmp_hetero))[tmp_hetero]
    rate = (np.exp(-E/tau)).flatten()[tmp_hetero]
    sum_rate = np.sum(rate)
    TST_tmp_ind = ind[tmp_hetero]
    rate = np.append(np.array([0]),rate)
    len_rate = len(rate)
    rate_ind = np.zeros((nm[0],nm[1],2),dtype=np.int64)
    for i, tmp_h in enumerate(tmp_hetero):
        rate_ind[ind[tmp_h,0],ind[tmp_h,1],ind[tmp_h,2]] = i+1
    free_ind = []
    len_free_ind = 0

    with objmode(start="double"):
        start = time.time()

    for time_MCS in range(MCS):
        # order parameter
        homob, heterob = 0.0, 0.0
        for i in range(nm[0]):
            for j in range(nm[1]):
                dhetero = np.abs(grid[i,j]-grid[(i+1)%nm[0],j])+np.abs(grid[i,j]-grid[i,(j+1)%nm[1]])
                heterob += (dhetero)
        homob = 1*(2*n-heterob)

        if (time_MCS % dataEveryStep) == 0:
            t_real.append(tmp_t_real)
            order.append(1.*homob)
            ifl.append(1.*heterob)
            if grid_hist_safe:
                grid_hist.append(np.copy(grid))
            order_rel.append(1.*homob/max_order)
            ifl_rel.append(1.*heterob/max_ifl)

            gammaMoor0, gammaMoor1, gammaNeum0, gammaNeum1 = getGamma(grid)
            gammasMoor.append([gammaMoor0,gammaMoor1])
            gammasNeum.append([gammaNeum0,gammaNeum1])

        cond_hist.append(1.*heterob / max_ifl)
        cond_hist.pop(0)
        cond = 0.0
        for ele in cond_hist:
            cond = cond + ele
        cond = cond / len(cond_hist)

        if 1. * cond < threshold:
            t_real.append(tmp_t_real)
            order.append(1.*homob)
            ifl.append(1.*heterob)
            order_rel.append(1.*homob/max_order)
            ifl_rel.append(1.*heterob/max_ifl)
            if grid_hist_safe:
                grid_hist.append(np.copy(grid))

            gammaMoor0, gammaMoor1, gammaNeum0, gammaNeum1 = getGamma(grid)
            gammasMoor.append([gammaMoor0,gammaMoor1])
            gammasNeum.append([gammaNeum0,gammaNeum1])

            with objmode():
                print("[+] Time needed: " + str(time.time() - start) + "s")

            break

        if (time_MCS % printEveryStep) == 0:
            with objmode():
                print("[" + str(time_MCS) + "] " + str(round(cond,3)) + " | ETA: " + str(round((MCS - time_MCS) * (time.time() - start) / (time_MCS+1),2)) + " s         ")

        # FOR LARGE FIELDS
        # draw waiting time
        rates = np.cumsum(rate[1:])
        sum_rate = rates[-1]
        # normalize rates
        rates = rates/sum_rate
        tmp_t_real += npr.exponential(scale=1./(np.exp(dEmin)*sum_rate))
        # draw bond
        random_variable = npr.uniform(0,1)
        switch_ind = 0
        while rates[switch_ind]<=random_variable:
            switch_ind += 1
        switch_ind = TST_tmp_ind[switch_ind]

        # switch cells and update bond energies
        o = switch_ind[:-1].copy() # grid index first cell
        t = switch_ind[:-1].copy()
        t[switch_ind[-1]] = t[switch_ind[-1]] + 1 # grid index second cell
        t = np.mod(t,nm)                          # periodic boundary

        # update bond energies
        for neigh in neighborhood:
            on, tn = np.mod(o+neigh,nm), np.mod(t+neigh,nm)
            on, tn = (on[0],on[1]), (tn[0],tn[1])
            # neighborhood first cell
            dEo = - b[grid[o[0],o[1]],grid[on]] + b[grid[t[0],t[1]],grid[on]]
            E_[on] = E_[on] + dEo
            E_[o[0],o[1]] = E_[o[0],o[1]] + dEo
            # neighborhood second cell
            dEt = - b[grid[t[0],t[1]],grid[tn]] + b[grid[o[0],o[1]],grid[tn]]
            E_[tn] = E_[tn] + dEt
            E_[t[0],t[1]] = E_[t[0],t[1]] + dEt

        E_[o[0],o[1]] = E_[o[0],o[1]] - b[grid[t[0],t[1]],grid[t[0],t[1]]] - b[grid[o[0],o[1]],grid[o[0],o[1]]] + 2*b[grid[t[0],t[1]],grid[o[0],o[1]]]
        E_[t[0],t[1]] = E_[t[0],t[1]] - b[grid[o[0],o[1]],grid[o[0],o[1]]] - b[grid[t[0],t[1]],grid[t[0],t[1]]] + 2*b[grid[t[0],t[1]],grid[o[0],o[1]]]

        # update sum of bond energies
        for neigh in neighborhood__:
            on, tn = np.mod(o+neigh,nm), np.mod(t+neigh,nm)
            on, tn = (on[0],on[1]), (tn[0],tn[1])
            # neighborhood first cell
            E[on][0] = E_[on] + E_[(on[0]+1) % nm[0],on[1]]
            E[(on[0]-1) % nm[0],on[1]][0] = E_[(on[0]-1) % nm[0],on[1]] + E_[on]
            E[on][1] = E_[on] + E_[on[0],(on[1]+1) % nm[1]]
            E[on[0],(on[1]-1) % nm[1]][1] = E_[on[0],(on[1]-1) % nm[1]] + E_[on]
            # neighborhood second cell
            E[tn][0] = E_[tn] + E_[(tn[0]+1) % nm[0],tn[1]]
            E[(tn[0]-1) % nm[0],tn[1]][0] = E_[(tn[0]-1) % nm[0],tn[1]] + E_[tn]
            E[tn][1] = E_[tn] + E_[tn[0],(tn[1]+1) % nm[1]]
            E[tn[0],(tn[1]-1) % nm[1]][1] = E_[tn[0],(tn[1]-1) % nm[1]] + E_[tn]

        # update grid
        grid[o[0],o[1]], grid[t[0],t[1]] = grid[t[0],t[1]], grid[o[0],o[1]]

        # FOR LARGE FIELDS
        bool_index = np.array([[o[0],o[1],0],
                               [(o[0]-1)%nm[0],o[1],0],
                               [o[0],o[1],1],
                               [o[0],(o[1]-1)%nm[1],1],
                               [t[0],t[1],0],
                               [(t[0]-1)%nm[0],t[1],0],
                               [t[0],t[1],1],
                               [t[0],(t[1]-1)%nm[1],1]])
        old_bool = np.zeros(8,dtype=np.bool_)
        for ct in range(8):
            old_bool[ct] = hetero[bool_index[ct,0],bool_index[ct,1],bool_index[ct,2]]

        # update heterotypic bonds
        # neighborhood first cell
        hetero[o[0],o[1]][0] = (grid[(o[0]+1)%nm[0],o[1]] != grid[o[0],o[1]])
        hetero[(o[0]-1)%nm[0],o[1]][0] = (grid[(o[0]-1)%nm[0],o[1]] != grid[o[0],o[1]])
        hetero[o[0],o[1]][1] = (grid[o[0],(o[1]+1)%nm[1]] != grid[o[0],o[1]])
        hetero[o[0],(o[1]-1)%nm[1]][1] = (grid[o[0],(o[1]-1)%nm[1]] != grid[o[0],o[1]])
        # neighborhood second cell
        hetero[t[0],t[1]][0] = (grid[(t[0]+1)%nm[0],t[1]] != grid[t[0],t[1]])
        hetero[(t[0]-1)%nm[0],t[1]][0] = (grid[(t[0]-1)%nm[0],t[1]] != grid[t[0],t[1]])
        hetero[t[0],t[1]][1] = (grid[t[0],(t[1]+1)%nm[1]] != grid[t[0],t[1]])
        hetero[t[0],(t[1]-1)%nm[1]][1] = (grid[t[0],(t[1]-1)%nm[1]] != grid[t[0],t[1]])

        # FOR LARGE FIELDS
        # update type of bond
        for ct in range(8):
            tuple_bool_index = (bool_index[ct,0],bool_index[ct,1],bool_index[ct,2])
            if old_bool[ct] and not hetero[tuple_bool_index]:    # switch hetero -> homo
                rate[rate_ind[tuple_bool_index]] = 0
                free_ind.append(rate_ind[tuple_bool_index])
                len_free_ind += 1
                TST_tmp_ind[rate_ind[tuple_bool_index]-1] = np.array([-1,-1,-1]) # just for safety
                rate_ind[tuple_bool_index] = 0
            elif not old_bool[ct] and hetero[tuple_bool_index]: # switch homo -> hetero
                if len_free_ind>0:
                    tmp_rate_ind = free_ind.pop()
                    len_free_ind -= 1
                else: # empty list
                    rate = np.append(rate,np.array([0])) # 0 as placeholder, actual rate computed below
                    TST_tmp_ind = np.append(TST_tmp_ind,np.array([[0,0,0]]),axis=0)
                    len_rate += 1
                    tmp_rate_ind = len_rate-1
                rate_ind[tuple_bool_index] = tmp_rate_ind
                TST_tmp_ind[tmp_rate_ind-1] = tuple_bool_index
        # update rates
        for neigh in neighborhood__:
            on, tn = np.mod(o+neigh,nm), np.mod(t+neigh,nm)
            on, tn = (on[0],on[1]), (tn[0],tn[1])
            # neighborhood first cell
            rate[rate_ind[on][0]] = np.exp(-E[on][0]/tau)
            rate[rate_ind[(on[0]-1) % nm[0],on[1]][0]] = np.exp(-E[(on[0]-1) % nm[0],on[1]][0]/tau)
            rate[rate_ind[on][1]] = np.exp(-E[on][1]/tau)
            rate[rate_ind[on[0],(on[1]-1) % nm[1]][1]] = np.exp(-E[on[0],(on[1]-1) % nm[1]][1]/tau)
            # neighborhood second cell
            rate[rate_ind[tn][0]] = np.exp(-E[tn][0]/tau)
            rate[rate_ind[(tn[0]-1) % nm[0],tn[1]][0]] = np.exp(-E[(tn[0]-1) % nm[0],tn[1]][0]/tau)
            rate[rate_ind[tn][1]] = np.exp(-E[tn][1]/tau)
            rate[rate_ind[tn[0],(tn[1]-1) % nm[1]][1]] = np.exp(-E[tn[0],(tn[1]-1) % nm[1]][1]/tau)

    with objmode():
        print("[+] Time needed: " + str(time.time() - start) + "s")

    return ifl_rel, order_rel, ifl, order, grid, grid_hist, t_real, (1. * cond < threshold), gammasMoor, gammasNeum

if __name__ == "__main__":

    b11, b12, b22 = -1.56, -3.06, -1.56
    tau = 1.
    MCS = 100
    threshold = 0.1 # = CondSort
    grid_size = 25
    printEveryStep = grid_size ** 2
    dataEveryStep = grid_size

    grid = npr.randint(2,size=(grid_size,grid_size)) # average state = 0.5
    gridn_start = copy.deepcopy(grid)

    ind = np.indices((grid_size,grid_size,2))
    ind = np.moveaxis(ind,0,-1)
    ind = ind.reshape((grid_size**2*2,3))

    print("[*] Starting ... ")

    ifl_rel, order_rel, ifl, order, grid, grid_hist, t_real, cond, gammasMoor, gammasNeum = cell_sorting_numba(grid,b11,b12,b22,tau,MCS,printEveryStep,dataEveryStep,threshold=threshold,ind=ind)
