import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from dataclasses import make_dataclass
import os
import sys
import pickle
import time
import csv
from labyrinth_env import LabyrinthEnv

# Value iteration algorithm
def value_iteration(env, gamma=0.9, theta=1e-6, save_all_policies=False):
    n_states = env.n_states
    n_actions = env.n_actions
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    P_a = env.get_transition_mat()

    R = np.zeros(n_states)
    if env.reward_map is None:
        R[env.water_port] = 1
    else:
        R = env.reward_map

    valid_actions_per_state = [[] for _ in range(n_states)]
    for s in range(n_states):
        for a in range(n_actions):
            if a == 3:
                valid_actions_per_state[s].append(a)
                continue
            next_state_indices = np.where(P_a[s, :, a] == 1)[0]
            if len(next_state_indices) == 1:
                next_s_prime = next_state_indices[0]
                if next_s_prime != s:
                    valid_actions_per_state[s].append(a)

    iteration = 0
    unique_policies = []
    unique_policies_set = set()
    while True:
        delta = 0
        V_new = np.copy(V)
        policy_new = np.copy(policy)

        for s in range(n_states):
            q_values = np.full(n_actions, -np.inf)
            for a in valid_actions_per_state[s]:
                next_state_indices = np.where(P_a[s, :, a] == 1)[0]
                next_s_prime = next_state_indices[0]
                q_values[a] = R[next_s_prime] + gamma * V[next_s_prime]
            if len(valid_actions_per_state[s]) > 0:
                V_new[s] = np.max(q_values)
                policy_new[s] = np.argmax(q_values)
            else:
                V_new[s] = V[s]
                policy_new[s] = policy[s]
            delta = max(delta, abs(V_new[s] - V[s]))

        # Save unique policies
        if save_all_policies:
            policy_tuple = tuple(policy_new.tolist())
            if policy_tuple not in unique_policies_set:
                unique_policies.append(policy_new.copy())
                unique_policies_set.add(policy_tuple)

        V = np.copy(V_new)
        policy = np.copy(policy_new)
        iteration += 1
        if delta < theta:
            break

    # Policy Extraction
    for s in range(n_states):
        q_values = np.full(n_actions, -np.inf)
        for a in valid_actions_per_state[s]:
            next_state_indices = np.where(P_a[s, :, a] == 1)[0]
            next_s_prime = next_state_indices[0]
            q_values[a] = R[next_s_prime] + gamma * V[next_s_prime]
        policy[s] = np.argmax(q_values)

    print(f"Value Iteration converged in {iteration} iterations.")
    if save_all_policies:
        return V, policy, unique_policies
    else:
        return V, policy

# Maze Utilities
Maze = make_dataclass('Maze', ['le','ru','pa','ch','xc','yc','ce','rc','di','cl','wa','st'])

def NewMaze(n=6):
    ru = []
    pa = []
    for i in range(n+1):
        xd = (i+1)%2
        yd = i%2
        di = 2**((n-i)//2)
        for j in range(2**i):
            if i==0:
                k = -1
                pa.append(k)
                (x,y) = (int(2**(n/2)-1),int(2**(n/2)-1))
                ru.append([(x1,y) for x1 in range(0,x+1)])
            else:
                k = 2**(i-1)-1+j//2
                pa.append(k)
                (x0,y0) = ru[k][-1]
                (xs,ys) = (xd*(2*(j%2)-1),yd*(2*(j%2)-1))
                (x,y) = (x0+xs*di,y0+ys*di)
                if xs==0:
                    ru.append([(x,y1) for y1 in range(y0+ys,y+ys,ys)])
                else:
                    ru.append([(x1,y) for x1 in range(x0+xs,x+xs,xs)])
    ce = {}
    lo = {}
    c = 0
    for r in ru:
        for p in r:
            ce[p] = c
            lo[c] = p
            c += 1
    nc = c
    ru = [[ce[p] for p in r] for r in ru]
    pa = np.array(pa)
    ch = np.full((len(ru),2),-1,dtype=int)
    for i,p in enumerate(pa):
        if p>=0:
            if ch[p,0]==-1:
                ch[p,0]=i
            else:
                ch[p,1]=i
    xc = np.array([lo[c][0] for c in range(nc)])
    yc = np.array([lo[c][1] for c in range(nc)])
    ma = Maze(le=n,ru=ru,pa=pa,ch=ch,xc=xc,yc=yc,ce=ce,rc=None,di=None,cl=None,wa=None,st=None)
    ma.rc = np.array([RunIndex(c,ma) for c in range(nc)])
    ma.di = ConnectDistance(ma)
    ma.cl = MazeCenter(ma)
    ma.wa = MazeWall(ma)
    ma.st = MakeStepType(ma)
    return ma

def RunIndex(c,m):
    for i,r in enumerate(m.ru):
        if c in r:
            return i

def HomePath(c,m):
    ret = []
    i = m.rc[c]
    ret+=m.ru[i][m.ru[i].index(c)::-1]
    i = m.pa[i]
    while i>=0:
        ret+=m.ru[i][::-1]
        i = m.pa[i]
    return ret

def HomeDistance(m):
    di = np.zeros(len(m.xc))
    for r in m.ru:
        for c in r:
            di[c]=len(HomePath(c,m))-1
    return di

def ConnectPath(c1,c2,m):
    r1 = HomePath(c1,m)
    r2 = HomePath(c2,m)[::-1]
    for i in r1:
        if i in r2:
            return (r1[:r1.index(i)]+r2[r2.index(i):])

def ConnectDistance(m):
    nc = len(m.xc)
    di = np.array([[len(ConnectPath(c1,c2,m))-1 for c2 in range(nc)] for c1 in range(nc)])
    return di

def MakeStepType(m):
    exitstate=len(m.ru)
    st = np.full((len(m.ru)+1,len(m.ru)+1),-1,dtype=int)
    for i in range (m.le+1):
        for j in range (2**i-1,2**(i+1)-1):
            if j>0:
                if (i+j+m.pa[j])%2 == 0:
                    st[j,m.pa[j]]=2
                else:
                    st[j,m.pa[j]]=3
            if i<m.le:
                for c in m.ch[j]:
                    if (i+j+c)%2 == 0:
                        st[j,c]=1
                    else:
                        st[j,c]=0
    st[0,exitstate]=3
    return st

def StepType(i,j,m):
    return m.st[int(i),int(j)]

def StepType2(i,j,m):
    st2 = m.st[int(i),int(j)]
    if st2==3:
        st2=2
    return st2

def StepType3(i,j,m):
    st3 = m.st[int(i),int(j)]
    if st3==0 or st3==1:
        st3=0
    elif st3==2 or st3==3:
        st3-=1
    return st3

def MazeCenter(m):
    def acc(i):
        r = m.ru[i][:]
        if m.ch[i,0]!=-1:
            r += acc(m.ch[i][0])
            r += [m.ru[i][-1]]
            r += acc(m.ch[i][1])
            r += m.ru[i][-1::-1]
        return r
    c = acc(0)
    return np.array([m.xc[c],m.yc[c]]).T

def PlotMazeCenter(m,axes=None,numbers=False):
    w = MazeCenter(m)
    if axes is None:
        fig, axes = plt.subplots(figsize=(6,6))
    axes.plot(w[:,0],w[:,1],color='r',linestyle='-',linewidth=1)
    axes.set_aspect('equal', adjustable='box')
    axes.invert_yaxis()
    if numbers:
        for c in range(len(m.xc)):
            axes.text(m.xc[c],m.yc[c],'{:d}'.format(c))
    return axes

def MazeWall(m):
    (xc,yc,ru,pa) = (m.xc,m.yc,m.ru,m.pa)
    ch = [np.where(np.array(pa)==i)[0].astype(int) for i in range(len(ru))]

    def acw(i):
        r = ru[i]
        c0 = np.array([xc[r[0]],yc[r[0]]])
        c1 = np.array([xc[r[-1]],yc[r[-1]]])
        if i==0:
            d = np.array([1,0])
        else:
            p1 = np.array([xc[ru[pa[i]][-1]],yc[ru[pa[i]][-1]]])
            d = c0-p1
        sw = 0.5*np.array([-d[0]-d[1],d[0]-d[1]])
        se = 0.5*np.array([-d[0]+d[1],-d[0]-d[1]])
        nw = 0.5*np.array([d[0]-d[1],d[0]+d[1]])
        ne = 0.5*np.array([d[0]+d[1],-d[0]+d[1]])
        if i==0:
            p = [c0+sw]
        else:
            p = []
        p += [c1+sw]
        if len(ch[i]):
            e = np.array([xc[ru[ch[i][0]][0]],yc[ru[ch[i][0]][0]]])-c1
            if np.array_equal(e,np.array([-d[1],d[0]])):
                il = ch[i][0]; ir = ch[i][1]
            else:
                il = ch[i][1]; ir = ch[i][0]
            p += acw(il)
            p += [c1+ne]
            p += acw(ir)
            p += [c0+se]
        else:
            p += [c1+nw, c1+ne, c1+se]
        return p
    return np.array(acw(0))

def PlotMazeWall(m,axes=None,figsize=4):
    if axes is None:
        fig, axes = plt.subplots(figsize=(figsize,figsize))
    axes.plot(m.wa[:,0],m.wa[:,1],color='k',linestyle='-',linewidth=2)
    axes.set_aspect('equal', adjustable='box')
    axes.invert_yaxis()
    axes.set_xticks([])
    axes.set_yticks([])
    return axes

def PlotMazeNums(m,ax,mode='cells',numcol='blue'):
    if mode=='nodes':
        for j,r in enumerate(m.ru):
            # Condition to label: if node has children (branch point) or is the root node
            if len(m.ch[j]) > 0 or j == 0:
                x = m.xc[r[-1]]; y=m.yc[r[-1]] # Get coordinates of the end cell of the run
                ax.text(x + 0.35, y + 0.35, '{:d}'.format(j), color=numcol, fontsize=8, ha='left', va='bottom', weight='bold')
                ax.plot(x,y, '.', color='dimgray', markersize=4)
    elif mode=='cells':
        for j in range(len(m.xc)):
            x = m.xc[j]; y=m.yc[j]
            ax.text(x-.35,y+.15,'{:d}'.format(j),color=numcol)

    elif mode=='runs':
        for j,r in enumerate(m.ru):
            xlo = min(m.xc[r]); xhi = max(m.xc[r]); ylo = min(m.yc[r]); yhi = max(m.yc[r])
            ax.add_patch(patches.Rectangle((xlo-0.5,ylo-0.5),xhi-xlo+1,yhi-ylo+1,lw=1,fill=False))
            x = 0.5*(m.xc[r[0]]+m.xc[r[-1]])-0.35; y = 0.5*(m.yc[r[0]]+m.yc[r[-1]])+0.15
            ax.text(x,y,'{:d}'.format(j),color=numcol)

def PlotMazeFunction(f,m,mode='cells',numcol='cyan',figsize=4,col=None,axes=None):
    if col is None:
        cmap = plt.cm.Purples
        norm = plt.Normalize(np.min(f), np.max(f))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    if axes is None:
        fig, axes = plt.subplots(figsize=(figsize,figsize))
    ax=axes
    PlotMazeWall(m,axes=ax,figsize=figsize)

    if mode=='nodes':
        for j,r in enumerate(m.ru):
            x = m.xc[r[-1]]; y=m.yc[r[-1]]
            if not(f is None):
                ax.add_patch(patches.Rectangle((x-0.5,y-0.5),1.0,1.0,lw=0,
                                           fc=cmap(norm(f[j])), ec='gray'))
            if numcol:
                ax.text(x-.35,y+.15,'{:d}'.format(j),color=numcol)
            ax.text(x-0.35,y+.15,'.',color='black',  fontsize=10)
        plt.colorbar(sm, ax=ax, ticks=[np.min(f),np.max(f)], fraction=0.046, pad=0.04)
    if mode=='cells':
        for j in range(len(m.xc)):
            x = m.xc[j]; y=m.yc[j]
            if not(f is None):
                ax.add_patch(patches.Rectangle((x-0.5,y-0.5),1,1,lw=0,
                                           fc=cmap(norm(f[j])), ec='gray'))
            if numcol:
                ax.text(x-.35,y+.15,'{:d}'.format(j),color=numcol)

    if mode=='runs':
        for j,r in enumerate(m.ru):
            xlo = min(m.xc[r]); xhi = max(m.xc[r]); ylo = min(m.yc[r]); yhi = max(m.yc[r])
            if not(f is None):
                ax.add_patch(patches.Rectangle((xlo-0.5,ylo-0.5),xhi-xlo+1,yhi-ylo+1,lw=0,
                                          fc=cmap(norm(f[j])), ec='gray'))
            else:
                ax.add_patch(patches.Rectangle((xlo-0.5,ylo-0.5),xhi-xlo+1,yhi-ylo+1,lw=1,
                                          color='black',fill=False))
            if numcol:
                x = 0.5*(m.xc[r[0]]+m.xc[r[-1]])-0.35; y = 0.5*(m.yc[r[0]]+m.yc[r[-1]])+0.15
                ax.text(x,y,'{:d}'.format(j),color=numcol)
    return ax

def PlotMazeCells(m,numcol='blue',figsize=6):
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    PlotMazeFunction(None,m,mode='cells',numcol=numcol,figsize=figsize,col=None, axes=ax)
    return ax

def PlotMazeRuns(m,numcol='blue',figsize=6):
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    PlotMazeFunction(None,m,mode='runs',numcol=numcol,figsize=figsize,col=None, axes=ax)
    return ax

def PlotMazeNodes(m,numcol='blue',figsize=6):
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    PlotMazeFunction(None,m,mode='nodes',numcol=numcol,figsize=figsize,col=None, axes=ax)
    return ax

def NodeLevel(n):
    return int(np.floor(np.log(n+1)/np.log(2)))


# Main Execution and Visualization
if __name__ == "__main__":
    n_states = 127 # LabyrinthEnv has n_states=127, which matches NewMaze(n=6)
    maze_n_param = 6 

    # Four Randomly Switching Rewarding Nodes
    print("\n\n--- Running Value Iteration for four randomly switching rewarding nodes (using Expected Rewards) ---")

    possible_reward_nodes_stochastic = [63, 84, 105, 126] # These are LabyrinthEnv states
    reward_value_at_target = 1
    prob_each_node_is_active = 1.0 / len(possible_reward_nodes_stochastic)

    R_expected = np.zeros(n_states)
    for s_idx in range(n_states):
        if s_idx in possible_reward_nodes_stochastic:
            R_expected[s_idx] = reward_value_at_target * prob_each_node_is_active

    print("\nCalculated Expected Reward (R_expected) for each state:")
    print(np.round(R_expected, 3))

    env_stochastic_reward = LabyrinthEnv(n_states=n_states, reward_state=0, reward_map=R_expected)
    gamma = 0.99
    theta = 1e-9
    optimal_V_stochastic, optimal_policy_stochastic, all_unique_policies = value_iteration(env_stochastic_reward, gamma, theta, save_all_policies=True)

    print("\nOptimal Value Function (V) for stochastic reward states:")
    print(np.round(optimal_V_stochastic, 3))
    print("\nOptimal Policy (0: left, 1: right, 2: reverse, 3: stay) for stochastic reward states:")
    print(optimal_policy_stochastic)

    print("\n--- Testing and Plotting for each unique policy during value iteration ---")
    maze_obj_stochastic = NewMaze(n=maze_n_param)
    start_state_stochastic = env_stochastic_reward.home_state
    get_node_coords = lambda state_idx, maze_obj: (maze_obj.xc[maze_obj.ru[state_idx][-1]], maze_obj.yc[maze_obj.ru[state_idx][-1]])
    policy_results = []
    for idx, policy_to_test in enumerate(all_unique_policies):
        print(f"\n--- Policy #{idx+1} ---")
        reset_result = env_stochastic_reward.reset(start_state_stochastic)
        if isinstance(reset_result, tuple):
            current_state = reset_result[0]
        else:
            current_state = reset_result
        path_coords = [get_node_coords(current_state, maze_obj_stochastic)]
        step_records = []
        total_reward = 0
        done = False
        step_count = 0
        start_time = time.time()
        while not done and step_count < env_stochastic_reward.max_episode_length:
            action = policy_to_test[current_state]
            prev_state = current_state
            _, _, next_state, reward_from_env_step, done = env_stochastic_reward.step(action)
            path_coords.append(get_node_coords(next_state, maze_obj_stochastic))
            step_records.append((prev_state, action, next_state, reward_from_env_step))
            total_reward += reward_from_env_step
            current_state = next_state
            step_count += 1
            if done:
                break
        elapsed_time = time.time() - start_time
        correct = 1 if done else -1
        policy_results.append({
            'Trial': idx+1,
            'Steps': step_count,
            'TrialDur': elapsed_time * 1000,
            'Correct': correct
        })
        print(f"Steps to reach reward: {step_count}")
        print(f"Wall-clock time to simulate policy: {elapsed_time:.6f} seconds")
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        PlotMazeWall(maze_obj_stochastic, axes=ax)
        PlotMazeNums(maze_obj_stochastic, ax, mode='nodes', numcol='red')
        start_coords = get_node_coords(start_state_stochastic, maze_obj_stochastic)
        ax.plot(start_coords[0], start_coords[1], 'o', color='blue', markersize=10, label='Start State', zorder=3)
        for node in possible_reward_nodes_stochastic:
            node_coords = get_node_coords(node, maze_obj_stochastic)
            ax.plot(node_coords[0], node_coords[1], 's', color='gold', markersize=12, markeredgecolor='black', markeredgewidth=1.5, alpha=0.7, zorder=3)
        ax.plot([], [], 's', color='gold', markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='Possible Reward Nodes')
        for i in range(len(step_records)):
            prev_s, action, next_s, _ = step_records[i]
            start_xy = get_node_coords(prev_s, maze_obj_stochastic)
            end_xy = get_node_coords(next_s, maze_obj_stochastic)
            ax.plot(start_xy[0], start_xy[1], 'o', color='cyan', markersize=5, zorder=2)
            arrow_color = 'cyan'
            mutation_scale = 15
            connection_style = "arc3,rad=0.0"
            ax.annotate('', xy=end_xy, xytext=start_xy, arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2, mutation_scale=mutation_scale, connectionstyle=connection_style), zorder=2)
        ax.set_title(f'Policy #{idx+1}: Steps to reward = {step_count}')
        ax.legend()
        plt.show()
    # Save results to CSV
    with open('policy_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Trial', 'Steps', 'TrialDur', 'Correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in policy_results:
            writer.writerow(row)
    print('Results saved to policy_results.csv')
