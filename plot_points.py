
''' cisualize generated point sequences '''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
import numpy as np
import time


def plot_points(dim, obj, pts, cmap='winter', color='dodgerblue', animate=False, save=False):

    # define rotation for 3d-animation
    def rotate(frame):
        # reconfigure the axes
        ax.cla()
        ax.set_axis_off()
        ax.set_xlim(-.6,.6)
        ax.set_ylim(-.6,.6)
        ax.set_zlim(-.6,.6)
        # plot the points
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap=cmap)
        # rotate
        ax.view_init(azim=180-frame, elev=180-frame)
        return fig,

    # 2d-plot
    if (obj in ['sphere','s'] and dim==1) or (obj in ['ball','b'] and dim==2):
        # set up figure and configure axes
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes()
        ax.set_axis_off()
        ax.set_xlim(-1.1,1.1)
        ax.set_xticks([])
        ax.set_ylim(-1.1,1.1)
        ax.set_yticks([])
        # plot the points
        ax.scatter(pts[:,0], pts[:,1], color=color)
        # plot the unit circle
        tt = np.linspace(0, 2*np.pi, 1000)
        ax.scatter(np.sin(tt), np.cos(tt), color='turquoise', alpha=.5, s=1)
        # show the plot
        plt.tight_layout()
        # save or show the plot
        if save:
            save_plot()
            plt.close()
        else:
            plt.show()

    # 3d-plot
    elif (obj in ['sphere','s'] and dim==2) or (obj in ['ball','b'] and dim==3):
        # set up figure and configure axes
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_axis_off()
        ax.view_init(azim=30, elev=-90)#150)#-90)
        ax.set_xlim(-.6,.6)
        ax.set_ylim(-.6,.6)
        ax.set_zlim(-.6,.6)
        # plot the points
        if animate:
            anim = animation.FuncAnimation(fig, rotate, frames=360, interval=40)
            plt.tight_layout()
            # save the animation or show the plot
            if save:
                save_animation(anim)
                plt.close()
            else:
                plt.show()
        else:
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap=cmap)
            # show the plot
            plt.tight_layout()
            # save or show the plot
            if save:
                save_plot()
                plt.close()
            else:
                plt.show()
    return

def save_plot():
    name = time.strftime('%Y-%m-%d %H.%M.%S', time.localtime())
    # plt.savefig('./images/{:s}.png'.format(name), dpi=300, format='png')
    plt.savefig('./images/{:s}.pdf'.format(name), dpi=300, format='pdf')
    return

def save_animation(anim):
    name = time.strftime('%Y-%m-%d %H.%M.%S', time.localtime())
    anim.save('./images/'+name+'.gif', writer='imagemagick', fps=25, dpi=100)
    # anim.save('./images/'+name+'.mp4', writer='ffmpeg', fps=25)
    return

def plot_lattice(pts0, pts1, color='dodgerblue', save=False):
    # set up figure and configure axes
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes()
    ax.set_xlim(-.01, 1.01)
    ax.set_ylim(-.01, 1.01)
    ax.set_xticks([])
    ax.set_yticks([])
    # plot the points
    ax.scatter(pts0, pts1, color=color)
    ax.scatter(pts0, np.zeros(pts0.shape)-.005, color='firebrick', marker='|')
    ax.scatter(np.zeros(pts1.shape)-.005, pts1, color='firebrick', marker='_')
    # show the plot
    plt.tight_layout()
    # save or show the plot
    if save:
        save_plot()
        plt.close()
    else:
        plt.show()
    return

def plot_sequences(seq, animate=False):
    # set up figure and configure axes
    fig = plt.figure(figsize=(20,8))
    ax = plt.axes()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([])
    # find the number of sequences
    num_seq, num_pts = seq.shape
    # generate y-coordinates
    y_crd = np.tile(np.linspace(0, 1, num_seq+2)[1:-1], (num_pts,1)).T
    if animate:
        # plot a frame
        def plot_frame(frame):
            # reconfigure the axes
            ax.cla()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticks([])
            # plot points
            for n in range(num_seq):
                ax.scatter(seq[n,:frame+1], y_crd[n,:frame+1])
            # highlight the largest interval
            ss = np.append(np.sort(seq[0,:frame+1]), 1)
            ii = np.argmax(ss[1:] - ss[:-1])
            ax.scatter(np.linspace(ss[ii], ss[ii+1], seq.shape[1]+2)[1:-1],\
                y_crd[0,:seq.shape[1]], marker='_', alpha=.5)
            return fig,
        # display animation
        anim = animation.FuncAnimation(fig, plot_frame, frames=num_pts, interval=100)
        anim.save('./animation/' + str(num_seq) + '_' + str(num_pts) + '.gif', writer='imagemagick', fps=num_pts/10, dpi=100)
        # anim.save('./animation/' + str(num_seq) + '_' + str(num_pts) + '.mp4', writer='ffmpeg', fps=num_pts/10)
    else:
        # plot points
        for n in range(num_seq):
            ax.scatter(seq[n], y_crd[n])
    plt.tight_layout()
    plt.show()
    return
