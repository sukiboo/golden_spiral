
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
		fig = plt.figure(figsize=(9.4,9.4))
		ax = plt.axes()
		ax.set_axis_off()
		ax.set_xlim(-1.1,1.1)
		ax.set_xticks([])
		ax.set_ylim(-1.1,1.1)
		ax.set_yticks([])

		# plot the points
		ax.scatter(pts[:,0], pts[:,1], color=color)
		# plot the unit circle
		t = np.linspace(0,2*np.pi,100)
		##ax.plot(np.cos(t), np.sin(t), linewidth=2, color='gold')
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
		fig = plt.figure(figsize=(9.4,9.4))
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



def plot_lattice(pts0, pts1, color='dodgerblue', save=False):

	# set up figure and configure axes
	fig = plt.figure(figsize=(9.4,9.4))
	ax = plt.axes()
	ax.set_xlim(-.01, 1.01)
	ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
	ax.set_ylim(-.01, 1.01)
	ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

	# plot the points
	ax.scatter(pts0, pts1, color=color)
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
	##plt.savefig('./images/{:s}.png'.format(name), dpi=300, format='png')
	plt.savefig('./images/{:s}.pdf'.format(name), dpi=300, format='pdf')
	return



def save_animation(anim):
	name = time.strftime('%Y-%m-%d %H.%M.%S', time.localtime())
	anim.save('./images/'+name+'.gif', writer='imagemagick', fps=25, dpi=100)
	# #anim.save('./images/'+name+'.mp4', writer='ffmpeg', fps=25)
	return





def plot_points_mult(pts, cmap='winter', animate=False, save=False):

	# define rotation for 3d-animation
	def rotate(frame):

		# reconfigure the axes
		ax1.cla()
		ax1.set_axis_off()
		ax1.set_xlim(-.6,.6)
		ax1.set_ylim(-.6,.6)
		ax1.set_zlim(-.6,.6)
		ax2.cla()
		ax2.set_axis_off()
		ax2.set_xlim(-.6,.6)
		ax2.set_ylim(-.6,.6)
		ax2.set_zlim(-.6,.6)
		ax3.cla()
		ax3.set_axis_off()
		ax3.set_xlim(-.6,.6)
		ax3.set_ylim(-.6,.6)
		ax3.set_zlim(-.6,.6)
		ax4.cla()
		ax4.set_axis_off()
		ax4.set_xlim(-.6,.6)
		ax4.set_ylim(-.6,.6)
		ax4.set_zlim(-.6,.6)
		ax5.cla()
		ax5.set_axis_off()
		ax5.set_xlim(-.6,.6)
		ax5.set_ylim(-.6,.6)
		ax5.set_zlim(-.6,.6)
		ax6.cla()
		ax6.set_axis_off()
		ax6.set_xlim(-.6,.6)
		ax6.set_ylim(-.6,.6)
		ax6.set_zlim(-.6,.6)
		# #ax7.cla()
		# #ax7.set_axis_off()
		# #ax7.set_xlim(-.6,.6)
		# #ax7.set_ylim(-.6,.6)
		# #ax7.set_zlim(-.6,.6)
		# #ax8.cla()
		# #ax8.set_axis_off()
		# #ax8.set_xlim(-.6,.6)
		# #ax8.set_ylim(-.6,.6)
		# #ax8.set_zlim(-.6,.6)

		# plot the points
		ax1.scatter(pts[0][:,0], pts[0][:,1], pts[0][:,2], c=pts[0][:,2], cmap='autumn')
		ax2.scatter(pts[1][:,0], pts[1][:,1], pts[1][:,2], c=pts[1][:,2], cmap='autumn')
		ax3.scatter(pts[2][:,0], pts[2][:,1], pts[2][:,2], c=pts[2][:,2], cmap='autumn')
		ax4.scatter(pts[3][:,0], pts[3][:,1], pts[3][:,2], c=pts[3][:,2], cmap=cmap)
		ax5.scatter(pts[4][:,0], pts[4][:,1], pts[4][:,2], c=pts[4][:,2], cmap=cmap)
		ax6.scatter(pts[5][:,0], pts[5][:,1], pts[5][:,2], c=pts[5][:,2], cmap=cmap)
		# #ax7.scatter(pts[6][:,0], pts[6][:,1], pts[6][:,2], c=pts[6][:,2], cmap=cmap)
		# #ax8.scatter(pts[7][:,0], pts[7][:,1], pts[7][:,2], c=pts[7][:,2], cmap=cmap)

		# rotate
		ax1.view_init(azim=180-frame, elev=180-frame)
		ax2.view_init(azim=180-frame, elev=180-frame)
		ax3.view_init(azim=180-frame, elev=180-frame)
		ax4.view_init(azim=180-frame, elev=180-frame)
		ax5.view_init(azim=180-frame, elev=180-frame)
		ax6.view_init(azim=180-frame, elev=180-frame)
		# #ax7.view_init(azim=180-frame, elev=180-frame)
		# #ax8.view_init(azim=180-frame, elev=180-frame)

		return fig,


	# 3d-plot
	# set up figure and configure axes
	##fig = plt.figure(figsize=(18,6))
	fig = plt.figure(figsize=(15,10))
	# #fig = plt.figure(figsize=(18,9))

	ax1 = fig.add_subplot(2, 3, 1, projection='3d')
	ax1.view_init(azim=0, elev=-90)
	ax1.set_axis_off()
	ax1.set_xlim(-.6,.6)
	ax1.set_ylim(-.6,.6)
	ax1.set_zlim(-.6,.6)
	ax2 = fig.add_subplot(2, 3, 2, projection='3d')
	ax2.view_init(azim=0, elev=-90)
	ax2.set_axis_off()
	ax2.set_xlim(-.6,.6)
	ax2.set_ylim(-.6,.6)
	ax2.set_zlim(-.6,.6)
	ax3 = fig.add_subplot(2, 3, 3, projection='3d')
	ax3.view_init(azim=0, elev=-90)
	ax3.set_axis_off()
	ax3.set_xlim(-.6,.6)
	ax3.set_ylim(-.6,.6)
	ax3.set_zlim(-.6,.6)
	ax4 = fig.add_subplot(2, 3, 4, projection='3d')
	ax4.view_init(azim=0, elev=-90)
	ax4.set_axis_off()
	ax4.set_xlim(-.6,.6)
	ax4.set_ylim(-.6,.6)
	ax4.set_zlim(-.6,.6)
	ax5 = fig.add_subplot(2, 3, 5, projection='3d')
	ax5.view_init(azim=0, elev=-90)
	ax5.set_axis_off()
	ax5.set_xlim(-.6,.6)
	ax5.set_ylim(-.6,.6)
	ax5.set_zlim(-.6,.6)
	ax6 = fig.add_subplot(2, 3, 6, projection='3d')
	ax6.view_init(azim=0, elev=-90)
	ax6.set_axis_off()
	ax6.set_xlim(-.6,.6)
	ax6.set_ylim(-.6,.6)
	ax6.set_zlim(-.6,.6)
	# #ax7 = fig.add_subplot(2, 4, 7, projection='3d')
	# #ax7.view_init(azim=0, elev=-90)
	# #ax7.set_axis_off()
	# #ax7.set_xlim(-.6,.6)
	# #ax7.set_ylim(-.6,.6)
	# #ax7.set_zlim(-.6,.6)
	# #ax8 = fig.add_subplot(2, 4, 8, projection='3d')
	# #ax8.view_init(azim=0, elev=-90)
	# #ax8.set_axis_off()
	# #ax8.set_xlim(-.6,.6)
	# #ax8.set_ylim(-.6,.6)
	# #ax8.set_zlim(-.6,.6)

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
		ax1.scatter(pts[0][:,0], pts[0][:,1], pts[0][:,2], c=pts[0][:,2], cmap='autumn')
		ax2.scatter(pts[1][:,0], pts[1][:,1], pts[1][:,2], c=pts[1][:,2], cmap='autumn')
		ax3.scatter(pts[2][:,0], pts[2][:,1], pts[2][:,2], c=pts[2][:,2], cmap='autumn')
		ax4.scatter(pts[3][:,0], pts[3][:,1], pts[3][:,2], c=pts[3][:,2], cmap=cmap)
		ax5.scatter(pts[4][:,0], pts[4][:,1], pts[4][:,2], c=pts[4][:,2], cmap=cmap)
		ax6.scatter(pts[5][:,0], pts[5][:,1], pts[5][:,2], c=pts[5][:,2], cmap=cmap)
		# #ax7.scatter(pts[6][:,0], pts[6][:,1], pts[6][:,2], c=pts[6][:,2], cmap=cmap)
		# #ax8.scatter(pts[7][:,0], pts[7][:,1], pts[7][:,2], c=pts[7][:,2], cmap=cmap)
		# show the plot
		plt.tight_layout()
		# save or show the plot
		if save:
			save_plot()
			plt.close()
		else:
			plt.show()

	return


