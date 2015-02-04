import string, itertools

import numpy as np
import matplotlib.pyplot as plt 
import Graphics as artist 
import matplotlib.gridspec as gridspec

from matplotlib import rcParams
from awesome_print import ap 

plt.xkcd()
def jaccard_similarity(a,b):
	a = set(a)
	b = set(b)

	try:
		return len(a & b)/float(len(a | b))
	except: 
		return 0 

def unique_words(aStr):
	return ' '.join([word for word in set(aStr.split())])


def on_draw(event):
    """Auto-wraps all text objects in a figure at draw-time"""
    import matplotlib as mpl
    fig = event.canvas.figure

    # Cycle through all artists in all the axes in the figure
    for ax in fig.axes:
        for artist in ax.get_children():
            # If it's a text artist, wrap it...
            if isinstance(artist, mpl.text.Text):
                autowrap_text(artist, event.renderer)

    # Temporarily disconnect any callbacks to the draw event...
    # (To avoid recursion)
    func_handles = fig.canvas.callbacks.callbacks[event.name]
    fig.canvas.callbacks.callbacks[event.name] = {}
    # Re-draw the figure..
    fig.canvas.draw()
    # Reset the draw event callbacks
    fig.canvas.callbacks.callbacks[event.name] = func_handles

def autowrap_text(textobj, renderer):
    """Wraps the given matplotlib text object so that it exceed the boundaries
    of the axis it is plotted in."""
    import textwrap
    # Get the starting position of the text in pixels...
    x0, y0 = textobj.get_transform().transform(textobj.get_position())
    # Get the extents of the current axis in pixels...
    clip = textobj.get_axes().get_window_extent()
    # Set the text to rotate about the left edge (doesn't make sense otherwise)
    textobj.set_rotation_mode('anchor')

    # Get the amount of space in the direction of rotation to the left and 
    # right of x0, y0 (left and right are relative to the rotation, as well)
    rotation = textobj.get_rotation()
    right_space = min_dist_inside((x0, y0), rotation, clip)
    left_space = min_dist_inside((x0, y0), rotation - 180, clip)

    # Use either the left or right distance depending on the horiz alignment.
    alignment = textobj.get_horizontalalignment()
    if alignment is 'left':
        new_width = right_space 
    elif alignment is 'right':
        new_width = left_space
    else:
        new_width = 2 * min(left_space, right_space)

    # Estimate the width of the new size in characters...
    aspect_ratio = 0.5 # This varies with the font!! 
    fontsize = textobj.get_size()
    pixels_per_char = aspect_ratio * renderer.points_to_pixels(fontsize)

    # If wrap_width is < 1, just make it 1 character
    wrap_width = max(1, new_width // pixels_per_char)
    try:
        wrapped_text = textwrap.fill(textobj.get_text(), wrap_width)
    except TypeError:
        # This appears to be a single word
        wrapped_text = textobj.get_text()
    textobj.set_text(wrapped_text)

def min_dist_inside(point, rotation, box):
    """Gets the space in a given direction from "point" to the boundaries of
    "box" (where box is an object with x0, y0, x1, & y1 attributes, point is a
    tuple of x,y, and rotation is the angle in degrees)"""
    from math import sin, cos, radians
    x0, y0 = point
    rotation = radians(rotation)
    distances = []
    threshold = 0.0001 
    if cos(rotation) > threshold: 
        # Intersects the right axis
        distances.append((box.x1 - x0) / cos(rotation))
    if cos(rotation) < -threshold: 
        # Intersects the left axis
        distances.append((box.x0 - x0) / cos(rotation))
    if sin(rotation) > threshold: 
        # Intersects the top axis
        distances.append((box.y1 - y0) / sin(rotation))
    if sin(rotation) < -threshold: 
        # Intersects the bottom axis
        distances.append((box.y0 - y0) / sin(rotation))
    return min(distances)

data = {}

for name in ['eigvecs','projections','eigvals']:
	data[name] = np.array(np.loadtxt('../data/%s'%name))

TEXT = 1
punkt = set(string.punctuation)
stopwords = set(open('stopwords').read().splitlines())
text = [line.lower() for line in open('../data/cleansed_abstracts','rb').read().splitlines()]
text = [' '.join([''.join(ch for ch in word if ord(ch)<128) for word in line.split() 
		if word not in stopwords and 'file://' not in word and not any([ch in punkt for ch in word])]) for line in text]
basis_vectors = [unique_words(line.split(':')[TEXT]) for line in open('../data/lda-topics.txt','rb').read().splitlines()]

data['vocab'] = text	
#Calculate Jaccard similarity of one text document with basis vectors
projections  = np.array([[jaccard_similarity(vector,sample) for vector in basis_vectors] for sample in data['vocab']])
orthogonal_projections = projections.dot(data['projections'])

max_color = max(orthogonal_projections.max(),projections.max())
min_color = min(orthogonal_projections.min(),projections.min())


#Figure out words in each eigentopic
all_words = list(set(list(itertools.chain.from_iterable([vector.split() for vector in basis_vectors]))))

#positive words
pos_words = set(' '.join([basis_vectors[i] for i,vl in enumerate(np.sort(data['projections'][0,:])) if vl>0]).split())
neg_words = set(' '.join([basis_vectors[i] for i,vl in enumerate(np.sort(data['projections'][0,:])) if vl<0]).split())
tmp = neg_words

pos_words -= neg_words
tmp -= pos_words
neg_words = tmp

pos_words = ' '.join(pos_words)
neg_words = ' '.join(neg_words)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(orthogonal_projections[:,0],orthogonal_projections[:,1],c=orthogonal_projections[:,2],cmap=plt.cm.jet)
ax.annotate(' '.join(pos_words.split()[:4]), xy=(.1, .95), xycoords='axes fraction',fontsize=14,color='green')
ax.annotate(' '.join(pos_words.split()[4:8]), xy=(.1, .9), xycoords='axes fraction',fontsize=14,color='green')

ax.annotate(' '.join(neg_words.split()[:4]), xy=(.1, .8), xycoords='axes fraction',fontsize=14,color='red')
ax.annotate(' '.join(neg_words.split()[4:8]), xy=(.1, .75), xycoords='axes fraction',fontsize=14,color='red')

fig.canvas.mpl_connect('draw_event', on_draw)
artist.adjust_spines(ax)
ax.set_xlabel('1st Eigentopic')
ax.set_ylabel('2nd Eigentopic')
plt.tight_layout()
plt.savefig('../images/scatterplot-xkcd.png',dpi=300)

'''
fig = plt.figure(figsize=(12,5))
gspec = gridspec.GridSpec(1, 3,width_ratios=[4,5,3])
nonorthogonal = plt.subplot(gspec[0])
orthogonal = plt.subplot(gspec[1])
scree = plt.subplot(gspec[2])

nonorthogonal.imshow(projections,interpolation='nearest',aspect='auto', vmin=min_color,vmax=max_color)
cax = orthogonal.imshow(orthogonal_projections,interpolation='nearest',aspect='auto',vmin=min_color,vmax=max_color)
scree.plot(data['eigvals']/data['eigvals'].sum(),'k.-',linewidth=2)

orthogonal.set_xlabel('Eigentopics')
orthogonal.set_title('Orthogonalized')
nonorthogonal.set_xlabel('Topics')
nonorthogonal.set_ylabel('Abstracts')
nonorthogonal.set_title('Not Orthogonalized')
scree.set_xlabel('Eigenvector')
scree.set_ylabel('Fraction of Variance')

map(artist.adjust_spines,[nonorthogonal,orthogonal,scree])
cbar = plt.colorbar(cax,ax=orthogonal,use_gridspec=True)
cbar.set_label('Projection')
plt.tight_layout()
plt.savefig('../images/orthogonal-nonorthogonal-projections')
'''
