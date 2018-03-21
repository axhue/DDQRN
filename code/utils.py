import os
from moviepy.editor import ImageSequenceClip
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def get_output_folder(args, parent_dir, env_name, task_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print('===== Folder did not exist; creating... %s'%parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id) + '-' + task_name
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print('===== Folder did not exist; creating... %s'%parent_dir)
    else:
        print('===== Folder exists; delete? %s'%parent_dir)
        input("Press Enter to continue...")
        os.system('rm -rf %s/' % (parent_dir))
    #s.makedirs(parent_dir+'/videos/')
    #os.makedirs(parent_dir+'/images/')
    return parent_dir

# https://gist.github.com/nirum/d4224ad3cd0d71bfef6eba8f3d6ffd59
def gif(array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy

    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e

    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)

    Parameters
    ----------

    array : array_like
        A numpy array that contains a sequence of images

    fps : int
        frames per second (default: 10)

    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    #fname, _ = os.path.splitext(filename)
    #filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    #clip.write_gif(filename, fps=fps)
    return clip
    
# https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(np.squeeze(image,-1) )
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
def smoothed_plot(dataframe,color,smooth,ax=None,title=None,xlabel=None,ylabel=None):
    '''
    dataframe is a pandas dataframe you can read in data using pd.read_csv x axis is 'Step' and y axis is 'Value'
    '''
    if ax is None:
        f,ax = plt.subplots()
    smooth_data = savgol_filter(dataframe['Value'],smooth,4)
    ax.plot(dataframe['Step'],dataframe['Value'],color=color,alpha=0.4)
    ax.plot(dataframe['Step'],smooth_data,color=color)
    return f,ax
    