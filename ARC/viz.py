from ARC import * 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def stroke_stats():    
    # load stroke data and display aggregate stroke statistics 
    
    filename = "./data/omniglot_strokes.npz"  
    load_data = np.load(filename, encoding =  'bytes')
    
    # data here has the format:
    # (delta X, delta Y, pen on/off)
    train_set = load_data['train']
    valid_set = load_data['valid']
    test_set = load_data['test']
    
    # computes euclidean distance of a stroke
    def stroke_distance(del_x, del_y):
        return np.sqrt( del_x**2 + del_y**2 )

    # keep  track of number of times the pen was lifted, so # of strokes
    # also the average stroke length 
    stroke_data = {}
    stroke_data['num_strokes'] = []
    stroke_data['avg_stroke_len'] = []
    select_indices = []

    for ind, sample in enumerate(train_set):
        sample_lines = []
        cur_line = 0
        for pen_movement in sample:
            cur_line += stroke_distance( pen_movement[0], pen_movement[1] ) 
            
            if pen_movement[2] == 1:
                sample_lines.append(cur_line)
                cur_line = 0
        
        if len(sample_lines) > 5:
            select_indices.append(sample)
            
        stroke_data['avg_stroke_len'].append( np.mean(sample_lines))
        stroke_data['num_strokes'].append(len(sample_lines)) 

    select_indices = np.array(select_indices)

    plt.hist(stroke_data['avg_stroke_len'], bins = 100 )
    plt.title('Stroke Length Distribution, all alphabets')
    plt.xlabel('pixels')
    plt.xlim(xmax=2500)
    plt.show()

    data = np.array( stroke_data['num_strokes'] )
    plt.hist( data , bins = 50 )
    plt.title('Distribution of number of strokes per character, all alphabets')
    plt.xlabel('number of strokes')
    plt.xlim( xmax=13 )
    plt.show()

def tsne_alph(alphabet_index = 23, perp= 80):
    # t-SNE projection for an alphabet with given perplexity

    a_size = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22,
                    16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26,
                    26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]
    a_start = [0, 20, 49, 75, 116, 156, 180, 226, 240, 266, 300, 333, 355, 381,
                    424, 448, 496, 518, 534, 586, 633, 673, 699, 739, 780, 813,
                    827, 869, 892, 909, 964, 984, 1010, 1036, 1062, 1088, 1114,
                    1159, 1204, 1245, 1271, 1318, 1358, 1388, 1433, 1479, 1507,
                    1530, 1555, 1597]

    chars = np.load('./data/omniglot.npy')
    image_size = 105 
    resized_chars = chars
    # (1623, 20, image_size, image_size)
    resized_chars = resized_chars.reshape(-1, image_size*image_size)
    resized_chars = resized_chars/255
    resized_chars -= resized_chars.mean()/255 

    #5, 6 = Japanese
    
    # here we include 2000 extra drawings during the t-SNE computation, but do not project them.
    # this adds more structure to the low dimension output
    indices = list( range( (a_start[alphabet_index]*20 )-2000,
                             (a_start[alphabet_index]+a_size[alphabet_index])*20 ) )
    alphabet_range = resized_chars[ indices  ]

    #PCA was also tried, the results are also interesting (sometimes)
    #chars_embedded = PCA(n_components=2).fit_transform(alphabet_range)

    chars_embedded = TSNE(n_components=2, perplexity=perp).fit_transform(alphabet_range)
    
    fig, ax = plt.subplots()

    colors = [  'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    for i in indices[2000:]: 

        x = chars_embedded[i- (a_start[alphabet_index]*20),0]
        y = chars_embedded[i -(a_start[alphabet_index]*20),1]
        image = resized_chars[i,:].reshape(image_size, image_size)
        
        ind = int(i/20)                            
        image = OffsetImage(image, cmap=colors[ ind%len(colors) ], zoom=0.4)

        # syntax to display images instead of points    
        ab = AnnotationBbox(image, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)
        ax.plot(x, y)

    plt.show()

def glimpse_stats():
    ''' to build the glimpse score graph, a special matrix was outputted at the forward() function of
    the model, for each batch. We then separate the negative values from the positive to understand 
    similarity score patterns, e.g. ARC seeks to discriminate based on evidence, not to look for similarities 
    '''

    path = './data/glimpse_stats/'
    glimpse_table = np.zeros(1)
    for numpy_file in glob.glob("%s*.npy" %path):
        array = np.load(numpy_file)

        if len(glimpse_table.shape) == 2 :
            glimpse_table = np.vstack( (glimpse_table, array) ) 
        else:
            glimpse_table = array

    neg_table = glimpse_table*(glimpse_table < 0)
    neg_table = np.mean( neg_table, axis = 0)
    pos_table = glimpse_table*(glimpse_table > 0)
    pos_table = np.mean( pos_table, axis =  0)

    plt.bar(range(16), pos_table, tick_label = list(range(16)), label = 'Avg Positive Score' )
    plt.bar(range(16), neg_table, tick_label = list(range(16)), label = 'Avg Negative Score' )
    plt.legend()
    plt.title('Average Similarity Score As Model Take Glimpses')
    plt.show()     
    