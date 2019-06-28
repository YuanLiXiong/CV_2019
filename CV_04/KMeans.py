import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
'''
KMeans alg:
0. Sure the number of classes for point clouds.
1. Initialize center of point clouds
2. Calculate the center on point clouds again.
4. Iterate the points until the distance weight of classes is convergent  or setting a iteration number.

other tasks: There are some other good methods to optimize KMean method.
'''

COLOR_NAMES = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

def gen_PC(pts_num=1000, cls=10, center_range=100):
    '''
    According to center points, generate each class of all point clouds.

    :param cls:  number for all class (center point)
    :param range: range of all center points
    :return: center points: [cls, 3]  point clouds [cls, cls_point_num, 3]
    '''

    center_pts = np.random.randint(-center_range, center_range, (cls, 3))
    pts_cloud = []

    for i, p in enumerate(center_pts):
        #  gaussian points: center                sigma(noise)        shape: 3 x pts_num
        pts = [d * np.random.randn(pts_num) + 20 * np.random.rand(pts_num) for d in p]
        pts_cloud.append(pts)

    pts_cloud = np.array(pts_cloud, dtype=np.int)
    pts_cloud = pts_cloud.reshape((pts_num, cls, 3))

    return pts_cloud, center_pts

def plot_points(points, center_pts=None):

    # pts_num, cls, dims = points.shape

    # color_names = np.random.choice(list(COLOR_NAMES.keys()), cls)
    # random_colors = [COLOR_NAMES[name] for name in color_names]
    # print(random_colors)

    fig = plt.figure()
    pts = points.reshape(-1, 3)
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]

    ax1 = Axes3D(fig)
    # ax.scatter(X, Y, Z, c=random_colors[i])
    ax1.scatter(X, Y, Z)
    if center_pts is not None:
        ax1.scatter(center_pts[:, 0], center_pts[:, 1], center_pts[:, 2], c="red")

    plt.show()

def aggressive_pts(pts, init_center=None, cls=2, iter=1000):
    nums, dim = pts.shape  # 100 x 3

    if init_center is not None:
        center_pts = init_center
    else:
        center_ids = np.random.choice(list(range(nums)), cls, replace=True)
        center_pts = pts[center_ids, :]
        print("random center points:")
        for p in center_pts:
            print(p)

    pts_props = np.zeros((nums, cls + 1))           # nums pts  x  (num cls dist, cls id)

    for i in range(iter):
        for c_id in range(cls):
            center_pt = np.expand_dims(center_pts[c_id], axis=0)  # nums x 3
            dist = np.sum(np.sqrt(np.power(pts - center_pt, 2)), axis=1)  # nums
            pts_props[:, c_id] = dist

        # update its class for each point
        min_inds = np.argmin(pts_props[:, :-1], axis=1)
        pts_props[:, -1] = min_inds

        # update center points by the min distance
        for c_id in range(cls):
            c_ids = np.where(pts_props[:, -1] == c_id)
            cls_pts = pts[c_ids, :]
            center_pts[c_id] = np.mean(cls_pts, axis=1)

    dst_pts = []
    for c_id in range(cls):
        dst_cls_ids = np.where(center_pts[:, -1] == c_id)
        dst_pts.append(pts[dst_cls_ids, :])

    print("KMeans:")
    print(center_pts)

    return center_pts, dst_pts

if __name__ == '__main__':
    data_PC, center_pts = gen_PC(pts_num=50)

    print('data_PC:', data_PC.shape, '  center_pts:', center_pts.shape)
    data = data_PC.reshape((-1, 3))
    # plot_points(data_PC, center_pts) # 一个列表的 id 对应着一个中心点的一个id
    center_pts, cls_pts = aggressive_pts(data, cls=3, iter=2)

    plot_points(data_PC, center_pts)

