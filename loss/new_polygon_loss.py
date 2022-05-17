import torch
import numpy as np
import cv2
from scipy.spatial import ConvexHull
import time


def get_poly(starting_points=20):
    test = torch.rand([starting_points, 2], requires_grad=False)
    test = get_hull(test)
    return test


def get_hull(points, indices=False):
    hull = ConvexHull(points.clone().detach()).vertices.astype(int)

    if indices:
        return hull

    points = points[hull, :]
    return points


def poly_area(polygon):
    """
    Returns the area of the polygon
    polygon - [n_vertices,2] tensor of clockwise points
    """
    x1 = polygon[:, 0]
    y1 = polygon[:, 1]

    x2 = x1.roll(1)
    y2 = y1.roll(1)

    # per this formula: http://www.mathwords.com/a/area_convex_polygon.htm
    area = -1 / 2.0 * (torch.sum(x1 * y2) - torch.sum(x2 * y1))

    return area


def plot_poly(poly, color=(0, 0, 255), im=None, lines=True, show=True, text=None):
    if im is None:
        s = 1000
        im = np.zeros([s, s, 3]) + 255
    else:
        s = im.shape[0]

    if len(poly) > 0:
        poly = poly * s / 2.0 + s / 4.0

    for p_idx, point in enumerate(poly):
        point = point.int()
        im = cv2.circle(im, (point[0], point[1]), 3, color, -1)

    if lines:
        for i in range(-1, len(poly) - 1):
            p1 = poly[i].int()
            p2 = poly[i + 1].int()
            im = cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), color, 1)

    if text is not None:
        im = cv2.putText(im, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    im = cv2.circle(im, (646, 653), 10, color, -1)
    if show:
        cv2.imshow("im", im)
        cv2.waitKey(10)
        cv2.destroyAllWindows()
    return im


def plot_adjacency(points, adjacency, color=(0, 0, 255), im=None):
    if im is None:
        s = 1000
        im = np.zeros([s, s, 3]) + 255
    else:
        s = im.shape[0]

    if len(points) > 0:
        points = points * s / 2.0 + s / 4.0

    for i in range(adjacency.shape[0]):
        p1 = points[i, :].int()

        for j in range(adjacency.shape[1]):
            if adjacency[i, j] == 1:
                p2 = points[j, :].int()
                im = cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), color, 2)
        im = cv2.putText(im, str(i), (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    cv2.imshow("im", im)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return im


def ccw(A, B, C):
    result = (C[:, :, 1] - A[:, :, 1]) * (B[:, :, 0] - A[:, :, 0]) - (B[:, :, 1] - A[:, :, 1]) * (
                C[:, :, 0] - A[:, :, 0])
    result[result > 1e-5] = 1
    result[result < -1e-5] = -1
    result[torch.abs(result) < 1e-5] = 0
    return result


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    o1 = ccw(A, C, D)
    o2 = ccw(B, C, D)
    o3 = ccw(A, B, C)
    o4 = ccw(A, B, D)
    result = torch.logical_and(torch.ne(o1, o2), torch.ne(o3, o4))
    return result


def do_something(poly1, poly2):
    """
    Calculate the intersection over union between two convex polygons
    poly1,poly2 - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    """
    poly1 = clockify(poly1)
    poly2 = clockify(poly2)
    mid = (torch.mean(poly1, dim=0) + torch.mean(poly2, dim=0)) / 2.0
    # print(poly1)
    # print(poly2)

    im = np.zeros([1000, 1000, 3]) + 255
    im = plot_poly(poly1, color=(0, 0, 255), im=im, show=False)
    im = plot_poly(poly2, color=(255, 0, 0), im=im, show=False)
    plot_poly([], im=im, show=False)

    poly1r = poly1.roll(3, 0)
    poly2r = poly2.roll(3, 0)
    C0 = torch.cat((poly1, poly2), dim=0)
    D0 = torch.cat((poly1r, poly2r), dim=0)

    # tensors such that elementwise each index corresponds to the interstection of a poly1 line and poly2 line
    xy1 = poly1.unsqueeze(1).expand([-1, poly2.shape[0], -1])
    xy3 = poly2.unsqueeze(0).expand([poly1.shape[0], -1, -1])

    # same data, but rolled to next element
    xy2 = poly1.roll(1, 0).unsqueeze(1).expand([-1, poly2.shape[0], -1])
    xy4 = poly2.roll(1, 0).unsqueeze(0).expand([poly1.shape[0], -1, -1])

    x1 = xy1[:, :, 0]
    y1 = xy1[:, :, 1]
    x2 = xy2[:, :, 0]
    y2 = xy2[:, :, 1]
    x3 = xy3[:, :, 0]
    y3 = xy3[:, :, 1]
    x4 = xy4[:, :, 0]
    y4 = xy4[:, :, 1]

    # Nx and Ny contain x and y intersection coordinates for each pair of line segments
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    Nx = ((x1 * y2 - x2 * y1) * (x3 - x4) - (x3 * y4 - x4 * y3) * (x1 - x2)) / D
    Ny = ((x1 * y2 - x2 * y1) * (y3 - y4) - (x3 * y4 - x4 * y3) * (y1 - y2)) / D

    # get points that intersect in valid range (Nx should be greater than exactly one of x1,x2 and exactly one of x3,x4)
    s1 = torch.sign(Nx - x1)
    s2 = torch.sign(Nx - x2)
    s12 = (s1 * s2 - 1) / -2.0
    s3 = torch.sign(Nx - x3)
    s4 = torch.sign(Nx - x4)
    s34 = (s3 * s4 - 1) / -2.0
    s_total = s12 * s34  # 1 if line segments intersect, 0 otherwise
    keep = torch.nonzero(s_total)

    keep = keep.detach()
    Nx = Nx[keep[:, 0], keep[:, 1]]
    Ny = Ny[keep[:, 0], keep[:, 1]]
    intersections = torch.stack([Nx, Ny], dim=1)
    # plot_poly(intersections, color=(0, 255, 0), im=im, lines=False, show=True)

    insec_vector = intersections - mid

    union = torch.cat((poly1, poly2, intersections), dim=0)
    mid = torch.Tensor([0.7356, 0.7897])
    # mid = torch.mean(intersections, dim=0)
    # im = cv2.circle(im, (616, 643), 10, color, -1)
    mid = mid.repeat(union.shape[0], 1)

    A1 = mid.unsqueeze(1).expand([-1, C0.shape[0], -1])
    C1 = C0.unsqueeze(0).expand([mid.shape[0], -1, -1])

    # same data, but rolled to next element
    B1 = union.unsqueeze(1).expand([-1, C0.shape[0], -1])
    D1 = D0.unsqueeze(0).expand([mid.shape[0], -1, -1])

    intersectnums = intersect(A1, B1, C1, D1)
    intersectnums = torch.sum(intersectnums, dim=1)
    # print(intersectnums.shape)
    # print(intersectnums)
    polyi = union[intersectnums <= 2, :]
    polyi = clockify(polyi)

    a1 = poly_area(poly1)
    a2 = poly_area(poly2)
    ai = poly_area(polyi)

    # print("Poly 1 area: {}".format(a1))
    # print("Poly 2 area: {}".format(a2))
    # print("Intersection area: {}".format(ai))
    iou = ai / (a1 + a2 - ai + 1e-10)

    # plot_poly(polyi, color=(0, 0, 0), im=im, lines=True, text="Polygon IOU: {}".format(iou))

    return iou


def clockify(polygon, clockwise=True):
    """
    polygon - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    clockwise - if True, clockwise, otherwise counterclockwise
    returns - [n_vertices,2] tensor of sorted coordinates
    """

    # get center
    center = torch.mean(polygon, dim=0)

    # get angle to each point from center
    diff = polygon - center.unsqueeze(0).expand([polygon.shape[0], 2])
    tan = torch.atan(diff[:, 1] / diff[:, 0])
    direction = (torch.sign(diff[:, 0]) - 1) / 2.0 * -np.pi

    angle = tan + direction
    sorted_idxs = torch.argsort(angle)

    if not clockwise:
        sorted_idxs.reverse()

    polygon = polygon[sorted_idxs.detach(), :]
    return polygon


poly1 = get_poly(starting_points=4) * torch.rand(1) + torch.rand(1)
poly1 = [[0.9077, 0.8054],
         [0.8052, 0.9274],
         [0.7305, 0.9237],
         [0.7336, 0.7877]]
poly1 = torch.Tensor(poly1)
print(poly1.shape)

poly2 = get_poly(starting_points=4)
poly2 = [[0.8070, 0.4038],
         [0.9020, 0.9710],
         [0.8386, 0.9664],
         [0.5124, 0.6275]]
poly2 = torch.Tensor(poly2)
print(poly2.shape)

poly = torch.autograd.Variable(poly1, requires_grad=True)

opt = torch.optim.SGD([poly], lr=0.0001)

piou = do_something(poly, poly2)

for k in range(1000):
    # riou = raster_poly_IOU(poly1,poly2,scale = 1000)
    # print("Raster IOU: {}".format(riou))

    piou = do_something(poly, poly2)
    opt.zero_grad()
    lp = (1.0 - piou)
    l2 = torch.pow((poly - poly2), 2).mean()
    loss = l2 + lp
    loss.backward()
    print(piou)

    opt.step()
    del loss

print("success")
