import torch
import numpy as np
import cv2
from scipy.spatial import ConvexHull


def raster_poly_IOU(poly1, poly2, scale=1000):
    poly1 = (poly1 * scale / 2.0 + scale / 4.0).detach().numpy().astype(np.int32)
    poly2 = (poly2 * scale / 2.0 + scale / 4.0).detach().numpy().astype(np.int32)

    im1 = np.zeros([scale, scale])
    im2 = np.zeros([scale, scale])
    imi = np.zeros([scale, scale])

    im1 = cv2.fillPoly(im1, [poly1], color=1)
    im2 = cv2.fillPoly(im2, [poly2], color=1)
    imi = (im1 + im2) / 2.0

    imi = np.floor(imi)

    ai = np.sum(imi)
    a1 = np.sum(im1)
    a2 = np.sum(im2)

    iou = ai / (a1 + a2 - ai)

    return iou


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

    if show:
        cv2.imshow("im", im)
        cv2.waitKey(500)
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
    result[result > 1e-7] = 1
    result[result < -1e-7] = -1
    result[torch.abs(result) < 1e-7] = 0
    return result


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    o1 = ccw(A, C, D)
    o2 = ccw(B, C, D)
    o3 = ccw(A, B, C)
    o4 = ccw(A, B, D)
    result = torch.logical_and(torch.ne(o1, o2), torch.ne(o3, o4))
    return result


def np_wn(pnts, poly, return_winding=False):
    """Return points in polygon using a winding number algorithm in numpy.

    Parameters
    ----------
    pnts : Nx2 array
        Points represented as an x,y array.
    poly : Nx2 array
        Polygon consisting of at least 4 points oriented in a clockwise manner.
    return_winding : boolean
        True, returns the winding number pattern for testing purposes.  Keep as
        False to avoid downstream errors.

    Returns
    -------
    The points within or on the boundary of the geometry.

    References
    ----------
    `<https://github.com/congma/polygon-inclusion/blob/master/
    polygon_inclusion.py>`_.  inspiration for this numpy version
    """
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T  # polygon `to` coordinates
    x, y = pnts.T  # point coordinates
    y_y0 = y[:, None] - y0
    x_x0 = x[:, None] - x0
    diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0  # diff => einsum in original
    chk1 = (y_y0 > 0.0)
    chk2 = np.less(y[:, None], y1)  # pnts[:, 1][:, None], poly[1:, 1])
    chk3 = np.sign(diff_).astype(int)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn = pos - neg
    out_ = pnts[np.nonzero(wn)]
    if return_winding:
        return out_, wn
    return out_


def torch_wn(pnts, poly, return_winding=False):
    # print(poly)
    x0, y0 = poly[:].T  # polygon `from` coordinates
    # print(x0)
    x1, y1 = torch.roll(poly, -1, 0).T  # polygon `to` coordinates
    # print(x1)
    x, y = pnts.T  # point coordinates
    y_y0 = y[:, None] - y0
    x_x0 = x[:, None] - x0
    diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0  # diff => einsum in original
    chk1 = (y_y0 > 0.0)
    chk2 = torch.less(y[:, None], y1)  # pnts[:, 1][:, None], poly[1:, 1])
    chk3 = torch.sign(diff_)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn = pos - neg
    out_ = pnts[torch.nonzero(wn)[:, 0]]
    if return_winding:
        return out_, wn
    return out_


def c_poly_loss(poly1, poly2):
    """
    Calculate the intersection over union between two convex polygons
    poly1,poly2 - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    """
    poly1 = clockify(poly1)
    poly2 = clockify(poly2)
    # print(poly1)
    # print(poly2)

    # im = np.zeros([1000, 1000, 3]) + 255
    # im = plot_poly(poly1, color=(0, 0, 255), im=im, show=False)
    # im = plot_poly(poly2, color=(255, 0, 0), im=im, show=False)
    # plot_poly([], im=im, show=False)

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
    # modified
    s_total[torch.abs(D) < 1e-9] = 0
    keep = torch.nonzero(s_total)

    keep = keep.detach()
    Nx = Nx[keep[:, 0], keep[:, 1]]
    Ny = Ny[keep[:, 0], keep[:, 1]]
    intersections = torch.stack([Nx, Ny], dim=1)
    # print(np_wn(poly1_np, poly2_np))
    poly1_np_keep = torch_wn(poly1, poly2)
    poly2_np_keep = torch_wn(poly2, poly1)
    # poly1_np_keep = torch.tensor(poly1_np_keep)
    # poly2_np_keep = torch.tensor(poly2_np_keep)

    # plot_poly(intersections, color=(0, 255, 0), im=im, lines=False, show=True)

    union = torch.cat((poly1_np_keep, poly2_np_keep, intersections), dim=0)
    # print(intersections)
    polyi = clockify(union)
    # print(polyi)

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


def mid_2_points(mids):
    mid = torch.unsqueeze(mids[0, :], 0)
    first_quadrant = torch.unsqueeze(mids[1, :], 0)
    second_quadrant = torch.unsqueeze(mids[2, :], 0)
    second_quadrant[0, 0].add(-mids[2, 0])
    return torch.cat((mid+first_quadrant, mid-second_quadrant, mid-first_quadrant, mid+second_quadrant), dim=0)
