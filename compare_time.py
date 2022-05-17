import torch
import cv2
import matplotlib.pyplot as plt
from loss.convex_polygon_loss import *

init_mid = [[200, 360],
         [108, 57],
         [117, 277]]
init_mid = torch.Tensor(init_mid)
target_mid = [[288, 300],
         [120, 160],
         [190, 100]]
target_mid = torch.Tensor(target_mid)

poly_init = torch.autograd.Variable(init_mid, requires_grad=True)
poly_target = mid_2_points(target_mid)

opt = torch.optim.Adam([poly_init], lr=0.35)

def l2_loss (poly_init, poly_target):
    return torch.pow((mid_2_points(poly_init) - poly_target), 2).mean()


for k in range(1000):
    opt.zero_grad()
    poly_1 = mid_2_points(poly_init)
    # loss = (1.0 - c_poly_loss(mid_2_points(poly_init), poly_target))
    loss = l2_loss(poly_init, poly_target)
    loss.backward()
    print("Polygon IOU: {}. lr:{}".format(loss.item(), opt.param_groups[0]['lr']))
    opt.step()

print("success")
