import torch
import cv2
import matplotlib.pyplot as plt
from loss.convex_polygon_loss import *
import numpy as np

total_iter_poly_hist = []
total_iter_l1_hist = []
total_iter_l2_hist = []
PRINT = True
learning_rate = 0.2
variance = 150
setting = "lr2e-2v{}".format(variance)

for idx in range(1):
    init_mid = [[200, 360],
                [108, 57],
                [117, 277]]
    init_mid_np = np.concatenate((np.random.normal(200, variance / 3, size=(1, 2)), np.random.normal(0, variance, size=(2, 2))), axis=0)
    init_mid = torch.Tensor(init_mid)
    init_mid2 = [[200, 360],
                 [108, 57],
                 [117, 277]]
    init_mid2 = torch.Tensor(init_mid2)
    init_mid3 = [[200, 360],
                 [108, 57],
                 [117, 277]]
    init_mid3 = torch.Tensor(init_mid3)
    target_mid = [[288, 300],
                  [120, 160],
                  [190, 100]]
    target_mid_np = np.concatenate((np.random.normal(200, variance / 3, size=(1, 2)), np.random.normal(0, variance, size=(2, 2))), axis=0)
    target_mid = torch.Tensor(target_mid)
    # print(mid_2_points(init_mid))

    poly_init = torch.autograd.Variable(init_mid, requires_grad=True)
    poly_init2 = torch.autograd.Variable(init_mid2, requires_grad=True)
    poly_init3 = torch.autograd.Variable(init_mid3, requires_grad=True)
    poly_target = mid_2_points(target_mid)

    opt = torch.optim.Adam([poly_init], lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    opt2 = torch.optim.Adam([poly_init2], lr=learning_rate)
    # scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=30, gamma=0.1)
    opt3 = torch.optim.Adam([poly_init3], lr=learning_rate)
    # scheduler3 = torch.optim.lr_scheduler.StepLR(opt3, step_size=30, gamma=0.1)

    # piou = c_poly_loss(mid_2_points(poly_init), poly_target)
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # plt.ion()
    # plot1, = axes[0].plot(torch.cat((poly_target[:, 0], poly_target[0, 0].unsqueeze(0))), torch.cat((poly_target[:, 1], poly_target[0, 1].unsqueeze(0))))
    total_iter_poly = 0
    total_iter_l1 = 0
    total_iter_l2 = 0
    poly_hist = []
    l1_hist = []
    l2_hist = []
    for k in range(10000):
        stop_poly = False
        old_loss = 0
        if (total_iter_poly > 0 or stop_poly) and total_iter_l2 > 0 and total_iter_l1 > 0:
            break
        if total_iter_poly < 1 and not stop_poly:
            opt.zero_grad()
            poly_1 = mid_2_points(poly_init)
            # riou = raster_poly_IOU(poly_1, poly_target, scale=1000)
            piou = c_poly_loss(poly_1, poly_target)
            lp = (1.0 - piou)
            if lp < 0.01:
                total_iter_poly = k
            l2 = torch.pow((poly_1 - poly_target), 2).mean()
            loss = lp
            poly_hist.append(lp.detach().numpy())
            if abs(old_loss - loss) < 1e-4:
                stop_poly = True
            old_loss = loss
            loss.backward()
            if PRINT and k % 50 == 0:
                print("Polygon IOU: {}. lr:{}".format(piou.item(), opt.param_groups[0]['lr']))
                poly_1_detach = poly_1.detach()
                im = np.zeros([700, 500, 3]) + 255
                im = cv2.putText(im, "Polygon IOU: {}. lr:{}".format(piou.item(), opt.param_groups[0]['lr']), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                for i in range(-1, len(poly_1_detach) - 1):
                    p1 = poly_1_detach[i].int()
                    p2 = poly_1_detach[i + 1].int()
                    im = cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 1)
                for i in range(-1, len(poly_target) - 1):
                    p1 = poly_target[i].int()
                    p2 = poly_target[i + 1].int()
                    im = cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 255), 1)

            opt.step()
            # scheduler.step()
            # del loss

        if total_iter_l2 < 1:
            opt2.zero_grad()
            poly_2 = mid_2_points(poly_init2)
            if torch.abs((poly_2 - poly_target)).mean() < 1:
                total_iter_l2 = k
            l2_2 = torch.pow((poly_2 - poly_target), 2).mean().sqrt()
            piou = c_poly_loss(poly_2, poly_target)
            lp = (1.0 - piou)
            l2_hist.append(lp.detach().numpy())
            l2_2.backward()
            if PRINT and k % 50 == 0:
                print("l2 loss: {}".format(l2_2.item()))
                poly_2_detach = poly_2.detach()
                im2 = np.zeros([700, 500, 3]) + 255
                im2 = cv2.putText(im2, "L2 loss: {}. lr:{}".format(l2_2.item(), opt2.param_groups[0]['lr']), (10, 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                for i in range(-1, len(poly_2_detach) - 1):
                    p1 = poly_2_detach[i].int()
                    p2 = poly_2_detach[i + 1].int()
                    im2 = cv2.line(im2, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 1)
                for i in range(-1, len(poly_target) - 1):
                    p1 = poly_target[i].int()
                    p2 = poly_target[i + 1].int()
                    im2 = cv2.line(im2, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 255), 1)

            opt2.step()
            # scheduler2.step()
            # del l2_2

        if total_iter_l1 < 1:
            opt3.zero_grad()
            poly_3 = mid_2_points(poly_init3)
            if torch.abs((poly_3 - poly_target)).mean() < 1:
                total_iter_l1 = k
            l1_3 = torch.abs((poly_3 - poly_target)).mean()
            piou = c_poly_loss(poly_3, poly_target)
            lp = (1.0 - piou)
            l1_hist.append(lp.detach().numpy())
            l1_3.backward()
            if PRINT and k % 50 == 0:
                print("l1 loss: {}".format(l1_3.item()))
                poly_3_detach = poly_3.detach()
                im3 = np.zeros([700, 500, 3]) + 255
                im3 = cv2.putText(im3, "L1 loss: {}. lr:{}".format(l1_3.item(), opt3.param_groups[0]['lr']), (10, 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                for i in range(-1, len(poly_3_detach) - 1):
                    p1 = poly_3_detach[i].int()
                    p2 = poly_3_detach[i + 1].int()
                    im3 = cv2.line(im3, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 1)
                for i in range(-1, len(poly_target) - 1):
                    p1 = poly_target[i].int()
                    p2 = poly_target[i + 1].int()
                    im3 = cv2.line(im3, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 255), 1)

            opt3.step()
            # scheduler2.step()
            # del l2_2

        if PRINT and k % 50 == 0:
            Hori = np.concatenate((im, im2, im3), axis=1)
            cv2.imshow("im", Hori)
            cv2.waitKey(20)
        # if not PRINT and k > 10:
        #     cv2.destroyAllWindows()
    poly_hist = np.array(poly_hist)
    l1_hist = np.array(l1_hist)
    l2_hist = np.array(l2_hist)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.set_title("training with polygon iou loss")
    ax1.plot(poly_hist)
    ax2.set_title("training with l1 loss")
    ax2.plot(l1_hist)
    ax3.set_title("training with l2 loss")
    ax3.plot(l2_hist)
    plt.show()

    print("poly: {} | l1: {} | l2: {}".format(total_iter_poly, total_iter_l1, total_iter_l2))
    total_iter_poly_hist.append(total_iter_poly)
    total_iter_l1_hist.append(total_iter_l1)
    total_iter_l2_hist.append(total_iter_l2)
print("success")

total_iter_poly_hist = np.array(total_iter_poly_hist)
total_iter_l1_hist = np.array(total_iter_l1_hist)
total_iter_l2_hist = np.array(total_iter_l2_hist)
np.save("results/poly_hist_{}.npy".format(setting), total_iter_poly_hist)
np.save("results/l1_hist_{}.npy".format(setting), total_iter_l1_hist)
np.save("results/l2_hist_{}.npy".format(setting), total_iter_l2_hist)

print("poly_zeros: {}".format(np.count_nonzero(total_iter_poly_hist == 0)))
print("l1_zeros: {}".format(np.count_nonzero(total_iter_l1_hist == 0)))
print("l2_zeros: {}".format(np.count_nonzero(total_iter_l2_hist == 0)))

print("poly_mean: {}".format(total_iter_poly_hist[total_iter_poly_hist != 0].mean()))
print("l1_mean: {}".format(total_iter_l1_hist[total_iter_l1_hist != 0].mean()))
print("l2_mean: {}".format(total_iter_l2_hist[total_iter_l2_hist != 0].mean()))
