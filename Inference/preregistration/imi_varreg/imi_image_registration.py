import argparse
import torch
import torch.nn.functional as F

from imi_image_warp import load_image_sitk, load_field_sitk, sitk_image_to_tensor, tensor_image_to_sitk, write_image_sitk
from imi_image_pyramid import build_pyramid, pyrup
from imi_regularized_warper import ImiRegularizedWarper
from imi_distance_ncc import NCC
from imi_deformation import identity_displacement_field


def imi_variational_registration(fixed_image, moving_image, initial_field, spacing=(1,1,1),
                                 alpha=1.0, tau=0.5, iterations=10,
                                 loss_type='demons', regularizer='diffusive', diffeomorphic=False,
                                 optimizer_type='sgd'):
    # Setup loss function
    if loss_type == 'demons':
        loss_func = torch.nn.MSELoss(reduction='mean')
        gradient_scaling_type = 'demons'
    elif loss_type == 'mse':
        loss_func = torch.nn.MSELoss(reduction='mean')
        gradient_scaling_type = 'normalize'
    elif loss_type == 'ncc':
        loss_func = NCC()
        gradient_scaling_type = 'spacing'
    else:
        raise NotImplementedError(f"Unknown loss type '{loss_type}', supported losses: 'demons'|'mse'|'ncc'.")
    # Setup Warper and Regularizer
    warper_regularizer = ImiRegularizedWarper(fixed_image.size(), alpha=alpha, tau=tau, spacing=spacing,
                                              regularizer=regularizer, gradient_scaling=gradient_scaling_type,
                                              diffeomorphic=diffeomorphic, use_spacing=True)
    # Setup displacement/velocity field to optimize
    displacement_field = initial_field
    if displacement_field is None:
        displacement_field = identity_displacement_field(fixed_image.size(), device=fixed_image.device,
                                                         dtype=fixed_image.dtype)
        displacement_field.requires_grad_(True)
    # Setup optimizer
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD([displacement_field], lr=1)
    elif optimizer_type == 'nesterov':
        optimizer = torch.optim.SGD([displacement_field], lr=1, momentum=0.1, dampening=0.0, nesterov=True)
    elif optimizer_type == 'adam':
        torch.optim.Adam([displacement_field], lr=1, betas=(0.1, 0.9999))
    else:
        raise NotImplementedError(f"Unknown optimizer type '{optimizer_type}', supported optimizers: 'sgd'|'nesterov'|'adam'.")

    # START REGISTRATION
    best_loss, best_field, best_iter = 0, None, 0
    for iter in range(iterations):
        warped_image, regularizer_loss = warper_regularizer(moving_image, displacement_field)
        distance_loss = loss_func(fixed_image, warped_image)
        total_loss = distance_loss + alpha * regularizer_loss
        print(f"Iter {iter}: loss = {total_loss} (dist={distance_loss}, regul={regularizer_loss}).")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if total_loss < best_loss or iter == 0:
            best_loss, best_field, best_iter = total_loss, displacement_field.detach().clone(), iter

    print(f"Best loss {best_loss} at iteration {best_iter}.")
    best_warped = warper_regularizer(moving_image, best_field)
    return best_warped, best_field, total_loss


def mse_loss(fixed, moving):
    return (fixed - moving).pow(2).mean()


def register_diffusive_2(fixed_image, moving_image, initial_field=None, iterations=50, spacing=(1, 1, 1)):
    tau = 0.005
    alpha = 1
    numpix = torch.prod(torch.tensor(fixed_image.size()))
    loss_func = torch.nn.MSELoss()
    ncc_func = NCC()
    warper_regularizer = ImiRegularizedWarper(fixed_image.size(), alpha=alpha, tau=tau, spacing=spacing)
    field = identity_displacement_field(fixed_image.shape, fixed_image.device) if initial_field is None else initial_field
    warped_image = moving_image
    best_loss = 1e10
    field.requires_grad_(True)
    #optimizer = torch.optim.SGD([field], lr=1, momentum=0.1, dampening=0.0, nesterov=True)
    optimizer = torch.optim.SGD([field], lr=1)
    #optimizer = torch.optim.Adam([field], lr=0.1, betas=(0.1, 0.01))
    for it in range(iterations):
        warped_image, regularizer_loss = warper_regularizer(moving_image, field)
        warped_image.grad = None
        field.grad = None
        dist_loss = F.mse_loss(fixed_image, warped_image, reduction='mean')  #loss_func(fixed_image, warped_image) / numpix
        #dist_loss = 2 * F.l1_loss(fixed_image, warped_image)
        #dist_loss = ncc_func.loss(fixed_image, warped_image)
        loss = dist_loss + alpha * regularizer_loss
        #jaco_det = compute_jacobian_det_3d(field, spacing=spacing).mean()
        #diff_loss = diffusive_loss_3d(field, spacing=spacing).mean()
        print(f"it {it}: loss={loss} ({dist_loss}, {regularizer_loss})" ) #,diff_loss={diff_loss},  jac={jaco_det}")
        if dist_loss < best_loss or it == 0:
            best_loss = dist_loss
            best_field = field.detach().clone()
        optimizer.zero_grad()
        (loss).backward()
        optimizer.step()
        # print((warped_image.grad - 2*(warped_image - fixed_image)).mean())
        #with torch.no_grad():
        #    field -= field.grad

    print(f'Best: {best_loss}')
    return warped_image.detach(), best_field.detach()


if __name__ == '__main__':
    # parse the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', '-f', required=True, type=str, help='path for input image')
    parser.add_argument('--moving', '-m', required=True, type=str, help='path for input image')
    parser.add_argument('--initial_field', '-init', type=str, help='path for displacement or velocity field')
    parser.add_argument('--iterations', '-i', type=int, default=8, help='path for displacement or velocity field')
    parser.add_argument('--levels', '-l', type=int, default=3, help='path for displacement or velocity field')
    parser.add_argument('--output', '-o', required=True, type=str, help='path for output image')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device')
    args = parser.parse_args()

    device = args.device
    levels = args.levels
    fixed_image, fixed_meta = sitk_image_to_tensor(load_image_sitk(args.fixed), return_meta_data=True)
    moving_image, moving_meta = sitk_image_to_tensor(load_image_sitk(args.moving), return_meta_data=True)
    fixed_image = fixed_image.to(device=device)
    moving_image = moving_image.to(device=device)
    initial_field = None
    if args.initial_field:
        initial_field = sitk_image_to_tensor(load_field_sitk(args.initial_field), return_meta_data=False)
        initial_field = initial_field.to(device=device)

    #diff, field = register_gaussian(fixed_image[:, :, :-1, :, :], moving_image[:, :, :-1, :, :], initial_field, iterations=args.iterations)

    #write_image_sitk(tensor_image_to_sitk(diff, moving_meta), args.output)

    fixed_pyramid = build_pyramid(fixed_image[:, :, :-1, :, :], max_level=levels)
    moving_pyramid = build_pyramid(moving_image[:, :, :-1, :, :], max_level=levels)
    init_field_pyramid = build_pyramid(initial_field[:, :, :-1, :, :], max_level=levels) if initial_field is not None else None
    init_field = init_field_pyramid[-1] if init_field_pyramid is not None else None
    spacing = (1, 1, 1)
    for level in reversed(range(len(fixed_pyramid))):
        print(f"PROCESSING LEVEL {level}:")
        s = [s * 2**level for s in spacing]
        print(s)
        fix = fixed_pyramid[level]
        mov = moving_pyramid[level]
        if init_field is not None:
            print(fix.shape, mov.shape, init_field.shape)
        #warp, field = register_gaussian(fix, mov, init_field, iterations=args.iterations, spacing=s)
        warp, field = register_diffusive_2(fix, mov, init_field, iterations=args.iterations, spacing=s)
        #warp2, field2 = register_diffusive_2(fix, mov, init_field, iterations=args.iterations, spacing=s)
        #idiff, fdiff = warp - warp2, field - field2
        #write_image_sitk(tensor_image_to_sitk(idiff, fixed_meta), f"diffi{level}.nii.gz")
        #write_image_sitk(tensor_image_to_sitk(fdiff, fixed_meta), f"difff{level}.nii.gz")
        #warp = warp_image_with_torch(mov, field)
        warpB = warp_image_with_displacement_field(mov, field, spacing=s)
        init_field = pyrup(field) if level > 0 else None
        tmp_field = field
        tmp_warp = warp
        for _ in range(level):
            tmp_field = pyrup(tmp_field)
            tmp_warp = pyrup(tmp_warp)

        write_image_sitk(tensor_image_to_sitk(tmp_warp, fixed_meta), f"warpA{level}.nii.gz")
        tmp_warp = warp = warp_image_with_displacement_field(mov, field, spacing=spacing)
        write_image_sitk(tensor_image_to_sitk(tmp_warp, fixed_meta), f"warp{level}.nii.gz")
        write_image_sitk(tensor_image_to_sitk(fix, fixed_meta), f"fpyr{level}.nii.gz")
        write_image_sitk(tensor_image_to_sitk(mov, fixed_meta), f"mpyr{level}.nii.gz")
        write_image_sitk(tensor_image_to_sitk(warp, fixed_meta), f"wpyr{level}.nii.gz")
        write_image_sitk(tensor_image_to_sitk(field, fixed_meta), f"dpyr{level}.nii.gz")
