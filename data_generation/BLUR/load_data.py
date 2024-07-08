import numpy as np

import os 
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    sh = imageio.imread(img0).shape

    sfx = ""

    if factor is not None and factor != 1:
        sfx = "_{}".format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, "images" + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, "does not exist, returning")
        return

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]

    if poses.shape[-1] != len(imgfiles):
        imagesfile = os.path.join(basedir, "sparse/0/images.bin")
        imdata = read_images_binary(imagesfile)
        imnames = [imdata[k].name[0:-4] for k in imdata]
        imgfiles = [
            os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f[0:-4] in imnames
        ]
        print(
            "{}: Mismatch between imgs {} and poses {} !!!!".format(
                basedir, len(imgfiles), poses.shape[-1]
            )
        )
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    def imread(f):
        if f.endswith("png"):
            return imageio.imread(f, ignoregamma=True)
            # return cv2.imread(f)#, ignoregamma=True)
        else:
            return imageio.imread(f)

    if not load_imgs:
        imgs = None
    else:
        imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
        imgs = np.stack(imgs, -1)
        print("Loaded image data", imgs.shape, poses[:, -1, 0])

    return poses, bds, imgs, imgfiles



def load_llff_data(
    basedir,
    factor=8,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    path_zflat=False,
    load_imgs=True,
):
    out = _load_data(
        basedir, factor=factor, load_imgs=load_imgs
    )  # factor=8 downsamples original imgs by 8x
    if out is None:
        return
    else:
        poses, bds, imgs, imgfiles = out

    # print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    if imgs is not None:
        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        images = imgs
        images = images.astype(np.float32)
    else:
        images = None
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        # print('recentered', c2w.shape)
        # print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = 0.8
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    # print('Data:')
    # print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    # print('HOLDOUT view is', i_test)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test, imgfiles

