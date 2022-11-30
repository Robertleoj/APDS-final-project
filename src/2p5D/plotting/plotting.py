import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider


def make_seg_img(seg):
    img_tensor = torch.zeros(seg.size(0), seg.size(1), 3)
    for i in range(3):
        img_tensor[:,:, i] += seg == (i + 1)
    return img_tensor.numpy()



def update_img(idx, im, seg, fig, imgs_t, segmentations):
    im.set(data=imgs_t[idx].unsqueeze(0).permute(1, 2, 0).numpy())
    seg_img = make_seg_img(segmentations[idx])
    seg.set(data=seg_img)
    fig.canvas.draw_idle()



def make_visual(imgs_t, segmentations):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.25)
    idx0 = 0

    # imgs_t = imgs_t.permute(2, 0, 1)#.squeeze(0)
    # segmentations = segmentations.permute(2, 0, 1)

    im = plt.imshow(imgs_t[idx0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap='gray')
    seg_img = make_seg_img(segmentations[idx0])
    seg = plt.imshow(seg_img, cmap='jet', alpha=0.25)

    allowed_values = range(imgs_t.size(0))

    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])

    slice_slider = Slider(
        ax=ax_slice, 
        label="Slice", 
        valmin=0,
        valmax=imgs_t.size(0) -1,
        valinit=idx0,
        valstep=allowed_values,
        color="blue"
    )
    slice_slider.on_changed(lambda x: update_img(x, im, seg, fig, imgs_t, segmentations))
    plt.show()
    return slice_slider

