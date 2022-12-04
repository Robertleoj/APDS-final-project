file size ranges from ... to 1 gb

* Deal with the differing z-axes
    * We need to investigate whether the scale changes with differing z-values

# Ideas for dealing with inconsistent z-size
1. Not care
    * Cons 
        * definitely harder to learn because of different
    * Pros
        * less work and complexity
        * Easy to try out

2. Use 2d slices
    * Cons
        * Lose a lot of information
    * Pro
        * Easy to do

3. 2.5D
We slice up the CT scan along the z-axis. The slices are constant depth, like 10. Then run it as a 2D image through a UNet, using the z-axis as the feature dimension. 

Unet works as usual, with one difference. At the end, we need to output a tensor with dimension $H \times W \times D \times C$. This can be done by projecting to $H \times W \times 3D$, and then slicing the $D$ dimension into three parts, adding that as an axis.

We can then finally concatenate the predicted masks.

If scan depth is not evenly divisible by $D$, we ignore the remainder at the bottom of the scan.


#Stuff to do next:
    Datasets, Dataloaders, Model 


# Model

## 2.5D Unet

Unet, ResNet blocks as a backbone, attention after every Resnet block

# nov 30 next steps
Remake cache
Incomplete layer:
* attn layers
* bottom
* maybe more

Rebalance dataset sampling

## Possible modifications
check out [this paper](https://www.sciencedirect.com/science/article/pii/S0925231222000650) that is doing 2.5D segmentation. They use spatial attention, which we are not using. They also do some more tricks, so check it out. 

The representation of the model is likely too small to be able to obtain class information about all the classes for each pixel. The last backbone block has a size that is three times as small as the desired output. You need to let the net have a larger representation.


Got rid of resizing due to scans in which the liver takes up the whole image 

New Approach: for each slice, concatenate the information of the neighboring slices to the current slice and then predict the segmentation of the middle slice


## 2022-12-03:
Try using two neighboring slices on each side instead of just one

Also add spatial attention instead or along with channel attention. 

