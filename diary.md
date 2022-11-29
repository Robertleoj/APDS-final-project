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