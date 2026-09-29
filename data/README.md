# CT Data

Place your testing `.nii.gz` files in this directory.

Required files for full prototype:
1. `ct.nii.gz` - The raw CT volume
2. `liver.nii.gz` - Binary mask of the liver (1=liver, 0=background)
3. `lesion.nii.gz` - Binary mask of the lesion/HCC (1=lesion, 0=background)

Note: The prototype requires these to be registered (i.e. have the same physical space/affine).
