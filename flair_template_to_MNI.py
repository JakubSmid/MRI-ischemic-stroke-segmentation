import ants

template = ants.image_read("FLAIR-and-NCCT-atlas-for-elderly/derivatives/flair/brain/miplab-flair_sym_brain.nii.gz")
warp = ants.image_read("FLAIR-and-NCCT-atlas-for-elderly/derivatives/mni-deformations/composite-deformations/mni_to_miplab_sym_warp.nii.gz")

transform = ants.transform_from_displacement_field(warp)
template_mni = transform.apply_to_image(template, interpolation="linear")
ants.image_write(template_mni, "datasets/template_flair_mni.nii.gz")