import nilearn
import nilearn.image
import nilearn.plotting
import matplotlib.pyplot as plt
import nilearn.plotting.glass_brain

def save_glass_brain(image_file, title, output_file, title_size=30, fig_size=(12,8)):
    """
    Save a glass brain image with a title to a file.

    Parameters:
        image_file (str): The path to the NIfTI lesion map file.
        title (str): The title of the image.
        output_file (str): The path to the output file.
        title_size (int, optional): The size of the title. Defaults to 30.
        fig_size (tuple, optional): The size of the figure. Defaults to (12,8).
    """
    image = nilearn.image.load_img(image_file)
    fig = plt.figure(figsize=fig_size)

    display = nilearn.plotting.plot_glass_brain(image, figure=fig, colorbar=True,
                                      threshold=0, display_mode="ortho", cbar_tick_format="%i",
                                      radiological=True, cmap="inferno_r")
    
    # center title
    #title_len_inch = len(title) * title_size/2 / 72.272
    #title_pos = 0.5 - title_len_inch/fig_size[0]/2
    title_pos = 0.01
    
    display.title(title, color="black", bgcolor="white", x=title_pos, size=title_size)
    display.savefig(output_file)

if __name__ == "__main__":
    title_first_row = "Výskyt léze u daného počtu pacientů\n"
    save_glass_brain("results/stat_map_ISLES15.nii.gz", f"{title_first_row}Dataset ISLES 2015", "results/glass_brain_ISLES15.png")
    save_glass_brain("results/stat_map_Motol.nii.gz", f"{title_first_row}Dataset Motol", "results/glass_brain_Motol.png")
    save_glass_brain("results/stat_map_ISLES22.nii.gz", f"{title_first_row}Dataset ISLES 2022", "results/glass_brain_ISLES22.png")
    save_glass_brain("results/stat_map_ISLES22_ISLES15.nii.gz", f"{title_first_row}Sjednocení datasetu ISLES 2022 a ISLES 2015", "results/glass_brain_ISLES22_ISLES15.png")
    save_glass_brain("results/stat_map_ISLES22_ISLES15_Motol.nii.gz", f"{title_first_row}Sjednocení datasetu ISLES 2022, ISLES 2015 a Motol", "results/glass_brain_ISLES22_ISLES15_Motol.png")
