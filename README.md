# Color_pix2pix

This porject is based on [pix2pix](https://phillipi.github.io/pix2pix/) and the code is based on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow). The original pix2pix doesn't have the ability to choose the color. So I made a little reform 
in the original pix2pix network. In color_pix2pix, the network is breaked into 2 parts. The first part generates the gray images according to the edges pictures. The second part is aimed at coloring the gray pictures according to a rough color distribution.<br>
![Image](src)
## Requirement
python 3.5<br>
tesorflow 1.0<br>
opencv3 <br>
PIL<br>
## Getting Stated
Download the data sets from the https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/.
<br>
Then extract the data to your local disk.
<br>
Run the data_prepare.py to generate the training data.
### Training the model
### Testing the model
