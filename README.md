# Color_pix2pix

This porject is based on [pix2pix](https://phillipi.github.io/pix2pix/) and the code is based on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow). The original pix2pix doesn't have the ability to choose the color. So I made a little reform 
in the original pix2pix network. In color_pix2pix, the network is breaked into 2 parts. The first part generates the gray images according to the edges pictures. The second part is aimed at coloring the gray pictures according to a rough color distribution.<br>
![Image](https://github.com/soap117/Color_pix2pix/blob/master/images/shoe.png)
## Requirement
python 3.5<br>
tesorflow 1.0<br>
opencv3 <br>
PIL or pillow<br>
scipy<br>
## Getting Stated
Download the data sets from the https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/.
<br>
Then extract the data to your local disk.
<br>
Run the /data/data_prepare.py to generate the training data.
<br>
Forexample:
```cmd
cd "the location of data"
python data_prepare.py
Enter the original data location: E:\\edges2handbags\\train\\
Where you want to save the new data: E:\\color_pix2pix\\handbags\\
```
### Training the model
The models will be saved in ./logs. <br>
Run the /train_edage2gray.py to train the first part of model.
<br>
Forexample:
```cmd
python train_edage2gray.py
Enter the training data location: E:\\color_pix2pix\\handbags\\
Enter the name of save files: handbags
```
Run the train_edage2gray.py to train the second part of model.
<br>
Forexample:
```cmd
python train_gray2real.py
Enter the training data location: E:\\color_pix2pix\\handbags\\
Enter the name of save files: handbags
```
### Testing the model
Run the eval.py to test model.
<br>
Forexample:
```cmd
python eval.py
Enter the name of save files: handbags
Enter the name of line picture: handbag.png
Enter the name of color picture: handbag_c.png
```
