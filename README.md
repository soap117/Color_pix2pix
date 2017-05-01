# Color_pix2pix

This porject is based on [pix2pix](https://phillipi.github.io/pix2pix/) and the code is based on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow). The original pix2pix doesn't have the ability to choose the color. So I made a little reform 
in the original pix2pix network. In color_pix2pix, the network is breaked into 2 parts. The first part generates the gray images according to the edges pictures. The second part is aimed at coloring the gray pictures according to a rough color distribution.
In addition, the second part of net is trained in directly minimizing the difference between generated images and groud turth images because I found this approach is better than adversarial training in coloring the images.<br>
![Image](https://github.com/soap117/Color_pix2pix/blob/master/images/image.jpg)
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
Run the /data/data_pre.py to generate the training data.
<br>
Forexample:
```cmd
cd "the location of data"
python data_pre.py
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
The pre-trained model can be donwloaded from the logs file.
Download the logs.zip files and extract them.
2 models of handbags and shoes are uploaded into the logs file.
<br>
Forexample:
```cmd
python eval.py
Enter the name of save files: handbags
Enter the name of line picture: handbag.png
Enter the name of color picture: handbag_c.png
```
## Generating your own data sets
The first is the same as in  [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow).
Then use the data_pre.py to generate the training set for color_pix2pix.
