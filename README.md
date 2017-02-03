# A Test Drive of Convnet

This project predicts the steering angle of a virtual car using the image from the front camera.
Use `model.py` to learn from human behaviour collected from the game ([macOS][11], [Windows][12], [Linux][13]), and use `drive.py` to let the model drive in real time.

## Model architecture
My model was inspired by [the NVIDIA paper][1], in which a series of three strided and two non-strided convolutional layers reduce the image into 1x18 (height \* width), followed by three fully-connected layers.
I designed a "mini-NVIDIA model", which has fewer layers and only 113,141 (instead of 251,749) parameters, and is therefore easy to train without too much data.

My model consists of a normalization layer, three convolutional layers, and two fully-connected layers.
It expects an input image of 38x160 pixels.
The normalization layer maps the input values from [0, 255] to [-1, 1].
The first two convolutional layers have 5x5 kernels and 2x2 strides.
After the first convolutional layer, max pooling of size 2x2 is applied to reduce the image shape to 8x39.
The second convolutional layer further reduces the shape to 2x18.
Finally, the third convolutional layer (non-strided 2x2 kernel) outputs 1x17 pixels, resulting in a single row of features, because the prediction is only done on the horizontal axis.
The three convolutional layers have 24, 36, and 48 filters (dimensions per pixel), respectively.
The flattened output from the last convolutional layer is passed onto two consecutive fully-connected layers of size 100 and 10, before reaching the final output of size one (steering prediction).
Using two fully-connected layers instead of one encourages the hierarchical processing of information.

The Keras layers (from input to output, excluding activation and dropout) are summarized in the following table:

| Layer type | Output shape | Param # |
| ---------- |:------------:| -------:|
| Lambda        | 38, 160, 3 | 0      |
| Convolution2D | 17, 78, 24 | 1824   |
| MaxPooling2D  | 8, 39, 24  | 0      |
| Convolution2D | 2, 18, 36  | 21636  |
| Convolution2D | 1, 17, 48  | 6960   |
| Flatten       | 816        | 0      |
| Dense         | 100        | 81700  |
| Dense         | 10         | 1010   |
| Dense         | 1          | 11     |

To encourage generalization and prevent overfitting, a dropout of 0.2 is applied after each convolutional layer, and a dropout of 0.5 is applied after the first fully-connected layer.
I do not use regularization for this purpose, because I found it not beneficial in a previous project.
Following [the commaai code][2], I use ELU (exponential linear unit) instead of ReLU as the activation to avoid vanishing gradient and push the mean activation closer to zero.
I use the Adam optimizer to minimize the MSE.


## Data augmentation
Data augmentation refers to the automatic creation of new training examples from existing data.
Practically, I found it to be more influential on the test results than changes of the model architecture, because it enables the limited amount of data we collected to multiply.

At first, I passed the augmentation information and the image filenames together to the generator.
The generator generated batches of examples by loading the images and performing the designated augmentations on the fly, and shuffled the long list when it restarted from the beginning.
However, I observed from the training and validation loss that the model probably overfitted the data.
The car was not able to pass the first two turns, until I came up with the idea of letting the generator decide the augmentation on the fly.
Instead of choosing from numerous (filename, augmentation, steering value) tuples, the generator now feeds the model with infinite possible examples.

#### Images from side cameras*
This is technically not image augmentation, but it is done within the generator so that the probability of using images from the left or the right camera (instead of the centre camera) can be adjusted for different epochs.

The simulator records images from the two side cameras for data points of scenarios when the car is not in the centre of the lane.
These data for recovery could be otherwise rare, or dangerous to collect in real life.
I add 0.15 to the steering value if the left image is used; -0.15 if the right image is used.
I set the adjustment value empirically; the resulting steering is capped by [-1, 1].
For example, the three images (after cropping) of the same frame are shown below.
The steering values of the two side images are calculated from the centre one.
Cropping out the hood removes the clue about the actual position of the car, which influences judgements even for a human viewer.

![Images from the three cameras][21]

#### Horizontal shift*
An image can be randomly shifted in the horizontal direction to achieve a similar effect to the use of side cameras.
The steering value is increased by 0.005 for shifting every 1 pixel of distance to the right (on the unscaled image).
I experimented with different changes such as increasing 0.002 \* (1 + 5 \* |steering|) per pixel, but they all resulted in indistinguishable performance.

#### Rotation
I randomly rotate an image to increase the variety of inputs.
Although an inclined road surface is typically associated with turns, I do not change the steering value for rotations.

Both horizontal shift and rotation are realized as affine transformations.
However, if both operations are to be applied, I combine them into a single perspective transformation, instead of rotation followed by shift.
This minimizes the loss of pixels due to the cropping of intermediate images.
For example, compare the black edges of the lower right corners in the following two images (transformations exaggerated for illustration).

![Two consecutive transformations vs. one single transformation][22]

#### Vertical shift
The vertical shift is implemented in the cropping process and simulates bumps.
Because large bumps are uncommon, I sample the shift value from a normal distribution.

#### Flipping
I flip any image horizontally with 50% probability, and flip the sign of its steering value.
This ensures the balance of left and right steering.
Flipping creates highly truthful examples because roads are mostly symmetrical (except for traffic signs, etc.).

#### Random brightness
After converting an image to the YUV colour space, I randomly scale the luminance (Y) by a factor of 0.5–1.25.
The maximum (normally 1.25) is bound by the largest Y value in that image, such that the resulting image does not contain values greater than 255.
This is different from my previous works, where I dynamically normalized all colour channels according to their means and standard deviations in each image, instead of using them for augmentation.

## Training
I trained the model using [the sample data][3] and the data I collected by running the simulator.
Probably because I do not have a joystick and have to control the car using the mouse, or because I am not good at gaming, I discovered that using more self-collected data makes the driving brutal and prone to fail.
Therefore, I weight the sample data by 2 (i.e. for an image, the probability of being chosen is doubled; should be changed into assigning the sum of probability to each dataset).

My current model was trained using the default parameters (256 examples per batch, 64 batches per epoch, 20 epochs).
The learning script is capable of loading a trained model for fine-tuning, but I did not use it for the current model.

#### Balanced sampling of data
I have balanced the left and right steering by flipping.
However, most steering angles in the training data are small, while it is crucial to predict the large angles correctly in order to pass the bends.
I classify the steering values into 7 categories, and in the sample dataset, the largest group (4,361 / 8,036 = 54%) is consisted of frames of which the steering value is exactly 0:

![Distribution of steering values in sample data][23]

Similarly in my dataset, 14,896 out of the 25,996 frames (57%) have zero steering, and the values are more concentrated due to the limited number of steering angles the mouse can produce:

![Distribution of steering values in my data][24]

Therefore, I want to increase the sampling probability of the non-zero-steering frames, and particularly the large-steering frames.
This can be done by multiplying the sampling probability of a frame with the ratio of sample size between the largest group and the current group (henceforth the *compensation ratio*), ensuring that each batch is almost evenly divided by the 7 categories.

Then I came up with the more sophisticated idea of adjusting the proportion of the 7 categories for different epochs.
I tested two opposite hypotheses:
1. The learning should begin with a relatively "natural" dataset, and the more challenging and rarer examples should be reserved until the model acquires the "norm".
2. The large-steering frames are more characteristic of the task and more selective of the model than the close-to-zero frames, so a model fitted to these frames can be fine-tuned for the latter, but not the other way round.

My experiments showed that in this case Hypothesis 2 led to better models.
In this version, I use the squared compensation ratio in the first epoch (strongly favouring the large-steering frames), and lower the exponent after each epoch, until the exponent becomes 0.5 in the final epoch.
In contrast, I control the probability of the two starred augmentations (whereas the others have constant probabilities) following Hypothesis 1, gradually increasing their probability from 0 to 0.5.
This is because these two augmentations involve unsophisticated modifications of the steering value which I am not confident about (not as confident as about the original data), and they mainly help with restoring the car to the centre of the lane.

#### Preprocessing
I preprocess the training data by shifting the list of steering values one frame ahead of their corresponding images.
This is to compensate for the human reaction time (though a well-trained player may take this into account and react before the real stimuli occur).
As the model does not learn from the time sequence, learning from a delayed response can be disastrous.
The data was recorded at an approximate frame rate of 10fps (sample data) or 15fps (my data), so one frame is probably an underestimate. (As a poor player I sometimes find in the recording that I should have turned more than five frames earlier! But the delay is not consistent.)
By reading the time of each frame from the image filename, my script is able to detect gaps in the recording.
For each continuous clip, the first steering value and the last image are not usable, as a result of the shifting.

At all times, I always crop out the sky and the hood from the input image, leaving only the height range of [59, 135).
The 76x320 image is resized by half to 38x160.
I convert from the input RGB or BGR into the YUV colour space for a independent grayscale channel and better correspondence with human perception.
I initially experimented with combinations of of colour channels such as YUV + edge detection (not trainable), RGB + edge detection, and RGB + YUV (random brightness disabled), but settled with YUV only.

## Results and discussion
[![Play the video][25]][4]

The autonomous car finishes Track 1 smoothly.
It also runs through most of Track 2, which it has never seen in any training data, and is significantly different from Track 1. (I did not collect training data from Track 2 because I kept popping onto ledges.)
Presumably, in Track 2 it suffers from the aggressive cropping of the input image, which would not work well for uphill, downhill, or bumpy roads.

I also made a small change to the driving script to use two different throttle values when the speed is low, so that the car is able to climb the uphill roads of Track 2.
For generality, I think an improved model should take multiple inputs (e.g. camera image, current steering angle and speed) and output multiple values (e.g. throttle and steering angle).
In Keras, such a model can be built via the functional API, instead the `Sequential` model.
However, the speed value is highly unevenly distributed in the sample training data, and also in any self-collected data because the throttle is binary in the simulator.
For this reason, I tried to circumvent the lack of data by training the model to predict the product of steering angle and speed.
Intuitively, smaller steering angles should be used as the speed increases.
But such a model was not superior than the current model (even though the current model steers a car that is generally slower than the training drives), presumably because relationship is not as simple as such.

[1]: https://arxiv.org/abs/1604.07316
[2]: https://github.com/commaai/research/blob/master/train_steering_model.py
[3]: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
[4]: https://www.youtube.com/watch?v=MaEChUzNap0
[11]: https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587525b2_udacity-sdc-udacity-self-driving-car-simulator-dominique-default-mac-desktop-universal-5/udacity-sdc-udacity-self-driving-car-simulator-dominique-default-mac-desktop-universal-5.zip
[12]: https://d17h27t6h515a5.cloudfront.net/topher/2017/January/58752736_udacity-sdc-udacity-self-driving-car-simulator-dominique-default-windows-desktop-64-bit-4/udacity-sdc-udacity-self-driving-car-simulator-dominique-default-windows-desktop-64-bit-4.zip
[13]: https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587527cb_udacity-sdc-udacity-self-driving-car-simulator-dominique-development-linux-desktop-64-bit-5/udacity-sdc-udacity-self-driving-car-simulator-dominique-development-linux-desktop-64-bit-5.zip
[21]: https://cloud.githubusercontent.com/assets/6981180/22576183/12416d10-e9b2-11e6-8721-9de160f258ad.png
[22]: https://cloud.githubusercontent.com/assets/6981180/22576185/15490658-e9b2-11e6-8e8e-48fdd6eaaf1f.png
[23]: https://cloud.githubusercontent.com/assets/6981180/22576189/1893c08c-e9b2-11e6-8ef5-3b98d74a5571.png
[24]: https://cloud.githubusercontent.com/assets/6981180/22576191/1a37d496-e9b2-11e6-8e1e-89712b85651f.png
[25]: https://cloud.githubusercontent.com/assets/6981180/22576193/1bcddca6-e9b2-11e6-9667-52c6e3b1bc32.png
