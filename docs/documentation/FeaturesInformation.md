The features used for this work are 32x2048 in size, as set by the pre-trained 
network dimensions. 

To allow for convenient feature concatenation, other features produced in this 
work are the same size.Here are some details on each feature type and how it was created:


RGB:
The RGB frames from each video are extracted and passed to the R3D feature extraction model. 
This outputs a 32x2048 feature for each video, regardless of the video length.

FLOW:
The flow frames can be extracted along with the RGB frames using the mmaction 
framework. As this is very computationally costly, older Flow features of size
32x1024 are used. To overcome the asymmetry in the features when concatenating,
the Flow feature can be concatenated along axis 1 to produce a 32x2048 feature.
It is more ideal to just reproduce the flow features from scratch, but within the
constrained budget of this work it was not possible. Testing showed this did not
excessively affect training times or accuracy.
