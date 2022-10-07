VIDEO FEATURE EXTRACTION - R3D 
The dataset is available at https://www.crcv.ucf.edu/projects/real-world/. 
It can be downloaded in parts from the Dropbox link listed at that page.
The downloaded files must be unzipped and the features should be extracted.

The feature extraction process can be carried out using a pretrained model
available from GluonCV (https://cv.gluon.ai), or using the pre-trained PyTorch 
models at https://pytorch.org/vision/stable/models.html.

The I3D, R3D and various other video feature extraction methods are available,
but the one used in this work is the R3D feature extractor pretrained on the
Kinetics-400 dataset. 

The 'number of segments' parameter must be chosen as 32 to split the video
into 32 segments. These features are available in the repo for convenience.
