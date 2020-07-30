# Using Fully Convolutional Network for Image Classification

Typically CNN models used for image classification work with fixed size inputs.

Using such models in your application requires one to crop or resize the input images as per the dimensions imposed by the underlying CNN architecture.

However this approach leads to certain limitations as below which may cause significant degradation in model performance/accuracy:
- Loss of resolution
- Non-square aspect ratio
