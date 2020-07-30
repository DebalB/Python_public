# Fully Convolutional Networks and Receptive Fields

## Using Fully Convolutional Networks for Image Classification

Typically CNN models used for image classification work with fixed size inputs.

Using such models in your application requires one to crop or resize the input images as per the dimensions imposed by the underlying CNN architecture.

However this approach leads to certain limitations as below which may cause significant degradation in model performance/accuracy:
- Loss of resolution
- Non-square aspect ratio

In order to overcome above issues, Fully Convolutional Networks (FCNN) can be used.

By definition, FCNNs architectures comprise only Convolutional layers i.e. there are no Fully Connected/Linear layers in these networks.

Although it does add a few extra steps in extracting the predicted class, the benefits in accuracy far outweigh this slight inconvenience.

## Computing CNN Receptive Fields through backpropagation

Receptive field for a pixel in a feature map in a CNN represents all the pixels from the previous feature maps that affected its value.

It is a very useful tool for debugging CNNs and to understand what the network “saw” and analyzed to predict the final class.

And it can also be used to obtain a fairly accurate bounding box of the predicted object.

