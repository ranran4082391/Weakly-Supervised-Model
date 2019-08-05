# Key ideas of Capsule
***Richer representation*** In classic convolutional networks, a `scalar value` represents the activation of a given `feature`. By contrast, a capsule outputs a **vector**. **The vector’s length represents the probability of a feature being present**. The vector’s orientation represents the various properties of the feature (such as pose, deformation, texture etc.).

***Dynamic routing*** The output of a capsule is preferentially sent to certain parents in the layer above based on how well the capsule’s prediction agrees with that of a parent. Such dynamic “routing-by-agreement” generalizes the static routing of max-pooling.
