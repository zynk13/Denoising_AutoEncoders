## CSE 676 - DEEP LEARNING
A 6 Layered Denoising Auto Encoder implemented to remove the background of an image.

Inspiration - https://www.kaggle.com/c/carvana-image-masking-challenge

## Execution
```
## Begin Training and save the model

$ python3 denoiser_train.py [--train_imdir TRAIN_DIRECTORY] [--train_maskdir TRAIN_MASK_DIRECTORY]

optional arguments:
    --train_imdir TRAIN_DIRECTORY           path to training images directory
    --train_maskdir TRAIN_MASK_DIRECTORY    path to training masks directory
    --target_size IMAGE_SIZE                input image size (default: (256, 256))
    --grayscale GRAYSCALE                   use grayscale (default: True)
    --batch_size BATCH_SIZE                 batch size (default: 2)
    

## Test existing model
$ python3 test.py
usage: python test.py

## Superimpose masks on resized training images
$ python3 image_super_imposer.py
```

## Dependencies
- Python 3
- numpy
- PIL
- Matplotlib
- keras = 2.0.3 [or above]
- tensorflow > 1.0.1 [or above]
