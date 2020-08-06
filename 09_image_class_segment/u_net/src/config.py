# training csv file path
TRAINING_CSV = "../input/train_pneumothorax.csv"

# training and test batch sizes
TRAINING_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4

# number of epochs
EPOCHS = 10

# define the encoder for U-Net
# check: https://github.com/qubvel/segmentation_models.pytorch
# for all supported encoders
ENCODER = "resnet18"

# we use imagenet pretrained weights for the encoder
ENCODER_WEIGHTS = "imagenet"

# train on gpu
DEVICE = "cuda"
