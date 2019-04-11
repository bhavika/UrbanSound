frames = 41
bands = 60
feature_size = 2460  # 60x41
num_labels = 10
num_channels = 2

parent_dir = '../UrbanSound8K/audio/'
metadata_csv = '../UrbanSound8K/metadata/UrbanSound8K.csv'
audio = '../UrbanSound8K/audio/'

train_features_pickle = './output/train_features_full.npy'
train_labels_pickle = './output/train_labels_full.npy'

test_features_pickle = './output/test_features_full.npy'
test_labels_pickle = './output/test_labels_full.npy'

cnn_batch_size = 50
cnn_kernel_size = 30
cnn_depth = 20
cnn_num_hidden = 200

cnn_learning_rate = 0.01
cnn_total_iterations = 100

sbcnn_iterations = 200
sbcnn_batch = 32
field_size = 3

target_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
                'jackhammer', 'siren', 'street_music']


urbansound_dataset = "https://serv.cusp.nyu.edu/files/jsalamon/datasets/UrbanSound8K.tar.gz"