import json
from keras.callbacks import ModelCheckpoint, CSVLogger
from nvidiamodel import NvidiaModel
from dataloader import DataLoader

def train():
    dl = DataLoader()
    train_data, val_data = dl.get_train_val(driving_data='data/driving_log.csv')

    model = NvidiaModel().model()
    model.compile(optimizer='adam', loss='mse')
    # save model architecture
    with open('logs/model.json', 'w') as f:
        f.write(json.dumps(json.loads(model.to_json()), indent=2))

    checkpoint = ModelCheckpoint('checkpoints/model.h5',
                                  monitor='val_loss',
                                  save_best_only=True)
    logger = CSVLogger(filename='logs/history.csv')

    # start the training
    print("training model...")
    model.fit_generator(generator=dl.generate_data_batch(train_data),
                        steps_per_epoch=300*dl.batch_size,
                        epochs=50,
                        validation_data=dl.generate_data_batch(val_data, 
                                                               augment_data=False, 
                                                               bias=1.0),
                        validation_steps=100*dl.batch_size,
                        callbacks=[checkpoint, logger])
    print("saving model...")
    model.save('model.h5')


if __name__ == "__main__":
    train()