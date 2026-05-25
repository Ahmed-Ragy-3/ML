from torch.utils.data import DataLoader

from dataset import UrbanSoundFeatureDataset
from model import EnvironmentalGRUClassifier
from trainer import ModelTrainer


def main():
    print("Loading datasets...")

    base_kwargs = {
        "root_dir": "UrbanSound8K",
        "processed_dir": "processed"
    }

    train_data = UrbanSoundFeatureDataset(folds=[1, 2, 3, 4, 5, 6], **base_kwargs)
    val_data   = UrbanSoundFeatureDataset(folds=[7, 8], **base_kwargs)
    test_data  = UrbanSoundFeatureDataset(folds=[9, 10], **base_kwargs)

    print("Preprocessing datasets (if needed)...")

    train_data.preprocess_all()
    val_data.preprocess_all()
    test_data.preprocess_all()

    print("Creating DataLoaders...")

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        collate_fn=UrbanSoundFeatureDataset.collate_batches
    )

    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        collate_fn=UrbanSoundFeatureDataset.collate_batches
    )

    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        collate_fn=UrbanSoundFeatureDataset.collate_batches
    )

    print("Initializing model and trainer...")

    model = EnvironmentalGRUClassifier()
    trainer = ModelTrainer(model, train_loader, val_loader, test_loader)

    print("Starting training...")
    trainer.train(num_epochs=15)

    print("Running final evaluation...")
    trainer.evaluate()


if __name__ == "__main__":
    main()