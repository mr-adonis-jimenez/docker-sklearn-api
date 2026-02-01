"""Train and serialize scikit-learn model for serving."""
import os
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def create_synthetic_data(n_samples=1000, n_features=4, n_classes=3, random_state=42):
    """Generate synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        n_classes=n_classes,
        random_state=random_state,
        shuffle=True,
        class_sep=1.0
    )
    return X, y


def train_model(X_train, y_train):
    """Train logistic regression model."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*60 + "\n")
    
    return accuracy


def save_model(model, model_path):
    """Save trained model to disk."""
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    file_size = os.path.getsize(model_path)
    print(f"✓ Model file size: {file_size / 1024:.2f} KB")


def main():
    """Main training pipeline."""
    print("\n🚀 SCIKIT-LEARN MODEL TRAINING PIPELINE\n")
    
    # Configuration
    MODEL_PATH = Path("models/model.joblib")
    N_SAMPLES = 1000
    N_FEATURES = 4
    N_CLASSES = 3
    TEST_SIZE = 0.2
    
    # Generate data
    print(f"📊 Generating synthetic data...")
    print(f"   - Samples: {N_SAMPLES}")
    print(f"   - Features: {N_FEATURES}")
    print(f"   - Classes: {N_CLASSES}")
    X, y = create_synthetic_data(N_SAMPLES, N_FEATURES, N_CLASSES)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"\n✓ Data split:")
    print(f"   - Train: {len(X_train)} samples")
    print(f"   - Test: {len(X_test)} samples")
    
    # Train model
    print("\n🔧 Training logistic regression...")
    model = train_model(X_train, y_train)
    print("✓ Training complete")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save
    print("💾 Saving model...")
    save_model(model, MODEL_PATH)
    
    # Info
    print("\n📋 Model Information:")
    print(f"   - Type: {type(model).__name__}")
    print(f"   - Features: {model.n_features_in_}")
    print(f"   - Classes: {model.classes_.tolist()}")
    
    print("\n✅ Training pipeline completed!\n")


if __name__ == "__main__":
    main()
