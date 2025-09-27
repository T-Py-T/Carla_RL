#!/usr/bin/env python3
"""Simple script to create dummy artifacts without torch dependency"""

import os
import pickle


def create_dummy_model():
    """Create a simple dummy model file"""
    # Simple dummy model - just a dict with some parameters
    dummy_model = {
        'type': 'dummy_policy',
        'input_size': 5,
        'output_size': 3,
        'weights': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]
    }
    return dummy_model

def create_dummy_preprocessor():
    """Create a simple dummy preprocessor"""
    class DummyPreprocessor:
        def transform(self, X):
            # Simple identity transform
            return X

    return DummyPreprocessor()

def main():
    artifacts_dir = "artifacts/v0.1.0"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Create dummy model.pt (actually a pickle file)
    model_path = os.path.join(artifacts_dir, "model.pt")
    with open(model_path, 'wb') as f:
        pickle.dump(create_dummy_model(), f)
    print(f"Created dummy model: {model_path}")

    # Create preprocessor.pkl
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(create_dummy_preprocessor(), f)
    print(f"Created dummy preprocessor: {preprocessor_path}")

    print("Dummy artifacts created successfully!")

if __name__ == "__main__":
    main()
