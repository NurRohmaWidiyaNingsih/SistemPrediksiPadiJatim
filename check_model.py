import pickle
import joblib
import numpy as np

# Coba load file model
try:
    model = joblib.load("model_svr_rbf_90_30_partikel_100_iterasi.save")
    print("✅ File loaded successfully dengan joblib")
    print(f"Type model: {type(model)}")
except Exception as e:
    print(f"❌ Joblib failed: {e}")
    try:
        with open("model_svr_rbf_90_30_partikel_100_iterasi.save", "rb") as f:
            model = pickle.load(f)
        print("✅ File loaded successfully dengan pickle")
        print(f"Type model: {type(model)}")
    except Exception as e2:
        print(f"❌ Pickle failed: {e2}")
        model = None

# Tampilkan informasi model
if model is not None:
    print("\n=== MODEL INFO ===")
    print(f"Model type: {type(model)}")
    
    if hasattr(model, 'n_features_in_'):
        print(f"Number of input features: {model.n_features_in_}")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature names: {model.feature_names_in_}")
    
    if hasattr(model, 'support_vectors_'):
        print(f"Number of support vectors: {len(model.support_vectors_)}")
    
    if hasattr(model, 'kernel'):
        print(f"Kernel: {model.kernel}")
    
    if hasattr(model, 'C'):
        print(f"C parameter: {model.C}")
    
    if hasattr(model, 'gamma'):
        print(f"Gamma: {model.gamma}")
    
    if hasattr(model, 'epsilon'):
        print(f"Epsilon: {model.epsilon}")
    
    # List semua attributes
    print("\n=== ALL ATTRIBUTES ===")
    for attr in sorted(dir(model)):
        if not attr.startswith('_'):
            print(f"  {attr}")
    
    # Coba predict
    print("\n=== TEST PREDICTION ===")
    try:
        if hasattr(model, 'n_features_in_'):
            test_input = np.random.rand(1, model.n_features_in_)
            pred = model.predict(test_input)
            print(f"✅ Test prediction successful: {pred}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
