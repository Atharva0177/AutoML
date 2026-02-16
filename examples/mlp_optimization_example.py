"""
MLP Model Optimization Examples.

Demonstrates advanced optimization features:
- Learning rate scheduling
- Gradient clipping
- Weight decay (L2 regularization)
- Different optimizers
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from automl.models.deep_learning.mlp_models import MLPClassifier, MLPRegressor


def example_1_lr_scheduling():
    """Example 1: Learning rate scheduling with ReduceLROnPlateau."""
    print("\n" + "=" * 80)
    print("Example 1: Learning Rate Scheduling with ReduceLROnPlateau")
    print("=" * 80)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train with ReduceLROnPlateau scheduler
    model = MLPClassifier(
        hidden_layers=[128, 64, 32],
        learning_rate=0.01,  # Start with higher LR
        max_epochs=100,
        lr_scheduler='plateau',  # Will reduce LR when val_loss plateaus
        lr_patience=5,  # Wait 5 epochs before reducing
        lr_factor=0.5,  # Reduce LR by 50%
        early_stopping_patience=15,
        use_gpu=False,
        random_state=42
    )
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)
    
    # Check training history
    print(f"\nTraining completed in {len(model.history['train_loss'])} epochs")
    print(f"Best validation loss: {min(model.history['val_loss']):.4f}")
    print(f"Final training accuracy: {model.history['train_metric'][-1]:.4f}")
    print(f"Final validation accuracy: {model.history['val_metric'][-1]:.4f}")


def example_2_gradient_clipping():
    """Example 2: Gradient clipping for stable training."""
    print("\n" + "=" * 80)
    print("Example 2: Gradient Clipping for Training Stability")
    print("=" * 80)
    
    # Generate dataset
    result = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10.0,
        random_state=42
    )
    X = result[0]
    y = result[1]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train with gradient clipping
    model = MLPRegressor(
        hidden_layers=[128, 64],
        learning_rate=0.01,
        max_epochs=50,
        gradient_clip_value=1.0,  # Clip gradients to max norm of 1.0
        lr_scheduler='plateau',
        use_gpu=False,
        random_state=42
    )
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)
    
    print(f"\nTraining with gradient clipping (max_norm=1.0)")
    print(f"Final training RMSE: {model.history['train_metric'][-1]:.4f}")
    print(f"Final validation RMSE: {model.history['val_metric'][-1]:.4f}")


def example_3_weight_decay():
    """Example 3: L2 regularization with weight decay."""
    print("\n" + "=" * 80)
    print("Example 3: L2 Regularization with Weight Decay")
    print("=" * 80)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=500,
        n_features=50,
        n_informative=20,
        n_redundant=30,  # High redundancy - regularization helps
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train without weight decay
    model_no_wd = MLPClassifier(
        hidden_layers=[64, 32],
        learning_rate=0.001,
        max_epochs=50,
        weight_decay=0.0,  # No regularization
        lr_scheduler=None,
        use_gpu=False,
        random_state=42
    )
    
    model_no_wd.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    
    # Train with weight decay
    model_with_wd = MLPClassifier(
        hidden_layers=[64, 32],
        learning_rate=0.001,
        max_epochs=50,
        weight_decay=0.01,  # L2 regularization
        lr_scheduler=None,
        use_gpu=False,
        random_state=42
    )
    
    model_with_wd.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    
    print("\nComparison:")
    print(f"Without weight decay - Val accuracy: {model_no_wd.history['val_metric'][-1]:.4f}")
    print(f"With weight decay    - Val accuracy: {model_with_wd.history['val_metric'][-1]:.4f}")
    print("\nWeight decay helps prevent overfitting, especially with high-dimensional data")


def example_4_cosine_scheduler():
    """Example 4: Cosine annealing learning rate schedule."""
    print("\n" + "=" * 80)
    print("Example 4: Cosine Annealing LR Schedule")
    print("=" * 80)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train with cosine annealing
    model = MLPClassifier(
        hidden_layers=[128, 64],
        learning_rate=0.01,
        max_epochs=100,
        lr_scheduler='cosine',  # Cosine annealing
        early_stopping_patience=20,
        use_gpu=False,
        random_state=42
    )
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)
    
    print(f"\nCosine annealing smoothly reduces LR from {0.01} to near 0")
    print(f"Final validation accuracy: {model.history['val_metric'][-1]:.4f}")


def example_5_different_optimizers():
    """Example 5: Comparing different optimizers with optimizations."""
    print("\n" + "=" * 80)
    print("Example 5: Comparing Optimizers (Adam vs AdamW vs SGD)")
    print("=" * 80)
    
    # Generate dataset
    result = make_regression(
        n_samples=800,
        n_features=15,
        random_state=42
    )
    X = result[0]
    y = result[1]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    optimizers = ['adam', 'adamw', 'sgd']
    results = {}
    
    for opt_name in optimizers:
        model = MLPRegressor(
            hidden_layers=[128, 64],
            learning_rate=0.001 if opt_name != 'sgd' else 0.01,  # SGD needs higher LR
            optimizer_name=opt_name,
            weight_decay=0.001,
            lr_scheduler='plateau',
            max_epochs=50,
            use_gpu=False,
            random_state=42
        )
        
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
        
        results[opt_name] = {
            'val_rmse': model.history['val_metric'][-1],
            'epochs': len(model.history['train_loss'])
        }
    
    print("\nOptimizer Comparison:")
    print(f"{'Optimizer':<15} {'Val RMSE':<15} {'Epochs':<10}")
    print("-" * 40)
    for opt, metrics in results.items():
        print(f"{opt:<15} {metrics['val_rmse']:<15.4f} {metrics['epochs']:<10}")


def example_6_full_optimization():
    """Example 6: All optimizations combined."""
    print("\n" + "=" * 80)
    print("Example 6: Full Optimization Stack")
    print("=" * 80)
    
    # Generate challenging dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=20,
        n_redundant=10,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train with all optimizations
    model = MLPClassifier(
        hidden_layers=[256, 128, 64],
        activation='relu',
        dropout_rate=0.3,
        use_batch_norm=True,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=100,
        early_stopping_patience=15,
        optimizer_name='adamw',
        weight_decay=0.01,  # L2 regularization
        lr_scheduler='plateau',  # Adaptive LR
        lr_patience=7,
        lr_factor=0.5,
        gradient_clip_value=1.0,  # Gradient clipping
        use_gpu=False,
        random_state=42
    )
    
    print("\nTraining with full optimization stack:")
    print("- AdamW optimizer with weight_decay=0.01")
    print("- ReduceLROnPlateau scheduler (patience=7, factor=0.5)")
    print("- Gradient clipping (max_norm=1.0)")
    print("- Batch normalization + Dropout (0.3)")
    print("- Early stopping (patience=15)")
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)
    
    # Evaluate
    train_acc = model.history['train_metric'][-1]
    val_acc = model.history['val_metric'][-1]
    
    print(f"\nFinal Results:")
    print(f"Training accuracy:   {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Epochs trained:      {len(model.history['train_loss'])}")
    print(f"\nOverfitting check: {abs(train_acc - val_acc):.4f} gap")
    
    # Model size
    size_info = model.get_model_size()
    print(f"\nModel size: {size_info['total_params']:,} parameters")
    print(f"Memory: {size_info['size_mb']:.2f} MB")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MLP OPTIMIZATION EXAMPLES")
    print("Demonstrating advanced training optimizations in PyTorch MLPs")
    print("=" * 80)
    
    # Run all examples
    example_1_lr_scheduling()
    example_2_gradient_clipping()
    example_3_weight_decay()
    example_4_cosine_scheduler()
    example_5_different_optimizers()
    example_6_full_optimization()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nKey Optimization Features:")
    print("- Learning Rate Scheduling (plateau, step, cosine)")
    print("- Gradient Clipping for stability")
    print("- Weight Decay (L2 regularization)")
    print("- Multiple Optimizers (Adam, AdamW, SGD, RMSprop)")
    print("- Backward compatible with existing models")
    print("=" * 80)
