def pipeline(featurizer, model, **kwargs):
    """
    Pipeline for training a model.
    """
    # Load data
    train_mols, val_mols, test_mols = load_data(featurizer, **kwargs)

    # Train model
    trained_model = train_model(model, train_mols, val_mols, **kwargs)

    # Evaluate model
    test_scores = evaluate_model(trained_model, test_mols)

    return trained_model, test_scores