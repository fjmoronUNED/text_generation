import setuptools

setuptools.setup(
    name="ml_training",
    version="0.0.1",
    author="fjmoronreyes",
    description="package for training ML models",
    packages=["ml_training", "ml_training.training", "ml_training.inference"],
    #package_data={'': 'config/lstm_keras_config.yaml'},
    include_package_data=True
)
