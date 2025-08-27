# inatVisionTraining

iNaturalist makes a subset of its machine learning models publicly available while keeping full species classification models private due to intellectual property considerations and organizational policy. We provide [“small” models](https://github.com/inaturalist/model-files) trained on approximately 500 taxa, including taxonomy files and a geographic model, which are suitable for on-device testing and other applications. Additionally, researchers have independently developed and released open-source models based on iNaturalist data, which can be found in various model distribution venues (for example [Hugging Face](https://huggingface.co/models?search=inaturalist) or [Kaggle](https://www.kaggle.com/models?query=inaturalist)).

## Export

In `eval_export/`, `export_cv_model.py` converts from checkpoints to an .h5 model for web deployment.  `export_coreml_geomodel.py` converts to the .h5 model to coreml for iOS deployment, `export_tflite_geomodel.py` converts the .h5 model to tflite for android deployment.
