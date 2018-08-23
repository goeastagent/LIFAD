# Deep learning-based Logitudinal heterogeneous data Integration Framework for AD-relevant feature extraction (LIFAD)

Deep learning-based python package for general purpose data integration framework. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Requirements
  * pandas
  * numpy
  * tensorflow
  * sklearn

### Example code

```
m = MMRNN()

m.append_component('m1', m1.shape[2], m1_hidden, m1.shape[1])
m.append_component('m2', m2.shape[2], m2_hidden, m2.shape[1])
m.append_component('m3', m3.shape[2], m3_hidden, m3.shape[1])
```

Setup each modality. In this example, 3 modalities of data (m1, m2, and m3) will be used.
The code above defines name of the modality, dimension of input, dimension of hidden state, and length of time series. Data m has a shape (#samples, length of time series, size of input dimension). 
```

m.append_data('m1', IDs_m1, m1, y_m1, seqlen_m1)
m.append_data('m2', IDs_m2, m2, y_m2, seqlen_m2)
m.append_data('m3', IDs_m3, m3, y_m3, seqlen_m3)

m.append_test_overlapIDs(testIDs)
m.append_training_overlapIDs(trainIDs)
```
Feeding data to LIFAD. Training samples as well test samples should be fed to LIFAD. And training samples and test samples are seperated by ID. IDs, data (independent variable), y (dependent variable), and seqlen (time lengths of individual sample) should be arranged in order. 

```
m.build_integrative_network()
m.training(batch_size)

m.evalute_accuracy()
```

Training and test.

The entire code is given in the file "exmple_code.py"

## Feature extraction

Explain how to run the automated tests for this system


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Garam Lee** - *Initial work* - [LIFAD](https://github.com/goeastagent/LIFAD)

See also the list of [contributors](https://github.com/goeastagent/LIFAD/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
