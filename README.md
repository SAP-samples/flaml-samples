# FLAML Samples
<!--- Register repository https://api.reuse.software/register, then add REUSE badge:
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/REPO-NAME)](https://api.reuse.software/info/github.com/SAP-samples/REPO-NAME)
-->

## Description

In this repository, we demonstrate the power of automated machine learning (AutoML) by going into how [FLAML](https://microsoft.github.io/FLAML/) - Microsoft's AutoML framework - can be used. We go over three examples in the following jupyter notebooks:
 1. [Simple FLAML demo](./SimpleFLAMLDemo.ipynb)
 2. [Classification FLAML Demo](./ClassificationFLAMLDemo.ipynb)
 3. [Regression FLAML Demo](./RegressionFLAMLDemo.ipynb)

The first demo introduces the user to the concept of how FLAML can be used, by working on the commonly used IRIS dataset. Examples two and three highlight how FLAML, and more generally AutoML, can be used in an SAP context. The second notebook goes over a classification example, where FLAML uses purchase order data to determine if a purchase order will be accepted or rejected - this is the Purchase Order Requisition Use Case. The third notebook goes over a regression example, where FLAML uses item data to determine the max allocation (or stock) of an item in a store - this is the Retail Order Use Case.

These demos cover the basics for how to use the FLAML library. For those who are interested in learning more, we strongly recommend their [repository](https://github.com/microsoft/FLAML), and specifically the [automl.py file](https://github.com/microsoft/FLAML/blob/main/flaml/automl/automl.py), for a deeper dive.

## Requirements

Python version >= 3.8. Specific Python packages, and their installation commands, are already contained in the notebook examples.

## Download and Installation

Python downloads can be found [here](https://www.python.org/downloads/).

## Known Issues

No known issues.

## How to obtain support

[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing

If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License

Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
