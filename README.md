# DCGP
Gaussian processes model for non-differentiable functions with jump discontinuities written in Matlab

# A brief set of instructions
For model training, call the function "DCGP_fit(xTrain,yTrain,lTrain,V,D,vargin)" where
  xTrain:   A column vector that is used to provide the observed input data.
  yTrain:   A column vector that is used to provide the observed output data.
  lTrain:   A column vector that is used to indicate the region that a sample is in.
  V, D:     These are the matrix and vector of the basis function that defines a boundary.
  vargin:   These are optional inputs that can include 'DC_Type' is the type of boundary that can be set to "Jump", "NonDif" and "Jump_NonDif" for jump discontinious, non-differentiable, and jointly non-differentiable and jump discontinious,       respectively. The type of boundary is set to "Jump" by default. In addition, 'l' is used to provide a function indicating the region that a sample is in, and that can be used to "Jump" discontinuities only. 

For prediction, call the function "Pred_DCGP(xPred,lPred,nSam,model,vargin)" where
  xPred:  A column vector that is used to provide new inputs for which to predict the output.
  lPred:  A column vector that is used to indicate the region that the new predictive samples are in.
  nSam:   A scalar indicating the number of new samples for which to make a prediction. 
  model:  The trained model is obtained from the training function. 
  vargin: These are optional inputs that can include 'Bayes_Type' that is used to indicate if you want to use full Bayesian prediction "Full" or partial Bayesian prediction that includes hyperparameters obtained through maximum likelihood estimation "Partial". The default is set to "Partial". In addition, 'Bayes_nMcSamples' is used to set the number of Monte Carlo samples in the Bayesian scheme, the default is set to 1000. 

The code includes multiple test problems that can be used as examples of how to use the model. 

# Citing
The associated conference and journal papers are currently under review and will be added here once accepted. 

  # Questions
  Create a pull request or contact me anton.vanbeek@ucd.ie

