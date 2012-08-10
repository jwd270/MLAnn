//
//  MLAnn.h
//  MLAnn
//
//  Created by James Dean on 8/8/12.
//  Copyright (c) 2012 James Dean. All rights reserved.
//
/*
 * This class implements a flexible multi-layer perceptron.  The internal node structure is limited to a lock format where each layer has the same number of hidden nodes.
 * The input and output layers can have a different number of nodes than the hidden layers.
 * Each layer is fully connected with the previous layer
 */

#ifndef __MLAnn__MLAnn__
#define __MLAnn__MLAnn__

#include <iostream>
#include <fstream>
#include <cmath>

#include "Eigen/Eigen"
class MLAnn{
public:
	//House keeping
	MLAnn(int, int, int, int);
	MLAnn(void);
	~MLAnn(void);
	bool writeToFile(Eigen::MatrixXd *, std::string);
	bool writeStateCsv(std::string);
	//Setters
	void setNumInputNodes(int num);
	void setNumOutputNodes(int num);
	void setNumLayers(int num);
	void setNumNodesPerLayer(int num);
	void setExpectedValues(Eigen::VectorXd);
	void setInputValues(Eigen::VectorXd);
	void setUseHyperbolic(bool);
	void setLearnRate(double);
	void setMomentumConst(double);
	//Getters
	int getNumInputNodes(void);
	int getNumOutputNodes(void);
	int getNumHiddenLayers(void);
	int getNumNodesPerLayer(void);
	Eigen::VectorXd getErrorVec(void);
	double getMeanError(void);
	bool isOuputValid(void);
	bool isErrorValid(void);
	bool isInitalized(void);
	bool init(int, int, int, int);
	
	//Business End
	bool forwardProp(void);
	bool reverseProp(void);
	bool doTrainEpoch(void);
	void printState(bool);
	
private:
	bool initalized;
	bool inputValid;
	bool outputValid;
	bool errorValid;
	bool userHyperbolic;
	int numInputNodes;
	int numOutputNodes;
	int numLayers;
	int numWeightLayers;
	int numNodesPerLayer;
	double meanError;
	double learnRate;
	double momentumConst;
	Eigen::VectorXd inputValues;	// Vector of inputs to system
	Eigen::VectorXd outputField;	// Local field for output neurons
	Eigen::VectorXd outputValues;	// Vector of outputs from the system
	Eigen::VectorXd expectedValues;	// expected output of the network
	Eigen::VectorXd outputError;	// Error vector at output layer
	Eigen::MatrixXd weightMat;		// weights of hidden layers
	Eigen::MatrixXd ilField;		// induced local field
	Eigen::MatrixXd lgField;		// Local gradient field
	Eigen::VectorXd outputGradField;	// Local gradient for output neurons
	Eigen::VectorXd inputGradField;		// LOcal gradient for input neurons
	
	double actFunc(double);			// computes value of activation function
	double actFuncPrime(double);	// computes value of the deritive of the activation function
};
#endif /* defined(__MLAnn__MLAnn__) */
