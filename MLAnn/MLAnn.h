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

#include "Eigen/Eigen"
class MLAnn{
public:
	//House keeping
	MLAnn(int, int, int, int);
	MLAnn(void);
	~MLAnn(void);
	bool writeToFile(Eigen::MatrixXd *, std::string);
	bool writeStateCsv(std::string);
	void setNumInputNodes(int num);
	void setNumOutputNodes(int num);
	void setNumHiddenLayers(int num);
	void setNumNodesPerLayer(int num);
	void setExpectedValues(Eigen::VectorXd);
	int getNumInputNodes(void);
	int getNumOutputNodes(void);
	int getNumHiddenLayers(void);
	int getNumNodesPerLayer(void);
	Eigen::VectorXd getError(void);
	bool isInitalized(void);
	bool init(int, int, int, int);
	
	//Business End
	bool forwardProp(void);
	bool reverseProp(void);
	void printState(void);
	
private:
	bool initalized;
	int numInputNodes;
	int numOutputNodes;
	int numHiddenLayers;
	int numNodesPerLayer;
	Eigen::VectorXd inputWeights;	// weights for the input layer
	Eigen::VectorXd outputWeights;	// weights fo the output layer
	Eigen::VectorXd expectedValues;	// expected output of the network
	Eigen::VectorXd outputError;	// Error vector at output layer
	Eigen::MatrixXd hiddenWeights;	// weights of hidden layers
	Eigen::MatrixXd ilField;		// induced local field
	
};
#endif /* defined(__MLAnn__MLAnn__) */
