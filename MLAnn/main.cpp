//
//  main.cpp
//  MLAnn
//
//  Created by James Dean on 8/8/12.
//  Copyright (c) 2012 James Dean. All rights reserved.
//
/*
 * This applicaaiton trains a multi-layer artifical neural network to solve f(x)=1/x
 */

#include <iostream>
#include <string>
#include <cmath>
#include "Eigen/Eigen"
#include "mlAnnApp.h"
#include "MLAnn.h"

#define TRAINING_START 1
#define TRAINING_STEP 1
#define TRAINING_END 10

using namespace Eigen;
using namespace std;

int main(int argc, const char * argv[])
{
	if (argc < 7) {
		printHelp();
		exit(0);
	}
	
	int trainDataSize = (int)(((TRAINING_END - TRAINING_START) + 1)/TRAINING_STEP);
	int iNodes = atoi(argv[1]);
	int oNodes = atoi(argv[2]);
	int layers = atoi(argv[3]);
	int nodesPerLayer = atoi(argv[4]);
	int trainingEpochs = atoi(argv[5]);
	string fName(argv[6]);

	
	MLAnn nn(iNodes, oNodes,layers, nodesPerLayer);
	
	cout << "Output file: " << fName << endl;
	cout << "Neural Network Initalized:" << endl;
	nn.printState(false);
	
	VectorXd trainData = VectorXd(trainDataSize);
	VectorXd resultData = VectorXd(trainDataSize);
	ArrayXXd errorArray = ArrayXXd(trainingEpochs,oNodes);
	
	trainData = VectorXd::LinSpaced(trainDataSize, TRAINING_START,TRAINING_END);
	resultData = recip(trainData);
	
	for (int epoch = 0; epoch < trainingEpochs; epoch++) {
		for (int sample = 0; sample < 1 /*trainDataSize*/; sample++) {
			nn.setInputValues(trainData.row(sample));
			nn.setExpectedValues(resultData.row(sample));
			nn.forwardProp();
			//nn.printState(true);
			nn.reverseProp();
			nn.printState(true);
			
		}
	}
/*
	for (int cnt = 0; cnt < trainDataSize; cnt++) {
		nn.setInputValues(trainData.row(cnt));
		nn.setExpectedValues(resultData.row(cnt));
		nn.forwardProp();
		nn.printState(false);
	}
*/
    return 0;
}

VectorXd recip(VectorXd vec){
	//Compute element wise reciprocal
	ArrayXd unit = ArrayXd(vec.size());
	unit = ArrayXd::Ones(vec.size());
	ArrayXd res = ArrayXd(vec.size());
	res = unit / vec.array();
	return res.matrix();
	
}

void printHelp(void){
	cout << "Multi Layer ANN training program." << endl
	<< "Useage: mlAnn <number of input nodes> <number of output nodes> <number of layers> <number of nodes per layer> <number of training epochs> <\"path to output file\">" << endl;
	
}