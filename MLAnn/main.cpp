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
	if (argc < 6) {
		printHelp();
		exit(0);
	}
	int iNodes = atoi(argv[1]);
	int oNodes = atoi(argv[2]);
	int hLayers = atoi(argv[3]);
	int nodesPerLayer = atoi(argv[4]);
	string fName(argv[5]);
	
	MLAnn nn(iNodes, oNodes,hLayers, nodesPerLayer);
	
	cout << "Neural Network Initalized:" << endl;
	cout << "Output file: " << fName << endl;
	nn.printState();
	
	
    return 0;
}

void printHelp(void){
	cout << "Multi Layer ANN training program." << endl
	<< "Useage: mlAnn <number of input nodes> <number of output nodes> <number of hidden layers> <number of nodes per hidden layer> <\"path to output file\"" << endl;
	
}