//
//  MLAnn.cpp
//  MLAnn
//
//  Created by James Dean on 8/8/12.
//  Copyright (c) 2012 James Dean. All rights reserved.
//

#include <fstream>
#include "MLAnn.h"

using namespace std;
using namespace Eigen;

MLAnn::MLAnn(void){
	initalized = false;
}

MLAnn::MLAnn(int iNodes, int oNodes, int hLayers, int nodesPerLayer){
	
	init(iNodes,oNodes,hLayers,nodesPerLayer);
}

bool MLAnn::forwardProp(void){
	
	return true;
}

bool MLAnn::reverseProp(void){
	
	return true;
}
bool MLAnn::init(int iNodes, int oNodes, int hLayers, int nodesPerLayer){
	numInputNodes = iNodes;
	numOutputNodes = oNodes;
	numHiddenLayers = hLayers;
	numNodesPerLayer = nodesPerLayer;
	
	inputWeights = VectorXd(iNodes);
	outputWeights = VectorXd(oNodes);
	expectedValues = VectorXd(oNodes);
	outputError = VectorXd(oNodes);
	hiddenWeights = MatrixXd(nodesPerLayer,hLayers);
	ilField = MatrixXd(nodesPerLayer,hLayers);
	
	initalized = true;
	return true;
}

bool MLAnn::writeToFile(MatrixXd *dat, std::string fName){
	fstream fileBuff;
	fileBuff.open(fName.c_str(), ios_base::out|ios_base::trunc);
	if(!fileBuff.is_open()){
		cout << "Failed to open output file: " << fName << endl;
		return false;
	}
		
	
	return true;
}

bool MLAnn::isInitalized(){
	return initalized;
}

int MLAnn::getNumInputNodes(){
	return numInputNodes;
}

int MLAnn::getNumOutputNodes(){
	return numOutputNodes;
}

int MLAnn::getNumHiddenLayers(){
	return numHiddenLayers;
}

int MLAnn::getNumNodesPerLayer(){
	return numNodesPerLayer;
}

void MLAnn::setNumInputNodes(int num){
	numInputNodes = num;
}

void MLAnn::setNumOutputNodes(int num){
	numOutputNodes = num;
}

void MLAnn::setNumHiddenLayers(int num){
	numHiddenLayers = num;
}

void MLAnn::setNumNodesPerLayer(int num){
	numNodesPerLayer = num;
	
}