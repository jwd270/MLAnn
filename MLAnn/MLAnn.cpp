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

MLAnn::~MLAnn(){
	initalized = false;
}

MLAnn::MLAnn(int iNodes, int oNodes, int hLayers, int nodesPerLayer){
	initalized = false;
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
	
	inputWeights = VectorXd::Ones(iNodes);
	outputWeights = VectorXd::Ones(oNodes);
	expectedValues = VectorXd(oNodes);
	outputError = VectorXd(oNodes);
	hiddenWeights = MatrixXd::Ones(nodesPerLayer,hLayers);
	ilField = MatrixXd::Zero(nodesPerLayer,hLayers);
	
	initalized = true;
	return true;
}

void MLAnn::printState(){
	cout << "Input Weight Vector: "  << inputWeights.rows() << "x" << inputWeights.cols() << endl << inputWeights << endl;
	cout << "Hidden Weights: " << hiddenWeights.rows() << "x" << hiddenWeights.cols() << endl << hiddenWeights << endl;
	cout << "Induced Local Field: " << ilField.rows() << "x" << ilField.cols() << endl << ilField << endl;
	cout << "Output Weight Vector: " << outputWeights.rows() << "x" << outputWeights.cols() << endl << outputWeights << endl;
	cout << "Error: " << outputError.rows() << "x" << outputError.cols() << endl << outputError << endl;
}

bool MLAnn::writeStateCsv(std::string fName){
	fName += "_state.csv";
	fstream fileBuff(fName.c_str(),ios_base::out|ios_base::trunc);
	
	if(!fileBuff.is_open()){
		cout << "Failed to open file: " << fName << endl;
		return false;
	}
	
	fileBuff.close();
	return true;
}

bool MLAnn::writeToFile(MatrixXd *dat, std::string fName){
	fName += "_item.csv";
	fstream fileBuff;
	fileBuff.open(fName.c_str(), ios_base::out|ios_base::trunc);
	if(!fileBuff.is_open()){
		cout << "Failed to open output file: " << fName << endl;
		return false;
	}
		
	fileBuff.close();
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

VectorXd MLAnn::getError(){
	return outputError;
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

void MLAnn::setExpectedValues(Eigen::VectorXd expVal){
	expectedValues = expVal;
}

