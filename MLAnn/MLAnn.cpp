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

MLAnn::MLAnn(int iNodes, int oNodes, int layers, int nodesPerLayer){
	initalized = false;
	init(iNodes,oNodes,layers,nodesPerLayer);
}

bool MLAnn::forwardProp(void){
	if (!inputValid) {
		return false;
	}
	inputValid = false;
	outputValid = true;
	
	return true;
}

bool MLAnn::reverseProp(void){
	
	return true;
}

double MLAnn::actFunc(double v, bool type=true){
	if(type){
		return pow(1 + exp(-v), -1);	// Sigmoidal activation function with a range of 0 to 1
	}else{
		return tanh(v);	// Sigmoidal activation function with range of -1 to 1
	}
}

bool MLAnn::init(int iNodes, int oNodes, int layers, int nodesPerLayer){
	numInputNodes = iNodes;
	numOutputNodes = oNodes;
	numHiddenLayers = layers;
	numNodesPerLayer = nodesPerLayer;
	
	expectedValues = VectorXd::Zero(oNodes);
	inputValues = VectorXd::Zero(iNodes);
	outputError = VectorXd::Zero(oNodes);
	weightMat = MatrixXd::Ones(nodesPerLayer,layers);
	ilField = MatrixXd::Zero(nodesPerLayer,layers);
	outputValid = false;
	errorValid = false;
	inputValid = false;
	
	initalized = true;
	return true;
}

void MLAnn::printState(){
	cout << "Weights: " << weightMat.rows() << "x" << weightMat.cols() << endl << weightMat << endl;
	cout << "Induced Local Field: " << ilField.rows() << "x" << ilField.cols() << endl << ilField << endl;
	cout << "Output:" << 
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

bool MLAnn::isErrorValid(){
	return errorValid;
}

bool MLAnn::isOuputValid(){
	return outputValid;
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

void MLAnn::setNumLayers(int num){
	numHiddenLayers = num;
}

void MLAnn::setNumNodesPerLayer(int num){
	numNodesPerLayer = num;
	
}

void MLAnn::setInputValues(Eigen::VectorXd in){
	inputValues = in;
	inputValid = true;
}

void MLAnn::setExpectedValues(Eigen::VectorXd expVal){
	expectedValues = expVal;
}

