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

/* 
 * Neural Network implementation code
 */


bool MLAnn::forwardProp(void){
	if (!inputValid) {
		return false;
	}
	//TODO: Add forward propagation code
	inputValid = false;
	outputValid = true;
	errorValid = true;
	return true;
}

bool MLAnn::reverseProp(void){
	if (!outputValid) {
		return false;
	}
	//TODO:  Add reverse propagation code
	outputValid = false;
	errorValid = false;
	return true;
}

double MLAnn::actFunc(double v){
	if(!userHyperbolic){
		return pow(1 + exp(-v), -1);	// Sigmoidal activation function with a range of 0 to 1
	}else{
		return tanh(v);					// Sigmoidal activation function with range of -1 to 1
	}
}

double MLAnn::actFuncPrime(double v){
	if(!userHyperbolic){
		return exp(v)/pow(exp(v)+1, 2);		// d/dt of 1/(1+e^(-1))
	}else{
		return 1/pow(cosh(v),2);			// d/dt of tanh(t) = 1/cosh^2(t)
	}
}

/*
 * House keeping code
 */

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

bool MLAnn::init(int iNodes, int oNodes, int layers, int nodesPerLayer){
	numInputNodes = iNodes;
	numOutputNodes = oNodes;
	numHiddenLayers = layers;
	numNodesPerLayer = nodesPerLayer;
	
	expectedValues = VectorXd::Zero(oNodes);
	inputValues = VectorXd::Zero(iNodes);
	outputValues = VectorXd::Zero(oNodes);
	outputError = VectorXd::Zero(oNodes);
	weightMat = MatrixXd::Ones(nodesPerLayer,layers);
	ilField = MatrixXd::Zero(nodesPerLayer,layers);
	outputValid = false;
	errorValid = false;
	inputValid = false;
	userHyperbolic = false;
	
	initalized = true;
	return true;
}

void MLAnn::printState(){
	cout << "Input: " << inputValues.rows() << "x" << inputValues.cols() << endl << inputValues << endl;
	cout << "Weights: " << weightMat.rows() << "x" << weightMat.cols() << endl << weightMat << endl;
	cout << "Induced Local Field: " << ilField.rows() << "x" << ilField.cols() << endl << ilField << endl;
	cout << "Output: " << outputValues.rows() << "x" << outputValues.cols() << endl << outputValues << endl;
	cout << "Error: " << outputError.rows() << "x" << outputError.cols() << endl << outputError << endl;
}

bool MLAnn::writeStateCsv(std::string fName){
	fName += "_state.csv";
	fstream fileBuff(fName.c_str(),ios_base::out|ios_base::trunc);
	
	if(!fileBuff.is_open()){
		cerr << "Failed to open file: " << fName << endl;
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
		cerr << "Failed to open output file: " << fName << endl;
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

void MLAnn::setUseHyperbolic(bool in){
	userHyperbolic = in;
}

