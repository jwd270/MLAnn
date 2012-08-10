//
//  MLAnn.cpp
//  MLAnn
//
//  Created by James Dean on 8/8/12.
//  Copyright (c) 2012 James Dean. All rights reserved.
//

#include <fstream>
#include <cstdlib>
#include "MLAnn.h"

using namespace std;
using namespace Eigen;

/* 
 * Neural Network implementation code
 */


bool MLAnn::forwardProp(void){
	if (!inputValid || !initalized) { // If it hasn't been initalized or the values loaded, quit
		return false;
	}
	VectorXd currentWeights = VectorXd(numNodesPerLayer);
	int nodeOffset = 0;		// Index into the weight array for the current node
	// Initalize first row of local field with inputs and input weights
	for (int inputCnt = 0; inputCnt < numInputNodes; inputCnt++) {
		currentWeights = weightMat.block(nodeOffset, 0, numNodesPerLayer, 1);
		for (int nodeCnt = 0; nodeCnt < numNodesPerLayer; nodeCnt++) {
			ilField(nodeCnt,0) += inputValues(inputCnt) * currentWeights(nodeCnt);
		}
		nodeOffset += numNodesPerLayer;
	}
	
	//Compute induced local field for hidden nodes
	nodeOffset = 0;
	VectorXd fieldVec = VectorXd::Zero(numNodesPerLayer);
	for (int layerCnt = 1; layerCnt < numLayers; layerCnt++) {
		for (int nodeCnt = 0; nodeCnt < numNodesPerLayer; nodeCnt++) {
			currentWeights = weightMat.block(nodeOffset, layerCnt, numNodesPerLayer, 1);
			for (int sumCnt = 0; sumCnt < numNodesPerLayer; sumCnt++) {
				ilField(nodeCnt,layerCnt) += actFunc(ilField(sumCnt,layerCnt - 1) * currentWeights(sumCnt)); //Can't do dot product becuase ilField does not take into account the activation function
			}
			nodeOffset += numNodesPerLayer;
		}
		
	}
	
	//Compute output layer solution
	nodeOffset = 0;
	for (int outputCnt = 0; outputCnt < numOutputNodes; outputCnt++) {
		currentWeights = weightMat.block(nodeOffset, numLayers, numNodesPerLayer, 1);
		for (int nodeCnt = 0; nodeCnt < numNodesPerLayer; nodeCnt++) {
			outputField(outputCnt) += ilField(nodeCnt,numLayers - 1) * currentWeights(nodeCnt);
		}
		nodeOffset += numNodesPerLayer;
	}
	
	for (int outputCnt = 0; outputCnt < numOutputNodes; outputCnt++) {
		outputValues(outputCnt) = actFunc(outputField(outputCnt));
	}
	
	//Compute error value
	for (int errorCnt = 0; errorCnt < numOutputNodes; errorCnt++) {
		outputError(errorCnt) = expectedValues(errorCnt) - outputValues(errorCnt);
		meanError += (double).5*(pow(outputError(errorCnt), 2));
	}
	
	

	outputValid = true;
	errorValid = true;
	return true;
}

bool MLAnn::reverseProp(void){
	if (!outputValid || !errorValid) { //If it hasn't run forward yet, quit
		return false;
	}
	VectorXd currentWeights = VectorXd(numNodesPerLayer);
	VectorXd gradVect = VectorXd(numNodesPerLayer);
	int nodeOffset = 0;
	/*
	 *This section computes the local gradients required for computing new weights.
	 */
	
	// Compute output local gradient
	for (int outputCnt = 0; outputCnt < numOutputNodes; outputCnt++) {
		outputGradField(outputCnt) = outputError(outputCnt) * actFuncPrime(outputField(outputCnt));
	}
	// Compute the first hidden layer local gradient
	for (int nodeCnt = 0; nodeCnt < numNodesPerLayer; nodeCnt++) {
		currentWeights = weightMat.block(nodeOffset, numWeightLayers - 1, numNodesPerLayer, 1);
		for (int outputCnt = 0; outputCnt < numOutputNodes; outputCnt++) {
			gradVect(outputCnt) = outputGradField(outputCnt) * currentWeights(outputCnt);
		}
		lgField(nodeCnt,numLayers - 1) = actFuncPrime(ilField(nodeCnt,numLayers - 1))*gradVect.sum();
		nodeOffset += numNodesPerLayer;
	}
	//Compute the hidden layers local gradients
	nodeOffset = 0;
	for (int layerCnt = numLayers - 2; layerCnt > -1; layerCnt--) {
		for (int nodeCnt = 0; nodeCnt < numNodesPerLayer; nodeCnt++) {
			currentWeights = weightMat.block(nodeOffset, layerCnt + 1, numNodesPerLayer, 1);
			for (int sumCnt = 0; sumCnt < numOutputNodes; sumCnt++) {
				gradVect(sumCnt) = lgField(sumCnt,layerCnt + 1) * currentWeights(sumCnt);
			}
			lgField(nodeCnt,layerCnt) = actFuncPrime(ilField(nodeCnt,layerCnt))*gradVect.sum();
			nodeOffset += numNodesPerLayer;
		}
		nodeOffset = 0;
	}

	/*
	 *This section computes and applies the weight updates
	 */

	//Compute output weight update
	VectorXd deltaWeight = VectorXd(numNodesPerLayer);
	nodeOffset = 0;
	for (int nodeCnt = 0; nodeCnt < numNodesPerLayer; nodeCnt++) {
		currentWeights = weightMat.block(nodeOffset, numWeightLayers - 1, numNodesPerLayer, 1); 
		for (int outputCnt = 0; outputCnt < numOutputNodes; outputCnt++) {
			deltaWeight(outputCnt) = momentumConst*currentWeights(outputCnt) + learnRate*outputGradField(outputCnt)*actFunc(ilField(nodeCnt, numLayers - 1));
		}
		weightMat.block(nodeOffset, numWeightLayers - 1, numNodesPerLayer, 1) += deltaWeight;
		nodeOffset += numNodesPerLayer;
	}
	
	//Compute the hidden network wieght updates
	nodeOffset = 0;
	for (int weightLayerCnt = numWeightLayers - 2; weightLayerCnt > 0 ; weightLayerCnt--) {
		for (int nodeCnt = 0; nodeCnt < numNodesPerLayer; nodeCnt++) {
			currentWeights = weightMat.block(nodeOffset, weightLayerCnt, numNodesPerLayer, 1);
			for (int weightCnt = 0; weightCnt < numNodesPerLayer; weightCnt++) {
				deltaWeight(weightCnt) = momentumConst*currentWeights(weightCnt) + learnRate*lgField(weightCnt, weightLayerCnt)*actFunc(ilField(nodeCnt,weightLayerCnt - 1));
			}
			weightMat.block(nodeOffset, weightLayerCnt, numNodesPerLayer, 1) += deltaWeight;
			nodeOffset += numNodesPerLayer;
		}
		nodeOffset = 0;
	}
	// Compute the input weight updates
	nodeOffset = 0;
	for (int inputCnt = 0; inputCnt < numInputNodes; inputCnt++) {
		currentWeights = weightMat.block(nodeOffset, 0, numNodesPerLayer, 1);
		for (int nodeCnt = 0; nodeCnt < numNodesPerLayer; nodeCnt++) {
			deltaWeight(nodeCnt) = momentumConst*currentWeights(nodeCnt) + learnRate*actFuncPrime(inputValues(inputCnt))*actFunc(ilField(nodeCnt, 0));
		}
		cout << "Input Delta Weight:" << endl << deltaWeight << endl;
		weightMat.block(nodeOffset, 0, numNodesPerLayer, 1) += deltaWeight;
		nodeOffset += numNodesPerLayer;
	}

	// Invalidate the data -- cant run back again with out first running forward
	outputValid = false;
	errorValid = false;
	inputValid = true;
	return true;
}

double MLAnn::actFunc(double v){
	if(!userHyperbolic){
		return pow(1+exp(-v), -1);// Sigmoidal activation function with a range of 0 to 1
	}else{
		return tanh(v);					// Sigmoidal activation function with range of -1 to 1
	}
}

double MLAnn::actFuncPrime(double v){
	if(!userHyperbolic){
		return exp(v)/pow(1 + exp(v), 2);		// d/dt of 1/(1+e^(-v)) = e^v / (1 + e^v)^2
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
	numLayers = layers;
	numWeightLayers = layers + 1;
	numNodesPerLayer = nodesPerLayer;
	
	expectedValues = VectorXd::Zero(oNodes);
	inputValues = VectorXd::Zero(iNodes);
	outputField = VectorXd::Zero(oNodes);
	outputValues = VectorXd::Zero(oNodes);
	outputError = VectorXd::Zero(oNodes);
	weightMat = MatrixXd::Ones(nodesPerLayer*nodesPerLayer,numWeightLayers);		//Each layer is fully connected, requireing nodesPerLayer^2 weights per layer
	ilField = MatrixXd::Zero(nodesPerLayer,layers);
	lgField = MatrixXd::Zero(nodesPerLayer, layers);
	outputGradField = VectorXd::Zero(oNodes);
	inputGradField = VectorXd::Zero(iNodes);
	outputValid = false;
	errorValid = false;
	inputValid = false;
	userHyperbolic = false;
	meanError = 0;
	learnRate = 1;
	momentumConst = 0;
	
	// Fill weight matrix with zero mean weights.  This impementation does not guarentee that for any input, the output will be in the sweet spot, but it should be for input values < 100.
	//srand(time(NULL));
	srand(1); //used for testing to generate the same weight matrix every time.
	for (int rowCnt = 0 ; rowCnt < weightMat.rows(); rowCnt++) {
		for (int colCnt = 0; colCnt < weightMat.cols(); colCnt++) {
			weightMat(rowCnt,colCnt) = ((double)(rand() % 20) - 10) * 0.0001;
		}
	}
	initalized = true;
	return true;
}

void MLAnn::printState(bool more){
	cout << "Input: " << inputValues.rows() << "x" << inputValues.cols() << endl << inputValues << endl;
	cout << "Weights: " << weightMat.rows() << "x" << weightMat.cols() << endl << weightMat << endl;
	cout << "Output: " << outputValues.rows() << "x" << outputValues.cols() << endl << outputValues << endl;
	cout << "Error Vector: " << outputError.rows() << "x" << outputError.cols() << endl << outputError << endl;
	cout << "Mean Squarred Error: " << meanError << endl;
	if(more){
		cout << "--- Extended State ---" << endl;
		cout << "Induced Local Field: " << ilField.rows() << "x" << ilField.cols() << endl << ilField << endl;
		cout << "Output Local Field:" << outputField.rows() << "x" << outputField.cols() << endl << outputField << endl;
		cout << "Local Gradient Field: " << lgField.rows() << "x" << lgField.cols() << endl << lgField << endl;
		cout << "Ouput Gradient Field:" << outputGradField.rows() << "x" << outputGradField.cols() << endl << outputGradField << endl;
	}
	cout << "--- End ---"	<< endl;
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

/*
 * Getters and Setters
 */
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
	return numLayers;
}

int MLAnn::getNumNodesPerLayer(){
	return numNodesPerLayer;
}

VectorXd MLAnn::getErrorVec(){
	return outputError;
}

double MLAnn::getMeanError(){
	return meanError;
}

void MLAnn::setNumInputNodes(int num){
	numInputNodes = num;
}

void MLAnn::setNumOutputNodes(int num){
	numOutputNodes = num;
}

void MLAnn::setNumLayers(int num){
	numLayers = num;
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

void MLAnn::setLearnRate(double lr){
	learnRate = lr;
}

void MLAnn::setMomentumConst(double mc){
	momentumConst = mc;
}
