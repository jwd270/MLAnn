//
//  main.cpp
//  MLAnn
//
//  Created by James Dean on 8/8/12.
//  Copyright (c) 2012 James Dean. All rights reserved.
//

#include <iostream>
#include <string>
#include "Eigen/Eigen"
#include "mlAnnApp.h"

using namespace Eigen;
using namespace std;

int main(int argc, const char * argv[])
{
	if (argc < 3) {
		printHelp();
	}
	
    return 0;
}





void printHelp(void){
	cout << "Multi Layer ANN training program." << endl
	<< "Useage: mlAnn <number of layers> <number of nodes per layer>" << endl;
	
}