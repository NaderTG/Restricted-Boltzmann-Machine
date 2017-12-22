/*
 * This code is an implementation of restricted Boltzmann machine.
 * Copyright (C) 2017 Nader Ganaba.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  main.cpp
 *
 *  Created by Nader on 10/12/2017.
 */

#include <iostream>
#include<cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Eigen/Dense"
#include "utils.h"
#include "mnist.h"
#include "rbm_vector.h"

int main(int argc, const char * argv[]) {
    std::cout<<"Testing restricted Boltzmann machine\n" << std::endl;
    
   // testSampling2(11);
    double learning_rate = 0.1;
    int training_epochs = 5000;
    int k = 1;

    int train_N = 6;
    int test_N = 2;
    int n_visible = 6;
    int n_hidden = 2;
    int batch_size=1;
 
    MNIST_DATA _dataset(6,6);
    _dataset.testSet();
    Eigen::VectorXd _test3 = Eigen::VectorXd::Zero(6); // = _dataset.getData(3);

    //Creating RBM
    RBM_Vector _rbmSingle(n_hidden, n_visible);
    _rbmSingle.setLearningRate(learning_rate);
 


    double error = 0.0;

    // train
    for(int epoch=0; epoch<training_epochs; epoch++) {
        error = 0.0;
        for(int i=0; i< train_N; i++) {
 
            MNIST_DATA _batch  =  _dataset.generateBatch_NonRand(1,i);
            if(epoch == 0){
                _test3 =  _batch.getData(0);
                std::cout << _test3 << std::endl<< std::endl<< std::endl;
            }
 
            error = _rbmSingle.RBM_contrastive_divergence(_batch,1,k);

        }

        if(epoch % 100 == 0){
            std::cout << "Epoch [" << epoch << "] error is = " << error << std::endl;
        }
    }
 
    MNIST_DATA _trailset(2,6);
    _trailset.trailSet();
 
    Eigen::VectorXd _test( 6);
    // Eigen::VectorXd _input(1,6);

    for(int i = 1; i < 2; i++){

        _test = _rbmSingle.RBM_reconstruction(_trailset, i);
        std::cout << _test << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
