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
 *  rbm_vector.h
 *
 *  Created by Nader on 10/12/2017.
 */

#ifndef rbm_vector_h
#define rbm_vector_h

#include "Eigen/Dense"
#include "utils.h"
#include "mnist.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#define _activationFunc(x) (_atvFunc->operator()(x))
#define _sample(y,x) (_rndSample->operator()(y,x))

class RBM_Vector{
private:
    Eigen::MatrixXd _weight;
    Eigen::VectorXd _hid_layer;
    Eigen::VectorXd _vis_layer;
    int _num_hid, _num_vis;
    double _learning_rate;
    std::vector<int> *img_sizes; //size =4. [height, width, h_tiles, w_tiles]
    
    RandVar<double> *_rndSample;
    AtvFunc<double> *_atvFunc;
    
public:
    RBM_Vector(int num_hid, int num_vis):_num_hid(num_hid), _num_vis(num_vis){
        double a = 1.0/ num_vis;
        _hid_layer = Eigen::VectorXd::Zero(_num_hid);
        _vis_layer = Eigen::VectorXd::Zero(_num_vis);
        
        _weight.resize(_num_hid, _num_vis);
        
        _weight = a*(Eigen::MatrixXd::Random(_num_hid, _num_vis));
        img_sizes = new std::vector<int>(4);
        img_sizes->at(0) = 28;img_sizes->at(1) = 28;
        img_sizes->at(2) = 10;img_sizes->at(3) = 10;
        _rndSample = new Binomial<double>();
        _atvFunc = new Sigmoid<double>();
    }
    
    //Finish implementing it
    void initWeights(){
        
        double range = 1.0/_num_vis;
        for(int i = 0; i < _num_hid; i++){
            for(int j = 0; j < _num_vis; j++){
                // _weight(i, j) =  rand() / (RAND_MAX + 1.0) * (2.0*range) - range;
            }
        }
    }
    
    //Sampling h given v
    Eigen::MatrixXd sample_h_given_v(Eigen::VectorXd& data){
        Eigen::MatrixXd _result = Eigen::MatrixXd::Zero(2, _num_hid);
        
        Eigen::VectorXd _temp;
        
        _temp = _weight*data;
        _temp += _hid_layer;
        
        _temp =_activationFunc(_temp);
        _result.row(0) = _temp ;
        _result.row(1) =_sample(1, _temp);
        return _result;
    }
    
    Eigen::MatrixXd sample_v_given_h(Eigen::VectorXd& data){
        Eigen::MatrixXd _result = Eigen::MatrixXd::Zero(2,_num_vis);
        
        Eigen::VectorXd _temp;
        _temp = (data.transpose())*_weight;
        _temp += _vis_layer;
        _temp = _activationFunc(_temp);
        _result.row(0) = _temp ;
        _result.row(1) =_sample(1, _temp);
        
        return _result;
    }
    
    void setLearningRate(double val) { _learning_rate = val;}
    
    double RBM_contrastive_divergence(MNIST_DATA &data1, int batch_size,  int k){
        double errRBM = 0.0;
        //It should be data_input vector or something
        Eigen::MatrixXd _positivePhase = Eigen::MatrixXd::Zero(2,_num_hid);
        Eigen::MatrixXd _neg_hid_Phase = Eigen::MatrixXd::Zero(2,_num_hid);
        Eigen::MatrixXd _neg_vis_Phase = Eigen::MatrixXd::Zero(2,_num_vis);
        
        Eigen::VectorXd data = Eigen::VectorXd::Zero( _num_vis);
        
        Eigen::MatrixXd _d_W = Eigen::MatrixXd::Zero(_num_hid, _num_vis);
        Eigen::VectorXd _d_hid_bias = Eigen::VectorXd::Zero(_num_hid);
        Eigen::VectorXd _d_vis_bias = Eigen::VectorXd::Zero(_num_vis);
        
        for(int i_batch = 0; i_batch <batch_size; i_batch++ ){
            data = data1.getData(i_batch);
            _positivePhase = sample_h_given_v(data);
            
            //K = 0
            Eigen::VectorXd _pos_phase_data = _positivePhase.row(1);
            
            _neg_vis_Phase = sample_v_given_h(_pos_phase_data);
            Eigen::VectorXd _neg_vis_data = _neg_vis_Phase.row(0);
            _neg_hid_Phase = sample_h_given_v(_neg_vis_data);
            

            //Define Macros for layer getMean and stuff
            double temp;
            
            
            for(int i=0; i< _num_hid; i++) {
                for(int j=0; j< _num_vis; j++) {
                    _d_W(i,j) = _d_W(i,j) +  _learning_rate *(_positivePhase(0,i)*data[j] -_neg_hid_Phase(0,i)*(_neg_vis_Phase(1,j)) ) / batch_size;
                }

                temp = _learning_rate *(_positivePhase(1,i) - _neg_hid_Phase(0,i)) / batch_size;
                _d_hid_bias(i) =_d_hid_bias(i) + temp;

            }
            
            for(int i=0; i< _num_vis; i++) {
                temp = _learning_rate *(data[i] - _neg_vis_Phase(1,i) ) / batch_size;
                _d_vis_bias(i) = _d_vis_bias(i) + temp;
                errRBM += (data[i] - _neg_vis_Phase(0,i))*(data[i] - _neg_vis_Phase(1,i));
                
            }
            
        }
        
        errRBM = errRBM / (double)_num_vis;
        _weight += _d_W;
        _hid_layer += _d_hid_bias;
        _vis_layer += _d_vis_bias;
        
        return sqrt(errRBM);
    }
    
    Eigen::VectorXd RBM_reconstruction(MNIST_DATA &data1, int i_batch){
        Eigen::VectorXd data = Eigen::VectorXd::Zero( _num_vis);
        Eigen::VectorXd _hid_vec = Eigen::VectorXd::Zero( _num_hid);
        Eigen::VectorXd _result = Eigen::VectorXd::Zero(_num_vis);
        data = data1.getData(i_batch);
        
        double pre_sigmoid_activation = 0.0;
        Eigen::VectorXd _temp;
        
        _temp = _weight*data;
        _temp += _hid_layer;

        _temp =_activationFunc(_temp);

       // _result.row(0) = _temp ;
        _temp = _sample(1, _temp);

        
        return _temp;

     
    }
    
};


#endif /* rbm_vector_h */


