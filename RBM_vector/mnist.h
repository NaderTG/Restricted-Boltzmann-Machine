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
 *  mnist.h
 *	Based on code supplied in https://compvisionlab.wordpress.com/
 *  Created by Nader on 10/12/2017.
 */

#ifndef mnist_h
#define mnist_h

#include <string>
#include <fstream>
#include <iostream>
#include "Eigen/Dense"
#include "utils.h"
//#define _data(a,b) (_dataSet->operator ()(a, b))

int
ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


class MNIST_DATA{
private:
    Eigen::MatrixXd _dataSet;
    int _N, _D;
    int _width, _height;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MNIST_DATA();
    MNIST_DATA(int N, int D):_N(N), _D(D){
        //D is 28*28 and N is the number of images
        _dataSet.resize(N,D);
        //_dataSet = NULL;
        _width = 0; _height = 0;
    }
    
    void setImgSize(int h, int w){
        _height = h; _width = w;
    }
    
    int getNSamples(){
        return _N;
    }
    int getSizeData(){
        return _D;
    }
    
    //Generate a mock MNIST training set for testing 
    void testSet(){
        int train_X[6][6] = {
            {1, 1, 1, 0, 0, 0},
            {1, 0, 1, 0, 0, 0},
            {1, 1, 1, 0, 0, 0},
            {0, 0, 1, 1, 1, 0},
            {0, 0, 1, 0, 1, 0},
            {0, 0, 1, 1, 1, 0}
        };
        _width = 6; _height = 1;
        for(int i = 0; i < 6; i++){
            for(int j = 0; j < 6; j++){
                _dataSet(i,j) = train_X[i][j];
            }
        }
        
        
    }
    
    //Generate a mock MNIST testing set
    void trailSet(){
        int test_X[2][6] = {
            {1, 1, 0, 0, 0, 0},
            {0, 0, 0, 1, 1, 0}
        };
        
        _width = 6; _height = 1;
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 6; j++){
                _dataSet(i,j) = test_X[i][j];
            }
        }
        
    }
    
    //Loading MNIST data
    void loadMNIST(std::string filename){
        std::ifstream file (filename, std::ios::binary);
        if (file.is_open()){
          
            int magicNumber = 0;
            int numImages = 0;
            int numRows = 0;
            int numCols = 0;
            file.read((char*) &magicNumber, sizeof(magicNumber));
            magicNumber = ReverseInt(magicNumber);
            file.read((char*) &numImages,sizeof(numImages));
            numImages = ReverseInt(numImages);
            file.read((char*) &numRows, sizeof(numRows));
            numRows = ReverseInt(numRows);
            file.read((char*) &numCols, sizeof(numCols));
            numCols = ReverseInt(numCols);
            
            _width = numCols; _height = numRows;

            for(int i = 0; i < numImages; ++i){
              
                for(int idx_r = 0; idx_r < numRows; ++idx_r){
                    for(int idx_c = 0; idx_c < numCols; ++idx_c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                    
                        if((double) temp>=128.0){

                            _dataSet(i, idx_r*numRows + idx_c) = 1.0;
                        }else{

                            _dataSet(i, idx_r*numRows + idx_c) = 0.0;
                        }
     
                    }
                }
  
            }
        }
    }
    void setData(int idx_x, int idx_y, double val){
        _dataSet( idx_x,  idx_y) = val;
    }
    
    void setData(const Eigen::MatrixXd& _data){
        
        //Check sizes
        if(_data.rows() != _N || _data.cols() != _D){
            std::cerr << "MNIST Error: Data sizes do not match\n";
        }
        
        _dataSet = _data;
        
    }

    
    MNIST_DATA generateBatch_NonRand(int batchSize, int startIdx) {
        MNIST_DATA _subset(batchSize, _D);
        
        int start_pos = randomIdx(_N, batchSize);
        // int start_pos = 1026;
        _subset.setImgSize(_height, _width);
        
        Eigen::MatrixXd  _sData = _dataSet.middleRows(startIdx, batchSize);
        _subset.setData(_sData);
        
        // _subset.printData(0);
        return _subset;
    }

    
    const MNIST_DATA generateBatch(int batchSize) const{
        MNIST_DATA _subset(batchSize, _D);

        int start_pos = 1026;
        _subset.setImgSize(_height, _width);
        for(int i = 0; i < batchSize; i++){
            for(int j = 0; j < _D; j++){
                
                _subset.setData(i, j,  _dataSet(start_pos + i,j));
            }
        }

        return _subset;
    }
    
    
    Eigen::VectorXd getData(int idx){
        return _dataSet.row(idx);
    }
    
    const Eigen::VectorXd  getData(int idx) const{
        return _dataSet.row(idx);
    }
    
    Eigen::MatrixXd getData(int start, int end){
        int diff = end - start + 1;
        return _dataSet.middleRows(start, diff);
    }
    
    const Eigen::MatrixXd getData(int start, int end) const{
        int diff = end - start + 1;
        return _dataSet.middleRows(start, diff);
    }
    
    void printData(int idx){
        int temp;
     
        for(int i = 0; i < _height; i++){
            for(int j = 0 ; j < _width-1; j++){
 
                temp = (int) _dataSet(idx,_width*i + j );
                std::cout <<    temp  << " ";
            }
            temp = (int) _dataSet(idx,_width*i + _width-1 );

            std::cout <<    temp  << "\n";

        }
        
    }
    
    void savePGM(int idx, std::string filename){
        std::ofstream fout (filename.c_str());
        
        if (!fout.is_open())
        {
            std::cerr << "Can't open output file"  << filename << std::endl;
            
        }
        
        // write the header
        
        fout << "P2\n" << _height << " " << _width << "\n255\n";
        int temp  ;
        // write the data
        for(int i = 0; i < _height; i++){
            for(int j = 0 ; j < _width-1; j++){

                temp = (int) _dataSet( idx, _width*i + j );
                fout <<    255*temp  << " ";
            }
            temp = (int) _dataSet( idx, _width*i + _width-1 );
            fout <<    255*temp  << "\n";
        }
        
        
        // close the stream
        fout.close();
    }
};

#endif /* mnist_h */
