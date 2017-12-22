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
 *  utils.h
 *
 *  Created by Nader on 10/12/2017.
 */

#ifndef utils_h
#define utils_h

//#include <boost/random.hpp>
//#include <boost/math/distributions.hpp>
//#include "boost/random/mersenne_twister.hpp"
//#include "boost/random/uniform_int_distribution.hpp"
#include "Eigen/Dense"
#include <cmath>
#include <random>
#include <time.h>


//Abstract class for activation function
template<class T>
class AtvFunc{
public:
    virtual T operator()(T val) = 0;
    virtual Eigen::VectorXd operator()(Eigen::VectorXd& val) = 0;
};

//Implementation of sigmoid function
template<class T>
class Sigmoid : public AtvFunc<T>
{
public:
    T operator()(T val) { return 1.0 / (1.0 + std::exp(-val));}
    Eigen::VectorXd operator()(Eigen::VectorXd& val){
        int N = (int) val.size();
        
        Eigen::VectorXd _result(N);
        
        for(int i = 0; i < N; i++){
            _result[i] = 1.0 / (1.0 + std::exp(-val[i]));
        }
        return _result;
    }
};

//Abstract class for random variable 
template< class T>
class RandVar{
public:
    virtual T operator()(int, double) = 0;
    virtual Eigen::VectorXd operator()(int, Eigen::VectorXd&) = 0;
};

//Implementation of binomial random variable
template<class T>
class Binomial : public RandVar<T>
{
private:
    std::random_device rd;
    typedef std::mt19937 MyRng;
    MyRng rng;
public:
    Binomial():rng(rd()){
        
        rng.seed(::time(NULL));
    }
    T operator ()(int n_trails, double prob){
        
        if(prob < 0 || prob > 1) return 0;
        
        int c = 0;
        double r;
        
        for(int i=0; i<n_trails; i++) {
            r = rand() / (RAND_MAX + 1.0);
            if (r < prob) c++;
        }
        
        return c;
    }
    
    
    Eigen::VectorXd operator ()(int n_trails, Eigen::VectorXd& prob){
        
        int N = (int) prob.size();
        Eigen::VectorXd _result(N);
 
        for(int i = 0; i < N; i++){
            std::binomial_distribution<int> distribution(n_trails,prob[i]);

            _result[i] = distribution(rng);
        }
        
        return _result;
    }
    
    
    
};


//Not used, yet
int randomIdx(int Nsize, int batchSize){
   // boost::random::uniform_int_distribution<> randPos(1,Nsize - batchSize);
    int result = 0.0;
    return result;
}


#endif /* utils_h */
